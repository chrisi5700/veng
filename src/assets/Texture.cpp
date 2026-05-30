//
// See Texture.hpp and review.md item 3.
//

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stb_image.h>
#include <utility>
#include <veng/assets/Texture.hpp>
#include <veng/context/Context.hpp>
#include <veng/managers/CommandManager.hpp>
#include <veng/resources/Buffer.hpp>

namespace veng::assets
{
namespace
{
// Number of mip levels for a `width x height` image: the full pyramid down to 1x1.
std::uint32_t mip_count(std::uint32_t width, std::uint32_t height)
{
	const auto largest = static_cast<float>(std::max(width, height));
	return static_cast<std::uint32_t>(std::floor(std::log2(largest))) + 1U;
}

// Generate the mip chain in-place by successively blitting level i-1 (downscaled) into level i,
// then leave every level in SHADER_READ_ONLY. The image arrives with level 0 populated and every
// level in TRANSFER_DST. Standard blit-down pyramid (requires the format to support linear blit,
// which R8G8B8A8 UNORM/SRGB universally do).
void generate_mips(vk::CommandBuffer cmd, vk::Image image, std::uint32_t width, std::uint32_t height,
				   std::uint32_t levels)
{
	auto mip_width	= static_cast<std::int32_t>(width);
	auto mip_height = static_cast<std::int32_t>(height);

	for (std::uint32_t level = 1; level < levels; ++level)
	{
		// Source level (level-1): TRANSFER_DST -> TRANSFER_SRC so it can be the blit source.
		CommandManager::image_barrier_range(cmd, image, vk::ImageLayout::eTransferDstOptimal,
											vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eTransfer,
											vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer,
											vk::AccessFlagBits2::eTransferRead, level - 1, 1);

		const std::int32_t next_width  = mip_width > 1 ? mip_width / 2 : 1;
		const std::int32_t next_height = mip_height > 1 ? mip_height / 2 : 1;
		const auto		   layers	   = [](std::uint32_t l)
		{
			return vk::ImageSubresourceLayers()
				.setAspectMask(vk::ImageAspectFlagBits::eColor)
				.setMipLevel(l)
				.setLayerCount(1);
		};
		const auto blit = vk::ImageBlit()
							  .setSrcSubresource(layers(level - 1))
							  .setSrcOffsets({vk::Offset3D{0, 0, 0}, vk::Offset3D{mip_width, mip_height, 1}})
							  .setDstSubresource(layers(level))
							  .setDstOffsets({vk::Offset3D{0, 0, 0}, vk::Offset3D{next_width, next_height, 1}});
		cmd.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal, blit,
					  vk::Filter::eLinear);

		// Source level is done: TRANSFER_SRC -> SHADER_READ_ONLY.
		CommandManager::image_barrier_range(
			cmd, image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
			vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
			vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead, level - 1, 1);

		mip_width  = next_width;
		mip_height = next_height;
	}

	// The last level was only ever a blit destination: TRANSFER_DST -> SHADER_READ_ONLY.
	CommandManager::image_barrier_range(
		cmd, image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
		vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead, levels - 1, 1);
}
} // namespace

std::expected<Texture, TextureError> Texture::from_pixels(const Context& ctx, std::span<const std::byte> rgba8,
														  std::uint32_t width, std::uint32_t height,
														  ColorSpace color_space)
{
	if (width == 0 || height == 0 || rgba8.size() < static_cast<std::size_t>(width) * height * 4)
	{
		return std::unexpected(TextureError::DecodeFailed);
	}

	const std::uint32_t levels = mip_count(width, height);
	const vk::Format format = color_space == ColorSpace::Srgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;

	// Sampled + transfer-dst (upload + blit destination) + transfer-src (mip blit source).
	auto image = Image::create(ctx.allocator(), ctx.device(), vk::Extent2D{width, height}, format,
							   vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
								   vk::ImageUsageFlagBits::eTransferSrc,
							   vk::ImageAspectFlagBits::eColor, levels);
	if (!image.has_value())
	{
		return std::unexpected(TextureError::GpuAllocation);
	}

	auto staging = Buffer::create(
		ctx.allocator(), rgba8.size(), vk::BufferUsageFlagBits::eTransferSrc, vma::MemoryUsage::eAuto,
		vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessSequentialWrite);
	if (!staging.has_value() || staging->mapped() == nullptr)
	{
		return std::unexpected(TextureError::GpuAllocation);
	}
	std::memcpy(staging->mapped(), rgba8.data(), rgba8.size());

	const vk::Image	 handle = image->image();
	const vk::Result result = ctx.immediate_submit(
		[&](vk::CommandBuffer cmd)
		{
			// Whole image: UNDEFINED -> TRANSFER_DST (all levels), then upload level 0.
			CommandManager::image_barrier_range(
				cmd, handle, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
				vk::PipelineStageFlagBits2::eTopOfPipe, vk::AccessFlagBits2::eNone,
				vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, 0, levels);

			const auto region =
				vk::BufferImageCopy()
					.setImageSubresource(
						vk::ImageSubresourceLayers().setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1))
					.setImageExtent(vk::Extent3D{width, height, 1});
			cmd.copyBufferToImage(staging->buffer(), handle, vk::ImageLayout::eTransferDstOptimal, region);

			generate_mips(cmd, handle, width, height, levels);
		});
	if (result != vk::Result::eSuccess)
	{
		return std::unexpected(TextureError::Upload);
	}

	return Texture(std::move(image.value()));
}

std::expected<Texture, TextureError> Texture::from_file(const Context& ctx, const std::string& path,
														ColorSpace color_space)
{
	int		 width	  = 0;
	int		 height	  = 0;
	int		 channels = 0;
	stbi_uc* pixels	  = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
	if (pixels == nullptr)
	{
		return std::unexpected(TextureError::FileUnreadable);
	}

	const auto bytes = std::span<const std::byte>(reinterpret_cast<const std::byte*>(pixels),
												  static_cast<std::size_t>(width) * height * 4);
	auto	   texture =
		from_pixels(ctx, bytes, static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), color_space);
	stbi_image_free(pixels);
	return texture;
}

std::expected<Texture, TextureError> Texture::from_encoded(const Context& ctx, std::span<const std::byte> encoded,
														   ColorSpace color_space)
{
	int		 width	  = 0;
	int		 height	  = 0;
	int		 channels = 0;
	stbi_uc* pixels =
		stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(encoded.data()), static_cast<int>(encoded.size()),
							  &width, &height, &channels, STBI_rgb_alpha);
	if (pixels == nullptr)
	{
		return std::unexpected(TextureError::DecodeFailed);
	}

	const auto bytes = std::span<const std::byte>(reinterpret_cast<const std::byte*>(pixels),
												  static_cast<std::size_t>(width) * height * 4);
	auto	   texture =
		from_pixels(ctx, bytes, static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), color_space);
	stbi_image_free(pixels);
	return texture;
}
} // namespace veng::assets
