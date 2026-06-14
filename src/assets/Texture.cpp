/**
 * @file
 * @author chris
 * @brief @ref veng::assets::Texture implementation: stb_image decoding, mip-chain generation, and
 *        GPU staging upload.
 * @ingroup assets
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stb_image.h>
#include <utility>
#include <veng/assets/Texture.hpp>
#include <veng/context/Context.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/rhi/CommandEncoder.hpp>

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
	const rhi::Format	format = color_space == ColorSpace::Srgb ? rhi::Format::RGBA8_SRGB : rhi::Format::RGBA8_UNORM;

	// Sampled + transfer-dst (upload + blit destination) + transfer-src (mip blit source).
	auto image = Image::create(ctx.rhi(), rhi::Extent2D{width, height}, format,
							   rhi::TextureUsageFlags::SAMPLED | rhi::TextureUsageFlags::TRANSFER_DST |
								   rhi::TextureUsageFlags::TRANSFER_SRC,
							   levels);
	if (!image.has_value())
	{
		return std::unexpected(TextureError::GpuAllocation);
	}

	auto staging =
		Buffer::create(ctx.rhi(), rgba8.size(), rhi::BufferUsageFlags::TRANSFER_SRC, rhi::MemoryAccess::HOST_VISIBLE);
	if (!staging.has_value() || staging->mapped() == nullptr)
	{
		return std::unexpected(TextureError::GpuAllocation);
	}
	std::memcpy(staging->mapped(), rgba8.data(), rgba8.size());

	// Upload level 0 + generate the mip chain, recorded and submitted entirely in RHI vocabulary —
	// the encoder owns the layout transitions, buffer copy, and blit-down pyramid.
	rhi::CommandEncoder enc = ctx.rhi().begin_commands();
	enc.upload_mipped_texture(staging->handle(), image->handle(), rhi::Extent2D{width, height}, levels);
	if (!ctx.rhi().submit(enc).has_value())
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
