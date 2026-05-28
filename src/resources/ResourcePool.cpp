//
// See ResourcePool.hpp.
//

#include <array>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/resources/ResourcePool.hpp>

namespace veng
{
namespace
{
// A two-stage layout transition + clear, recorded into the caller's command buffer. Used by
// constant_image to bring its single copy from Undefined -> TransferDst -> clear -> ShaderReadOnly
// in one immediate submit.
void clear_to_shader_readonly(vk::CommandBuffer cmd, vk::Image image, std::array<float, 4> clear)
{
	const auto range =
		vk::ImageSubresourceRange().setAspectMask(vk::ImageAspectFlagBits::eColor).setLevelCount(1).setLayerCount(1);
	const auto to_dst = vk::ImageMemoryBarrier2()
							.setSrcStageMask(vk::PipelineStageFlagBits2::eTopOfPipe)
							.setSrcAccessMask(vk::AccessFlagBits2::eNone)
							.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
							.setDstAccessMask(vk::AccessFlagBits2::eTransferWrite)
							.setOldLayout(vk::ImageLayout::eUndefined)
							.setNewLayout(vk::ImageLayout::eTransferDstOptimal)
							.setImage(image)
							.setSubresourceRange(range);
	cmd.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(to_dst));
	cmd.clearColorImage(image, vk::ImageLayout::eTransferDstOptimal, vk::ClearColorValue(clear), range);
	const auto to_read = vk::ImageMemoryBarrier2()
							 .setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
							 .setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite)
							 .setDstStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
							 .setDstAccessMask(vk::AccessFlagBits2::eShaderSampledRead)
							 .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
							 .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
							 .setImage(image)
							 .setSubresourceRange(range);
	cmd.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(to_read));
}
} // namespace

ResourcePool::ResourcePool(vk::Device device, vma::Allocator allocator, std::size_t frames_in_flight) noexcept
	: m_device(device)
	, m_allocator(allocator)
	, m_frames_in_flight(static_cast<std::int64_t>(frames_in_flight == 0 ? 1 : frames_in_flight))
{
}

void ResourcePool::destroy() noexcept
{
	// Image/Buffer are RAII (each owns its allocator+device handles), so clearing the copy
	// lists frees every GPU resource regardless of this pool's own (non-owning) handles.
	m_images.clear();
	m_buffers.clear();
	m_device	= nullptr;
	m_allocator = nullptr;
}

ResourcePool::ResourcePool(ResourcePool&& other) noexcept
	: m_device(std::exchange(other.m_device, nullptr))
	, m_allocator(std::exchange(other.m_allocator, nullptr))
	, m_frames_in_flight(other.m_frames_in_flight)
	, m_frame(other.m_frame)
	, m_images(std::move(other.m_images))
	, m_buffers(std::move(other.m_buffers))
{
}

ResourcePool& ResourcePool::operator=(ResourcePool&& other) noexcept
{
	if (this != &other)
	{
		destroy();
		m_device		   = std::exchange(other.m_device, nullptr);
		m_allocator		   = std::exchange(other.m_allocator, nullptr);
		m_frames_in_flight = other.m_frames_in_flight;
		m_frame			   = other.m_frame;
		m_images		   = std::move(other.m_images);
		m_buffers		   = std::move(other.m_buffers);
	}
	return *this;
}

ResourcePool::~ResourcePool()
{
	destroy();
}

ImageId ResourcePool::declare_image(vk::Format format, vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect)
{
	m_images.push_back(ImageResource{.format = format, .usage = usage, .aspect = aspect});
	return static_cast<ImageId>(m_images.size() - 1);
}

BufferId ResourcePool::declare_buffer(vk::BufferUsageFlags usage)
{
	m_buffers.push_back(BufferResource{.usage = usage});
	return static_cast<BufferId>(m_buffers.size() - 1);
}

std::expected<Image*, vk::Result> ResourcePool::acquire_image(ImageId id, vk::Extent2D extent)
{
	ImageResource& res = m_images[id];

	// A resize reallocates: the producer drives this only when its screen-size edge changed, and
	// the driver waits the device idle on resize, so dropping the old copies is safe.
	if (res.extent != extent)
	{
		res.copies.clear();
		res.extent	= extent;
		res.current = NONE;
	}

	// Reuse the first copy whose last touch (write OR read) is older than the in-flight window;
	// such a copy has retired on the GPU and no in-flight frame still references it.
	std::size_t pick = NONE;
	for (std::size_t i = 0; i < res.copies.size(); ++i)
	{
		if (res.copies[i]->last_use <= retired_through())
		{
			pick = i;
			break;
		}
	}
	if (pick == NONE)
	{
		auto image = Image::create(m_allocator, m_device, extent, res.format, res.usage, res.aspect);
		if (!image.has_value())
		{
			return std::unexpected(image.error());
		}
		res.copies.push_back(std::make_unique<Copy<Image>>(std::move(image.value())));
		pick = res.copies.size() - 1;
	}

	res.copies[pick]->last_use = m_frame;
	res.current				   = pick;
	return &res.copies[pick]->resource;
}

Image* ResourcePool::read_image(ImageId id) noexcept
{
	ImageResource& res = m_images[id];
	if (res.current == NONE)
	{
		return nullptr;
	}
	res.copies[res.current]->last_use = m_frame; // retain while this frame is in flight
	return &res.copies[res.current]->resource;
}

void ResourcePool::touch(ImageId id) noexcept
{
	if (id >= m_images.size())
	{
		return;
	}
	ImageResource& res = m_images[id];
	if (res.is_constant || res.current == NONE)
	{
		return; // constants are never recycled, so stamping their last_use is pointless (and
				// would actually *expose* them to reuse since the initial sentinel is INT64_MAX).
	}
	res.copies[res.current]->last_use = m_frame;
}

std::expected<gpu::ImageRef, vk::Result> ResourcePool::constant_image(const Context& ctx, vk::Extent2D extent,
																	  vk::Format format, std::array<float, 4> clear)
{
	const vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	const ImageId			  id	= declare_image(format, usage);

	ImageResource& res = m_images[id];
	res.is_constant	   = true;
	res.extent		   = extent;
	auto image		   = Image::create(m_allocator, m_device, extent, format, usage);
	if (!image.has_value())
	{
		return std::unexpected(image.error());
	}
	res.copies.push_back(std::make_unique<Copy<Image>>(std::move(image.value())));
	res.copies.back()->last_use = INT64_MAX; // never recycled
	res.current					= 0;

	Image&			 img = res.copies.back()->resource;
	const vk::Result init =
		ctx.immediate_submit([&](vk::CommandBuffer cmd) { clear_to_shader_readonly(cmd, img.image(), clear); });
	if (init != vk::Result::eSuccess)
	{
		return std::unexpected(init);
	}
	return gpu::ImageRef{.image = img.image(), .view = img.view(), .extent = extent, .format = format, .pool_id = id};
}

std::expected<Buffer*, vk::Result> ResourcePool::acquire_buffer(BufferId id, vk::DeviceSize size)
{
	BufferResource& res = m_buffers[id];
	if (res.size != size)
	{
		res.copies.clear();
		res.size	= size;
		res.current = NONE;
	}

	std::size_t pick = NONE;
	for (std::size_t i = 0; i < res.copies.size(); ++i)
	{
		if (res.copies[i]->last_use <= retired_through())
		{
			pick = i;
			break;
		}
	}
	if (pick == NONE)
	{
		auto buffer = Buffer::create(m_allocator, size, res.usage, vma::MemoryUsage::eAuto,
									 vma::AllocationCreateFlagBits::eMapped |
										 vma::AllocationCreateFlagBits::eHostAccessSequentialWrite);
		if (!buffer.has_value() || buffer->mapped() == nullptr)
		{
			return std::unexpected(buffer.has_value() ? vk::Result::eErrorMemoryMapFailed : buffer.error());
		}
		res.copies.push_back(std::make_unique<Copy<Buffer>>(std::move(buffer.value())));
		pick = res.copies.size() - 1;
	}

	res.copies[pick]->last_use = m_frame;
	res.current				   = pick;
	return &res.copies[pick]->resource;
}

std::size_t ResourcePool::image_copy_count(ImageId id) const noexcept
{
	return m_images[id].copies.size();
}

std::size_t ResourcePool::buffer_copy_count(BufferId id) const noexcept
{
	return m_buffers[id].copies.size();
}
} // namespace veng
