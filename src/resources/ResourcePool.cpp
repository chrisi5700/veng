/**
 * @file
 * @author chris
 * @brief @ref veng::ResourcePool implementation — copy acquisition, barrier transitions, and retirement.
 * @ingroup resources
 */

#include <array>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/resources/ResourcePool.hpp>
#include <veng/rhi/Convert.hpp>

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

ResourcePool::ResourcePool(vk::Device device, rhi::Device& rhi, vma::Allocator allocator,
						   std::size_t frames_in_flight) noexcept
	: m_device(device)
	, m_rhi(&rhi)
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
	, m_rhi(std::exchange(other.m_rhi, nullptr))
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
		m_rhi			   = std::exchange(other.m_rhi, nullptr);
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

ImageId ResourcePool::declare_image(vk::Format format, vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect,
									vk::SampleCountFlagBits samples)
{
	m_images.push_back(ImageResource{.format = format, .usage = usage, .aspect = aspect, .sample_count = samples});
	return static_cast<ImageId>(m_images.size() - 1);
}

BufferId ResourcePool::declare_buffer(rhi::BufferUsageFlags usage)
{
	m_buffers.push_back(BufferResource{.usage = rhi::to_vk(usage)});
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
		auto image = Image::create(m_allocator, m_device, *m_rhi, extent, res.format, res.usage, res.aspect, 1,
								   res.sample_count);
		if (!image.has_value())
		{
			return std::unexpected(image.error());
		}
		res.copies.push_back(std::make_unique<ImageCopy>(std::move(image.value())));
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

void ResourcePool::touch_buffer(BufferId id) noexcept
{
	if (id >= m_buffers.size())
	{
		return;
	}
	BufferResource& res = m_buffers[id];
	if (res.current == NONE)
	{
		return;
	}
	res.copies[res.current]->last_use = m_frame;
}

void ResourcePool::purge_retired_buffers() noexcept
{
	std::erase_if(m_retiring_buffers, [retired = retired_through()](const std::unique_ptr<Copy<Buffer>>& copy)
				  { return copy->last_use <= retired; });
}

void ResourcePool::transition_image(ImageId id, vk::CommandBuffer cmd, vk::ImageLayout new_layout,
									vk::PipelineStageFlags2 new_stage, vk::AccessFlags2 new_access) noexcept
{
	if (id >= m_images.size())
	{
		return;
	}
	ImageResource& res = m_images[id];
	if (res.current == NONE)
	{
		return;
	}
	ImageCopy& copy = *res.copies[res.current];
	if (copy.current_layout == new_layout)
	{
		return; // already in the requested layout
	}
	const auto barrier = vk::ImageMemoryBarrier2()
							 .setSrcStageMask(copy.last_stage)
							 .setSrcAccessMask(copy.last_access)
							 .setDstStageMask(new_stage)
							 .setDstAccessMask(new_access)
							 .setOldLayout(copy.current_layout)
							 .setNewLayout(new_layout)
							 .setImage(copy.resource.image())
							 .setSubresourceRange(vk::ImageSubresourceRange()
													  .setAspectMask(res.aspect)
													  .setBaseMipLevel(0)
													  .setLevelCount(1)
													  .setBaseArrayLayer(0)
													  .setLayerCount(1));
	cmd.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(barrier));
	copy.current_layout = new_layout;
	copy.last_stage		= new_stage;
	copy.last_access	= new_access;
}

std::expected<gpu::ImageRef, vk::Result> ResourcePool::constant_image(const Context& ctx, vk::Extent2D extent,
																	  vk::Format format, std::array<float, 4> clear)
{
	const vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	const ImageId			  id	= declare_image(format, usage);

	ImageResource& res = m_images[id];
	res.is_constant	   = true;
	res.extent		   = extent;
	auto image		   = Image::create(m_allocator, m_device, *m_rhi, extent, format, usage);
	if (!image.has_value())
	{
		return std::unexpected(image.error());
	}
	res.copies.push_back(std::make_unique<ImageCopy>(std::move(image.value())));
	res.copies.back()->last_use = INT64_MAX; // never recycled
	res.current					= 0;

	Image&			 img = res.copies.back()->resource;
	const vk::Result init =
		ctx.immediate_submit([&](vk::CommandBuffer cmd) { clear_to_shader_readonly(cmd, img.image(), clear); });
	if (init != vk::Result::eSuccess)
	{
		return std::unexpected(init);
	}
	// Record the post-init layout so the auto-barrier path skips a no-op transition for the
	// constant on every sample (it lives in SHADER_READ_ONLY forever).
	res.copies.back()->current_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
	res.copies.back()->last_stage	  = vk::PipelineStageFlagBits2::eFragmentShader;
	res.copies.back()->last_access	  = vk::AccessFlagBits2::eShaderSampledRead;
	return gpu::ImageRef{
		.texture = img.handle(), .extent = rhi::to_rhi(extent), .format = rhi::to_rhi(format), .pool_id = id};
}

std::expected<Buffer*, vk::Result> ResourcePool::acquire_buffer(BufferId id, vk::DeviceSize size)
{
	// Free any resize graveyard entries that have now retired. This MUST run here, during node record,
	// not in begin_frame — begin_frame is called before the swapchain acquire waits the slot's
	// in-flight fence, so retired_through() is only accurate once we are past that wait (which we are
	// by the time any node records). Purging in begin_frame would destroy buffers a frame still
	// executing on the GPU references (VUID-vkDestroyBuffer-00922).
	purge_retired_buffers();

	BufferResource& res = m_buffers[id];
	if (res.size != size)
	{
		// A buffer resize (e.g. a cull's light-index list growing as the camera moves) is NOT gated by
		// a device-idle the way an image resize is, so an in-flight frame's descriptor set may still
		// reference these copies. Move them to the retirement graveyard instead of freeing them now;
		// purge_retired_buffers() (above) releases them once their last_use has aged past the window.
		for (auto& copy : res.copies)
		{
			m_retiring_buffers.push_back(std::move(copy));
		}
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
		auto buffer = Buffer::create(m_allocator, *m_rhi, size, res.usage, vma::MemoryUsage::eAuto,
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
