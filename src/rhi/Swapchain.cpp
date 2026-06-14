/**
 * @file
 * @author chris
 * @brief @ref veng::rhi::Swapchain implementation — the RHI-level windowed present loop.
 * @ingroup rhi
 */

#include <utility>
#include <veng/context/Context.hpp>
#include <veng/rhi/CommandEncoder.hpp>
#include <veng/rhi/Convert.hpp>
#include <veng/rhi/Device.hpp>
#include <veng/rhi/Swapchain.hpp>
#include <VkBootstrap.h>

namespace veng::rhi
{
namespace
{
constexpr vk::Result result_of(VkResult raw)
{
	return raw != VK_SUCCESS ? static_cast<vk::Result>(raw) : vk::Result::eErrorInitializationFailed;
}
} // namespace

Swapchain::Swapchain(const Context& context) noexcept
	: m_device(context.device())
	, m_physical(context.physical_device())
	, m_surface(context.surface())
	, m_queue(context.graphics_queue())
	, m_graphics_family(context.queue_indices().graphics)
	, m_rhi(&context.rhi())
	, m_swapchain(nullptr)
{
}

std::expected<Swapchain, Error> Swapchain::create(const Context& context, Extent2D extent)
{
	Swapchain swapchain(context);
	if (auto built = swapchain.build(extent, nullptr); !built.has_value())
	{
		return std::unexpected(to_error(built.error()));
	}
	// One acquire semaphore + one in-flight fence: single-buffered, the simplest correct loop. The
	// fence starts signalled so the first acquire does not block on a frame that never ran.
	const auto available = swapchain.m_device.createSemaphore({});
	if (available.result != vk::Result::eSuccess)
	{
		swapchain.destroy();
		return std::unexpected(to_error(available.result));
	}
	swapchain.m_image_available = available.value;
	const auto fence =
		swapchain.m_device.createFence(vk::FenceCreateInfo().setFlags(vk::FenceCreateFlagBits::eSignaled));
	if (fence.result != vk::Result::eSuccess)
	{
		swapchain.destroy();
		return std::unexpected(to_error(fence.result));
	}
	swapchain.m_frame_fence = fence.value;
	return swapchain;
}

std::expected<void, vk::Result> Swapchain::build(Extent2D extent, vk::SwapchainKHR old)
{
	// Build the new swapchain first (using `old` to recycle resources) so a failure leaves the
	// previous images/views/semaphores intact for the caller to keep presenting with.
	vkb::SwapchainBuilder builder(m_physical, m_device, m_surface, m_graphics_family, m_graphics_family);
	auto				  result =
		builder.set_desired_extent(extent.width, extent.height)
			.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR) // vsync; always supported
			// sRGB surface: the fragment shader's linear output is hardware-encoded to sRGB for the
			// display, matching the engine's own swapchain format.
			.set_desired_format(VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
			.add_image_usage_flags(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) // rendered straight into
			.set_old_swapchain(static_cast<VkSwapchainKHR>(old))
			.build();
	if (!result)
	{
		return std::unexpected(result_of(result.vk_result()));
	}
	vkb::Swapchain swapchain = result.value();
	auto		   images	 = swapchain.get_images();
	if (!images)
	{
		vkb::destroy_swapchain(swapchain);
		return std::unexpected(result_of(images.vk_result()));
	}

	// The new swapchain is committed — release the previous frame's per-image objects (views,
	// registered handles, present semaphores). The old `vk::SwapchainKHR` itself is destroyed by the
	// caller (recreate) after this returns, since `set_old_swapchain` still references it.
	for (const rhi::TextureHandle handle : m_textures)
	{
		m_rhi->release_texture(handle);
	}
	m_textures.clear();
	for (const vk::ImageView view : m_views)
	{
		m_device.destroyImageView(view);
	}
	m_views.clear();
	for (const vk::Semaphore semaphore : m_render_finished)
	{
		m_device.destroySemaphore(semaphore);
	}
	m_render_finished.clear();

	m_swapchain = vk::SwapchainKHR(swapchain.swapchain);
	m_format	= to_rhi(vk::Format(swapchain.image_format));
	m_extent	= Extent2D{swapchain.extent.width, swapchain.extent.height};

	// Register each image with a view (begin_rendering needs the view) and a present semaphore.
	for (const VkImage raw : images.value())
	{
		const vk::Image image(raw);
		const auto		view_info = vk::ImageViewCreateInfo()
									 .setImage(image)
									 .setViewType(vk::ImageViewType::e2D)
									 .setFormat(to_vk(m_format))
									 .setSubresourceRange(vk::ImageSubresourceRange()
															  .setAspectMask(vk::ImageAspectFlagBits::eColor)
															  .setLevelCount(1)
															  .setLayerCount(1));
		const auto view = m_device.createImageView(view_info);
		if (view.result != vk::Result::eSuccess)
		{
			return std::unexpected(view.result);
		}
		m_views.push_back(view.value);
		m_textures.push_back(m_rhi->register_texture(image, view.value));

		const auto semaphore = m_device.createSemaphore({});
		if (semaphore.result != vk::Result::eSuccess)
		{
			return std::unexpected(semaphore.result);
		}
		m_render_finished.push_back(semaphore.value);
	}
	return {};
}

std::expected<std::optional<Swapchain::Frame>, Error> Swapchain::acquire()
{
	// Wait out the previous frame on this single in-flight slot before the caller re-records.
	if (m_device.waitForFences(m_frame_fence, vk::True, UINT64_MAX) != vk::Result::eSuccess)
	{
		return std::unexpected(to_error(vk::Result::eErrorDeviceLost));
	}
	const auto acquired = m_device.acquireNextImageKHR(m_swapchain, UINT64_MAX, m_image_available, nullptr);
	if (acquired.result == vk::Result::eErrorOutOfDateKHR)
	{
		return std::optional<Frame>{}; // leave the fence signalled; nothing was submitted
	}
	if (acquired.result != vk::Result::eSuccess && acquired.result != vk::Result::eSuboptimalKHR)
	{
		return std::unexpected(to_error(acquired.result));
	}
	// Reset only now that we are committed to submitting (which re-signals the fence).
	if (m_device.resetFences(m_frame_fence) != vk::Result::eSuccess)
	{
		return std::unexpected(to_error(vk::Result::eErrorDeviceLost));
	}
	const std::uint32_t index = acquired.value;
	return std::optional<Frame>{Frame{.target = m_textures[index], .image_index = index, .extent = m_extent}};
}

std::expected<bool, Error> Swapchain::present(CommandEncoder& enc, const Frame& frame)
{
	const vk::CommandBuffer cmd = enc.vk();
	if (const vk::Result result = cmd.end(); result != vk::Result::eSuccess)
	{
		return std::unexpected(to_error(result));
	}
	// Wait on the acquire at color-attachment-output (the first stage that touches the image), signal
	// the per-image present semaphore, and fence the submit so the next acquire knows it has retired.
	const vk::Semaphore			 wait		= m_image_available;
	const vk::Semaphore			 signal		= m_render_finished[frame.image_index];
	const vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	const auto					 submit		= vk::SubmitInfo()
							  .setWaitSemaphores(wait)
							  .setWaitDstStageMask(wait_stage)
							  .setCommandBuffers(cmd)
							  .setSignalSemaphores(signal);
	if (const vk::Result result = m_queue.submit(submit, m_frame_fence); result != vk::Result::eSuccess)
	{
		return std::unexpected(to_error(result));
	}
	const auto info =
		vk::PresentInfoKHR().setWaitSemaphores(signal).setSwapchains(m_swapchain).setImageIndices(frame.image_index);
	const vk::Result result = m_queue.presentKHR(info);
	if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR)
	{
		return true;
	}
	if (result != vk::Result::eSuccess)
	{
		return std::unexpected(to_error(result));
	}
	return false;
}

std::expected<void, Error> Swapchain::recreate()
{
	(void)m_device.waitIdle(); // the simplest barrier before swapping the image set out
	const Extent2D extent = surface_extent(m_extent);
	if (extent.width == 0 || extent.height == 0)
	{
		return {}; // minimised — keep the old swapchain and try again next frame
	}
	const vk::SwapchainKHR old	= m_swapchain;
	auto				   built = build(extent, old);
	if (!built.has_value())
	{
		return std::unexpected(to_error(built.error()));
	}
	if (old)
	{
		m_device.destroySwapchainKHR(old);
	}
	return {};
}

Extent2D Swapchain::surface_extent(Extent2D fallback) const
{
	const auto caps = m_physical.getSurfaceCapabilitiesKHR(m_surface);
	if (caps.result != vk::Result::eSuccess)
	{
		return fallback;
	}
	const vk::Extent2D current = caps.value.currentExtent;
	// 0xFFFFFFFF means "the surface defers to the swapchain extent" — keep the last size we used.
	if (current.width == UINT32_MAX)
	{
		return fallback;
	}
	return Extent2D{current.width, current.height};
}

void Swapchain::destroy() noexcept
{
	if (!m_device)
	{
		return;
	}
	for (const rhi::TextureHandle handle : m_textures)
	{
		m_rhi->release_texture(handle);
	}
	m_textures.clear();
	for (const vk::ImageView view : m_views)
	{
		if (view)
		{
			m_device.destroyImageView(view);
		}
	}
	m_views.clear();
	for (const vk::Semaphore semaphore : m_render_finished)
	{
		if (semaphore)
		{
			m_device.destroySemaphore(semaphore);
		}
	}
	m_render_finished.clear();
	if (m_image_available)
	{
		m_device.destroySemaphore(m_image_available);
		m_image_available = nullptr;
	}
	if (m_frame_fence)
	{
		m_device.destroyFence(m_frame_fence);
		m_frame_fence = nullptr;
	}
	if (m_swapchain)
	{
		m_device.destroySwapchainKHR(m_swapchain);
		m_swapchain = nullptr;
	}
	m_device = nullptr; // mark emptied so a moved-from / destroyed swapchain is inert
}

Swapchain::Swapchain(Swapchain&& other) noexcept
	: m_device(std::exchange(other.m_device, nullptr))
	, m_physical(other.m_physical)
	, m_surface(other.m_surface)
	, m_queue(other.m_queue)
	, m_graphics_family(other.m_graphics_family)
	, m_rhi(other.m_rhi)
	, m_swapchain(std::exchange(other.m_swapchain, nullptr))
	, m_views(std::move(other.m_views))
	, m_textures(std::move(other.m_textures))
	, m_render_finished(std::move(other.m_render_finished))
	, m_image_available(std::exchange(other.m_image_available, nullptr))
	, m_frame_fence(std::exchange(other.m_frame_fence, nullptr))
	, m_format(other.m_format)
	, m_extent(other.m_extent)
{
	other.m_views.clear();
	other.m_textures.clear();
	other.m_render_finished.clear();
}

Swapchain& Swapchain::operator=(Swapchain&& other) noexcept
{
	if (this != &other)
	{
		destroy();
		m_device		  = std::exchange(other.m_device, nullptr);
		m_physical		  = other.m_physical;
		m_surface		  = other.m_surface;
		m_queue			  = other.m_queue;
		m_graphics_family = other.m_graphics_family;
		m_rhi			  = other.m_rhi;
		m_swapchain		  = std::exchange(other.m_swapchain, nullptr);
		m_views			  = std::move(other.m_views);
		m_textures		  = std::move(other.m_textures);
		m_render_finished = std::move(other.m_render_finished);
		m_image_available = std::exchange(other.m_image_available, nullptr);
		m_frame_fence	  = std::exchange(other.m_frame_fence, nullptr);
		m_format		  = other.m_format;
		m_extent		  = other.m_extent;
		other.m_views.clear();
		other.m_textures.clear();
		other.m_render_finished.clear();
	}
	return *this;
}

Swapchain::~Swapchain()
{
	destroy();
}
} // namespace veng::rhi
