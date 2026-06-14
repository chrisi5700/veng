/**
 * @file
 * @author chris
 * @brief Implementation of @ref veng::SwapchainManager.
 * @ingroup managers
 */

#include <utility>
#include <veng/managers/SwapchainManager.hpp>
#include <VkBootstrap.h>

namespace veng
{
namespace
{
constexpr vk::Result result_of(VkResult raw)
{
	return raw != VK_SUCCESS ? static_cast<vk::Result>(raw) : vk::Result::eErrorInitializationFailed;
}
} // namespace

SwapchainManager::SwapchainManager(const Context& context, std::size_t frames_in_flight) noexcept
	: m_device(context.device())
	, m_rhi(&context.rhi())
	, m_physical(context.physical_device())
	, m_surface(context.surface())
	, m_graphics_family(context.queue_indices().graphics)
	, m_frames_in_flight(frames_in_flight)
	, m_swapchain(nullptr)
{
}

std::expected<SwapchainManager, vk::Result> SwapchainManager::create(const Context& context, vk::Extent2D extent,
																	 std::size_t frames_in_flight)
{
	SwapchainManager manager(context, frames_in_flight);
	if (auto built = manager.build_swapchain(extent, nullptr); !built.has_value())
	{
		return std::unexpected(built.error());
	}
	for (std::size_t i = 0; i < frames_in_flight; ++i)
	{
		const auto semaphore = manager.m_device.createSemaphore({});
		if (semaphore.result != vk::Result::eSuccess)
		{
			manager.destroy();
			return std::unexpected(semaphore.result);
		}
		manager.m_image_available.push_back(semaphore.value);

		// Created signaled so the first acquire of each slot does not block.
		const auto fence =
			manager.m_device.createFence(vk::FenceCreateInfo().setFlags(vk::FenceCreateFlagBits::eSignaled));
		if (fence.result != vk::Result::eSuccess)
		{
			manager.destroy();
			return std::unexpected(fence.result);
		}
		manager.m_in_flight.push_back(fence.value);
	}
	return manager;
}

std::expected<void, vk::Result> SwapchainManager::build_swapchain(vk::Extent2D extent, vk::SwapchainKHR old)
{
	vkb::SwapchainBuilder builder(m_physical, m_device, m_surface, m_graphics_family, m_graphics_family);
	auto				  result =
		builder.set_desired_extent(extent.width, extent.height)
			.set_desired_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR) // uncapped; FIFO fallback
			// sRGB surface so the final write into the swapchain image hardware-encodes linear ->
			// sRGB for the display. A blit into an _SRGB image encodes its
			// (linear) source, so a scene rendered in linear light presents correctly. Was
			// _UNORM, which presented linear values un-encoded — too dark once lighting is linear.
			.set_desired_format(VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
			.add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT) // we blit into the swapchain image
			.set_old_swapchain(static_cast<VkSwapchainKHR>(old))
			.build();
	if (!result)
	{
		return std::unexpected(result_of(result.vk_result()));
	}
	vkb::Swapchain swapchain = result.value();

	auto images = swapchain.get_images();
	if (!images)
	{
		vkb::destroy_swapchain(swapchain);
		return std::unexpected(result_of(images.vk_result()));
	}

	// Render-finished semaphores are per swapchain image (a present may still be using
	// one), so rebuild them whenever the image set changes.
	for (const vk::Semaphore semaphore : m_render_finished)
	{
		m_device.destroySemaphore(semaphore);
	}
	m_render_finished.clear();
	for (const rhi::SemaphoreHandle handle : m_render_finished_handles)
	{
		m_rhi->release_semaphore(handle);
	}
	m_render_finished_handles.clear();

	m_swapchain = vk::SwapchainKHR(swapchain.swapchain);
	m_format	= vk::Format(swapchain.image_format);
	m_extent	= vk::Extent2D{swapchain.extent.width, swapchain.extent.height};
	m_images.clear();
	for (const VkImage handle : images.value())
	{
		m_images.emplace_back(handle);
	}

	// Register each swapchain image as an RHI texture so the frame's ImageRef flows as a handle. The
	// swapchain has no views (blit/present work on the image), so register a null view. Old handles
	// (from the previous swapchain on a rebuild) are released first.
	for (const rhi::TextureHandle handle : m_texture_handles)
	{
		m_rhi->release_texture(handle);
	}
	m_texture_handles.clear();
	for (const vk::Image image : m_images)
	{
		m_texture_handles.push_back(m_rhi->register_texture(image, vk::ImageView{}));
	}
	for (std::size_t i = 0; i < m_images.size(); ++i)
	{
		const auto semaphore = m_device.createSemaphore({});
		if (semaphore.result != vk::Result::eSuccess)
		{
			return std::unexpected(semaphore.result);
		}
		m_render_finished.push_back(semaphore.value);
		m_render_finished_handles.push_back(m_rhi->register_semaphore(semaphore.value));
	}
	return {};
}

std::expected<std::optional<SwapchainManager::Frame>, vk::Result> SwapchainManager::acquire(std::size_t frame_slot)
{
	const std::size_t slot		= frame_slot % m_frames_in_flight;
	const vk::Fence	  in_flight = m_in_flight[slot];

	// Wait out the previous frame on this slot, then reset the fence for this one. Folding
	// this into acquire is what lets the caller never see a fence: after acquire returns,
	// the slot's command pool is safe to recycle.
	if (m_device.waitForFences(in_flight, vk::True, UINT64_MAX) != vk::Result::eSuccess)
	{
		return std::unexpected(vk::Result::eErrorDeviceLost);
	}

	const vk::Semaphore available = m_image_available[slot];
	const auto			acquired  = m_device.acquireNextImageKHR(m_swapchain, UINT64_MAX, available, nullptr);
	if (acquired.result == vk::Result::eErrorOutOfDateKHR)
	{
		return std::optional<Frame>{}; // leave the fence signaled; nothing was submitted
	}
	if (acquired.result != vk::Result::eSuccess && acquired.result != vk::Result::eSuboptimalKHR)
	{
		return std::unexpected(acquired.result);
	}
	// Reset only once we are committed to submitting this frame (which will re-signal it).
	if (m_device.resetFences(in_flight) != vk::Result::eSuccess)
	{
		return std::unexpected(vk::Result::eErrorDeviceLost);
	}
	return std::optional<Frame>{Frame{acquired.value, available, m_render_finished[acquired.value], in_flight}};
}

std::expected<bool, vk::Result> SwapchainManager::present(vk::Queue queue, std::uint32_t image_index,
														  rhi::SemaphoreHandle wait)
{
	const vk::Semaphore wait_sem = m_rhi->semaphore(wait);
	const auto			info =
		vk::PresentInfoKHR().setWaitSemaphores(wait_sem).setSwapchains(m_swapchain).setImageIndices(image_index);
	const vk::Result result = queue.presentKHR(info);
	if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR)
	{
		return true;
	}
	if (result != vk::Result::eSuccess)
	{
		return std::unexpected(result);
	}
	return false;
}

std::expected<void, vk::Result> SwapchainManager::rebuild(vk::Extent2D extent)
{
	const vk::SwapchainKHR old	 = m_swapchain;
	auto				   built = build_swapchain(extent, old);
	if (built.has_value() && old)
	{
		m_device.destroySwapchainKHR(old);
	}
	return built;
}

void SwapchainManager::destroy() noexcept
{
	if (!m_device)
	{
		return;
	}
	for (const rhi::TextureHandle handle : m_texture_handles)
	{
		m_rhi->release_texture(handle);
	}
	m_texture_handles.clear();
	for (const rhi::SemaphoreHandle handle : m_render_finished_handles)
	{
		m_rhi->release_semaphore(handle);
	}
	m_render_finished_handles.clear();
	for (const vk::Semaphore semaphore : m_render_finished)
	{
		if (semaphore)
		{
			m_device.destroySemaphore(semaphore);
		}
	}
	for (const vk::Semaphore semaphore : m_image_available)
	{
		if (semaphore)
		{
			m_device.destroySemaphore(semaphore);
		}
	}
	for (const vk::Fence fence : m_in_flight)
	{
		if (fence)
		{
			m_device.destroyFence(fence);
		}
	}
	m_render_finished.clear();
	m_image_available.clear();
	m_in_flight.clear();
	if (m_swapchain)
	{
		m_device.destroySwapchainKHR(m_swapchain);
	}
	m_swapchain = nullptr;
	m_device	= nullptr; // mark emptied so a moved-from / destroyed manager is inert
}

SwapchainManager::SwapchainManager(SwapchainManager&& other) noexcept
	: m_device(std::exchange(other.m_device, nullptr))
	, m_rhi(other.m_rhi)
	, m_physical(other.m_physical)
	, m_surface(other.m_surface)
	, m_graphics_family(other.m_graphics_family)
	, m_frames_in_flight(other.m_frames_in_flight)
	, m_swapchain(std::exchange(other.m_swapchain, nullptr))
	, m_images(std::move(other.m_images))
	, m_texture_handles(std::move(other.m_texture_handles))
	, m_format(other.m_format)
	, m_extent(other.m_extent)
	, m_image_available(std::move(other.m_image_available))
	, m_render_finished(std::move(other.m_render_finished))
	, m_render_finished_handles(std::move(other.m_render_finished_handles))
	, m_in_flight(std::move(other.m_in_flight))
{
	other.m_images.clear();
	other.m_texture_handles.clear();
	other.m_image_available.clear();
	other.m_render_finished.clear();
	other.m_render_finished_handles.clear();
	other.m_in_flight.clear();
}

SwapchainManager& SwapchainManager::operator=(SwapchainManager&& other) noexcept
{
	if (this != &other)
	{
		destroy();
		m_device				  = std::exchange(other.m_device, nullptr);
		m_rhi					  = other.m_rhi;
		m_physical				  = other.m_physical;
		m_surface				  = other.m_surface;
		m_graphics_family		  = other.m_graphics_family;
		m_frames_in_flight		  = other.m_frames_in_flight;
		m_swapchain				  = std::exchange(other.m_swapchain, nullptr);
		m_images				  = std::move(other.m_images);
		m_texture_handles		  = std::move(other.m_texture_handles);
		m_format				  = other.m_format;
		m_extent				  = other.m_extent;
		m_image_available		  = std::move(other.m_image_available);
		m_render_finished		  = std::move(other.m_render_finished);
		m_render_finished_handles = std::move(other.m_render_finished_handles);
		m_in_flight				  = std::move(other.m_in_flight);
		other.m_images.clear();
		other.m_texture_handles.clear();
		other.m_image_available.clear();
		other.m_render_finished.clear();
		other.m_render_finished_handles.clear();
		other.m_in_flight.clear();
	}
	return *this;
}

SwapchainManager::~SwapchainManager()
{
	destroy();
}
} // namespace veng
