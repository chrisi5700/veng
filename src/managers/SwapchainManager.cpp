//
// Created by chris on 5/25/26.
//
// See SwapchainManager.hpp and design.md §L4.
//

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
	}
	return manager;
}

std::expected<void, vk::Result> SwapchainManager::build_swapchain(vk::Extent2D extent, vk::SwapchainKHR old)
{
	vkb::SwapchainBuilder builder(m_physical, m_device, m_surface, m_graphics_family, m_graphics_family);
	auto				  result =
		builder.set_desired_extent(extent.width, extent.height)
			.set_desired_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR) // uncapped; FIFO fallback
			.set_desired_format(VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
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

	m_swapchain = vk::SwapchainKHR(swapchain.swapchain);
	m_format	= vk::Format(swapchain.image_format);
	m_extent	= vk::Extent2D{swapchain.extent.width, swapchain.extent.height};
	m_images.clear();
	for (const VkImage handle : images.value())
	{
		m_images.emplace_back(handle);
	}
	for (std::size_t i = 0; i < m_images.size(); ++i)
	{
		const auto semaphore = m_device.createSemaphore({});
		if (semaphore.result != vk::Result::eSuccess)
		{
			return std::unexpected(semaphore.result);
		}
		m_render_finished.push_back(semaphore.value);
	}
	return {};
}

std::expected<std::optional<SwapchainManager::Frame>, vk::Result> SwapchainManager::acquire(std::size_t frame_slot)
{
	const vk::Semaphore available = m_image_available[frame_slot % m_frames_in_flight];
	const auto			acquired  = m_device.acquireNextImageKHR(m_swapchain, UINT64_MAX, available, nullptr);
	if (acquired.result == vk::Result::eErrorOutOfDateKHR)
	{
		return std::optional<Frame>{};
	}
	if (acquired.result != vk::Result::eSuccess && acquired.result != vk::Result::eSuboptimalKHR)
	{
		return std::unexpected(acquired.result);
	}
	return std::optional<Frame>{Frame{acquired.value, available, m_render_finished[acquired.value]}};
}

std::expected<bool, vk::Result> SwapchainManager::present(vk::Queue queue, std::uint32_t image_index,
														  vk::Semaphore wait)
{
	const auto info =
		vk::PresentInfoKHR().setWaitSemaphores(wait).setSwapchains(m_swapchain).setImageIndices(image_index);
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
	m_render_finished.clear();
	m_image_available.clear();
	if (m_swapchain)
	{
		m_device.destroySwapchainKHR(m_swapchain);
	}
	m_swapchain = nullptr;
	m_device	= nullptr; // mark emptied so a moved-from / destroyed manager is inert
}

SwapchainManager::SwapchainManager(SwapchainManager&& other) noexcept
	: m_device(std::exchange(other.m_device, nullptr))
	, m_physical(other.m_physical)
	, m_surface(other.m_surface)
	, m_graphics_family(other.m_graphics_family)
	, m_frames_in_flight(other.m_frames_in_flight)
	, m_swapchain(std::exchange(other.m_swapchain, nullptr))
	, m_images(std::move(other.m_images))
	, m_format(other.m_format)
	, m_extent(other.m_extent)
	, m_image_available(std::move(other.m_image_available))
	, m_render_finished(std::move(other.m_render_finished))
{
	other.m_images.clear();
	other.m_image_available.clear();
	other.m_render_finished.clear();
}

SwapchainManager& SwapchainManager::operator=(SwapchainManager&& other) noexcept
{
	if (this != &other)
	{
		destroy();
		m_device		   = std::exchange(other.m_device, nullptr);
		m_physical		   = other.m_physical;
		m_surface		   = other.m_surface;
		m_graphics_family  = other.m_graphics_family;
		m_frames_in_flight = other.m_frames_in_flight;
		m_swapchain		   = std::exchange(other.m_swapchain, nullptr);
		m_images		   = std::move(other.m_images);
		m_format		   = other.m_format;
		m_extent		   = other.m_extent;
		m_image_available  = std::move(other.m_image_available);
		m_render_finished  = std::move(other.m_render_finished);
		other.m_images.clear();
		other.m_image_available.clear();
		other.m_render_finished.clear();
	}
	return *this;
}

SwapchainManager::~SwapchainManager()
{
	destroy();
}
} // namespace veng
