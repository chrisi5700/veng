//
// Created by chris on 5/25/26.
//
// L2 manager — the windowed swapchain (design.md §L4 swapchain & present). Owns the
// VkSwapchainKHR over the Context's surface, its images, and the binary acquire /
// render-finished semaphores. The acquired swapchain image is the per-frame target the
// PresentNode blits into; `acquire` returns the image index plus the semaphore the
// submit must wait on, `present` queues it for display. RAII, move-only.
//
// The surface itself is owned by the Context (it must be destroyed before the
// instance); this manager only borrows it.
//

#ifndef VENG_SWAPCHAINMANAGER_HPP
#define VENG_SWAPCHAINMANAGER_HPP

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <vector>
#include <veng/context/Context.hpp>
#include <vulkan/vulkan.hpp>

namespace veng
{
class SwapchainManager
{
	 public:
	struct Frame
	{
		std::uint32_t image_index;
		vk::Semaphore image_available; // the submit waits on this (per frame-in-flight slot)
		vk::Semaphore render_finished; // the submit signals, present waits (per swapchain image)
	};

	[[nodiscard]] static std::expected<SwapchainManager, vk::Result> create(const Context& context, vk::Extent2D extent,
																			std::size_t frames_in_flight = 1);

	SwapchainManager(const SwapchainManager&)			 = delete;
	SwapchainManager& operator=(const SwapchainManager&) = delete;
	SwapchainManager(SwapchainManager&& other) noexcept;
	SwapchainManager& operator=(SwapchainManager&& other) noexcept;
	~SwapchainManager();

	[[nodiscard]] vk::Extent2D extent() const noexcept { return m_extent; }
	[[nodiscard]] vk::Format   format() const noexcept { return m_format; }
	[[nodiscard]] vk::Image	   image(std::uint32_t index) const noexcept { return m_images[index]; }

	/// Acquire the next image for `frame_slot`. A `nullopt` Frame means the swapchain is
	/// out of date and the caller should `rebuild`.
	[[nodiscard]] std::expected<std::optional<Frame>, vk::Result> acquire(std::size_t frame_slot);

	/// Present `image_index`, waiting on `wait`. Returns true if the swapchain is now
	/// out of date / suboptimal (rebuild).
	[[nodiscard]] std::expected<bool, vk::Result> present(vk::Queue queue, std::uint32_t image_index,
														  vk::Semaphore wait);

	/// Recreate the swapchain at `extent`. Call after the device is idle.
	[[nodiscard]] std::expected<void, vk::Result> rebuild(vk::Extent2D extent);

	 private:
	SwapchainManager(const Context& context, std::size_t frames_in_flight) noexcept;
	[[nodiscard]] std::expected<void, vk::Result> build_swapchain(vk::Extent2D extent, vk::SwapchainKHR old);
	void										  destroy() noexcept;

	vk::Device				   m_device;
	vk::PhysicalDevice		   m_physical;
	vk::SurfaceKHR			   m_surface; // borrowed from Context, not destroyed here
	std::uint32_t			   m_graphics_family{};
	std::size_t				   m_frames_in_flight{};
	vk::SwapchainKHR		   m_swapchain;
	std::vector<vk::Image>	   m_images;
	vk::Format				   m_format = vk::Format::eUndefined;
	vk::Extent2D			   m_extent{};
	std::vector<vk::Semaphore> m_image_available; // size == frames_in_flight
	std::vector<vk::Semaphore> m_render_finished; // size == image count
};
} // namespace veng

#endif // VENG_SWAPCHAINMANAGER_HPP
