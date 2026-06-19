/**
 * @file
 * @author chris
 * @brief L2 manager for the windowed Vulkan swapchain.
 *
 * Owns the `VkSwapchainKHR` over the @ref veng::Context's surface, its images, and the binary
 * acquire/render-finished semaphores. The acquired swapchain image is the per-frame target
 * that `PresentNode` blits into: @ref veng::SwapchainManager::acquire "acquire" returns the image
 * index plus the semaphore the submit must wait on, and @ref veng::SwapchainManager::present
 * "present" queues it for display. RAII, move-only.
 *
 * The surface itself is owned by the @ref veng::Context (it must be destroyed before the
 * instance); this manager only borrows it.
 *
 * @ingroup managers
 */

#ifndef VENG_SWAPCHAINMANAGER_HPP
#define VENG_SWAPCHAINMANAGER_HPP

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <vector>
#include <veng/context/Context.hpp>
#include <veng/rhi/Convert.hpp>
#include <vulkan/vulkan.hpp>

namespace veng
{
/**
 * @brief Manages the `VkSwapchainKHR`, its images, and the per-frame synchronisation
 *        semaphores and fences.
 *
 * @ingroup managers
 * @see FrameExecutor
 * @see Context
 */
class SwapchainManager
{
	 public:
	/**
	 * @brief Per-frame synchronisation bundle returned by @ref acquire.
	 * @ingroup managers
	 */
	struct Frame
	{
		std::uint32_t image_index;
		vk::Semaphore image_available; ///< The submit waits on this (one per frame-in-flight slot).
		vk::Semaphore render_finished; ///< The submit signals this; present waits on it (one per swapchain image).
		vk::Fence	  in_flight;	   ///< The submit signals this; `acquire` waited + reset it (one per slot).
	};

	/**
	 * @brief Create the swapchain manager and build the initial swapchain.
	 * @param context          The engine @ref veng::Context owning the surface.
	 * @param extent           The initial swapchain extent in pixels.
	 * @param frames_in_flight Number of in-flight frame slots (defaults to 1).
	 * @return A ready manager, or a `vk::Result` error on failure.
	 */
	[[nodiscard]] static std::expected<SwapchainManager, vk::Result> create(const Context& context,
																			rhi::Extent2D  extent,
																			std::size_t	   frames_in_flight = 1);

	SwapchainManager(const SwapchainManager&)			 = delete;
	SwapchainManager& operator=(const SwapchainManager&) = delete;
	SwapchainManager(SwapchainManager&& other) noexcept;
	SwapchainManager& operator=(SwapchainManager&& other) noexcept;
	~SwapchainManager();

	/** @return The current swapchain extent in pixels. */
	[[nodiscard]] rhi::Extent2D extent() const noexcept { return rhi::to_rhi(m_extent); }
	/**
	 * @brief Monotonic counter bumped every time the swapchain is (re)built.
	 *
	 * A rebuild replaces every image with a fresh RHI texture handle, so any graph source fed the
	 * swapchain image is genuinely stale afterwards. A driver (or @ref FrameExecutor) compares this
	 * against the last value it acted on to know it must re-dirty that source — the reactive signal
	 * that a rebuilt (and so far never-drawn, black) image needs a frame even when the scene is quiet.
	 */
	[[nodiscard]] std::uint64_t generation() const noexcept { return m_generation; }
	/** @return The swapchain surface format. */
	[[nodiscard]] rhi::Format format() const noexcept { return rhi::to_rhi(m_format); }
	/**
	 * @brief Return the swapchain `vk::Image` at @p index.
	 * @param index Swapchain image index as returned by `acquire`.
	 */
	[[nodiscard]] vk::Image image(std::uint32_t index) const noexcept { return m_images[index]; }

	/// @brief The RHI texture handle for the swapchain image at @p index (resolves to its `vk::Image`).
	[[nodiscard]] rhi::TextureHandle texture_handle(std::uint32_t index) const noexcept
	{
		return m_texture_handles[index];
	}

	/// @brief The RHI handle for the render-finished semaphore of the swapchain image at @p index — the
	///        present-signal token the frame hands a present sink (no raw `vk::Semaphore` crosses out).
	[[nodiscard]] rhi::SemaphoreHandle render_finished_handle(std::uint32_t index) const noexcept
	{
		return m_render_finished_handles[index];
	}

	/**
	 * @brief Acquire the next swapchain image for @p frame_slot.
	 *
	 * First waits on (and resets) that slot's in-flight fence so the previous frame using
	 * the slot has fully retired before the caller reuses its command pool — the caller never
	 * touches a fence itself. A `nullopt` @ref Frame means the swapchain is out of date and
	 * the caller should call @ref rebuild.
	 *
	 * @param frame_slot The in-flight slot index for this frame.
	 * @return The synchronisation @ref Frame, or `nullopt` if out-of-date, or a `vk::Result`
	 *         error on a hard failure.
	 */
	[[nodiscard]] std::expected<std::optional<Frame>, vk::Result> acquire(std::size_t frame_slot);

	/**
	 * @brief Queue @p image_index for display, waiting on @p wait.
	 * @param queue       The graphics queue to present on.
	 * @param image_index Index of the swapchain image to present.
	 * @param wait        Semaphore the presentation engine waits on before scanning out.
	 * @return `true` if the swapchain is now out of date or suboptimal (caller should
	 *         rebuild); `false` on clean success; unexpected `vk::Result` on hard failure.
	 */
	[[nodiscard]] std::expected<bool, vk::Result> present(vk::Queue queue, std::uint32_t image_index,
														  rhi::SemaphoreHandle wait);

	/**
	 * @brief Recreate the swapchain at @p extent.
	 * @pre The device is idle (no frames in flight).
	 * @param extent The new swapchain extent in pixels.
	 * @return `void` on success, or a `vk::Result` error.
	 */
	[[nodiscard]] std::expected<void, vk::Result> rebuild(rhi::Extent2D extent);

	 private:
	SwapchainManager(const Context& context, std::size_t frames_in_flight) noexcept;
	[[nodiscard]] std::expected<void, vk::Result> build_swapchain(vk::Extent2D extent, vk::SwapchainKHR old);
	void										  destroy() noexcept;

	vk::Device						m_device;
	rhi::Device*					m_rhi = nullptr; ///< Registry the swapchain images register their handles with.
	vk::PhysicalDevice				m_physical;
	vk::SurfaceKHR					m_surface; ///< Borrowed from @ref veng::Context; not destroyed here.
	std::uint32_t					m_graphics_family{};
	std::size_t						m_frames_in_flight{};
	vk::SwapchainKHR				m_swapchain;
	std::uint64_t					m_generation{}; ///< Bumped on every successful (re)build; see @ref generation.
	std::vector<vk::Image>			m_images;
	std::vector<rhi::TextureHandle> m_texture_handles; ///< One per swapchain image; re-registered on rebuild.
	vk::Format						m_format = vk::Format::eUndefined;
	vk::Extent2D					m_extent{};
	std::vector<vk::Semaphore>		m_image_available; ///< One per frame-in-flight slot.
	std::vector<vk::Semaphore>		m_render_finished; ///< One per swapchain image.
	std::vector<rhi::SemaphoreHandle>
						   m_render_finished_handles; ///< RHI handles for m_render_finished; rebuilt on resize.
	std::vector<vk::Fence> m_in_flight;				  ///< One per frame-in-flight slot; created signalled.
};
} // namespace veng

#endif // VENG_SWAPCHAINMANAGER_HPP
