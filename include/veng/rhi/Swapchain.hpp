/**
 * @file
 * @author chris
 * @brief The RHI windowed swapchain — acquire/present in RHI vocabulary, no Vulkan at the call site.
 *
 * @ref veng::rhi::Swapchain is the present half of the RHI: it owns the `VkSwapchainKHR` over the
 * engine @ref veng::Context's surface, registers each swapchain image (with a view) as an RHI @ref
 * TextureHandle, and exposes a frame loop in pure engine vocabulary — @ref acquire hands back the
 * next image as a handle, the caller records into it through a @ref CommandEncoder, and @ref present
 * submits and queues it for display. A windowed renderer therefore never names a `vk::SwapchainKHR`,
 * `vk::Semaphore`, or queue operation — see `example/rhi_triangle`.
 *
 * It is deliberately single-buffered (one acquire semaphore, one in-flight fence): the simplest
 * correct present loop, prioritising clarity over throughput. The engine's own render path uses the
 * N-buffered @ref veng::SwapchainManager (which sits a layer up, in `veng_managers`); this is the
 * RHI-level primitive that one could be rebuilt on top of, the same way @ref veng::Image/`Buffer`
 * relate to @ref Device::create_texture / @ref Device::create_buffer.
 *
 * Created from a windowed @ref veng::Context (one with a surface). RAII, move-only; destroyed before
 * the `vk::Device`/surface it borrows.
 *
 * @ingroup rhi
 */

#ifndef VENG_RHI_SWAPCHAIN_HPP
#define VENG_RHI_SWAPCHAIN_HPP

#include <cstdint>
#include <expected>
#include <optional>
#include <vector>
#include <veng/rhi/Enums.hpp>
#include <veng/rhi/Error.hpp>
#include <veng/rhi/Handles.hpp>
#include <vulkan/vulkan.hpp>

namespace veng
{
class Context;
}

namespace veng::rhi
{
class Device;
class CommandEncoder;

/**
 * @brief A windowed swapchain presented in RHI vocabulary.
 *
 * Non-copyable, movable. Build with @ref create from a windowed @ref veng::Context, then drive the
 * per-frame loop with @ref acquire → record → @ref present, recreating on @ref Frame loss.
 *
 * @ingroup rhi
 */
class Swapchain
{
	 public:
	/// @brief The image to render into this frame, returned by @ref acquire.
	struct Frame
	{
		TextureHandle target;		///< The swapchain image (image + view) to render into and present.
		std::uint32_t image_index;	///< Its index in the swapchain (selects the present-signal semaphore).
		Extent2D	  extent;		///< The current swapchain extent (use it for the render area + viewport).
	};

	/**
	 * @brief Create a swapchain over @p context's surface at (approximately) @p extent.
	 * @param context A windowed @ref veng::Context (its `surface()` must be non-null).
	 * @param extent  The desired initial extent in pixels (clamped to the surface's capabilities).
	 * @return A ready swapchain, or the failing @ref Error.
	 */
	[[nodiscard]] static std::expected<Swapchain, Error> create(const Context& context, Extent2D extent);

	Swapchain(const Swapchain&)			   = delete;
	Swapchain& operator=(const Swapchain&) = delete;
	Swapchain(Swapchain&& other) noexcept;
	Swapchain& operator=(Swapchain&& other) noexcept;
	~Swapchain();

	/// @return The swapchain image format — feed this to the pipeline's `color_formats`.
	[[nodiscard]] Format format() const noexcept { return m_format; }
	/// @return The current swapchain extent in pixels.
	[[nodiscard]] Extent2D extent() const noexcept { return m_extent; }

	/**
	 * @brief Acquire the next image to render into.
	 *
	 * First waits out the previous frame (the in-flight fence) so the command buffer the caller is
	 * about to re-record is free. A `nullopt` result means the swapchain is out of date (e.g. the
	 * window resized) — call @ref recreate and skip the frame.
	 *
	 * @return The @ref Frame to render into, `nullopt` if out of date, or an @ref Error on failure.
	 */
	[[nodiscard]] std::expected<std::optional<Frame>, Error> acquire();

	/**
	 * @brief End @p enc, submit it (waiting on the acquire, signalling present-ready), and queue
	 *        @p frame for display.
	 *
	 * The recorded commands must have left @p frame's target in @ref TextureUsage::PRESENT.
	 *
	 * @return `true` if the swapchain is now out of date (call @ref recreate), `false` on clean
	 *         success, or an @ref Error on failure.
	 */
	[[nodiscard]] std::expected<bool, Error> present(CommandEncoder& enc, const Frame& frame);

	/**
	 * @brief Rebuild the swapchain at the surface's current size (re-queried from its capabilities).
	 * @pre Safe to call any time; it waits the device idle first.
	 * @return `void` on success (a no-op while the window is minimised to a zero extent), or an @ref Error.
	 */
	[[nodiscard]] std::expected<void, Error> recreate();

	 private:
	explicit Swapchain(const Context& context) noexcept;
	[[nodiscard]] std::expected<void, vk::Result> build(Extent2D extent, vk::SwapchainKHR old);
	[[nodiscard]] Extent2D						   surface_extent(Extent2D fallback) const;
	void										   destroy() noexcept;

	vk::Device		   m_device;
	vk::PhysicalDevice m_physical;
	vk::SurfaceKHR	   m_surface; ///< Borrowed from the @ref veng::Context; not destroyed here.
	vk::Queue		   m_queue;	  ///< The graphics+present queue submit/present run on.
	std::uint32_t	   m_graphics_family{};
	Device*			   m_rhi = nullptr; ///< Registry the swapchain images register their handles with.

	vk::SwapchainKHR				m_swapchain;
	std::vector<vk::ImageView>		m_views;		   ///< One per swapchain image; owned here.
	std::vector<rhi::TextureHandle> m_textures;		   ///< One per image; re-registered on rebuild.
	std::vector<vk::Semaphore>		m_render_finished; ///< One per image; present waits on it.
	vk::Semaphore					m_image_available; ///< Signalled by acquire, waited by the submit.
	vk::Fence						m_frame_fence;	   ///< Signalled by the submit; acquire waits + resets it.
	Format							m_format = Format::UNDEFINED;
	Extent2D						m_extent{};
};
} // namespace veng::rhi

#endif // VENG_RHI_SWAPCHAIN_HPP
