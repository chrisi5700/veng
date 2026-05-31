/**
 * @file
 * @author chris
 * @brief Post-submit context supplied to sink nodes after frame command-buffer submission.
 *
 * After the driver ends the frame's command buffer and submits it on the graphics queue, every
 * @ref veng::gpu::Sink in the executed plan gets one call to `on_submitted` with this context — that is
 * where `vkQueuePresentKHR` is issued by `PresentNode` (a queue op that must follow submit,
 * not a recordable command), where a future screenshot or video-encode sink would enqueue its
 * post-submit work, and where any behaviour that must happen after submit lives. Most nodes do
 * not override the hook; the few that do (sinks) receive only what they need here: graph data
 * resolution (to read their input edges), the Vulkan @ref veng::Context (for the queue), and the
 * current frame slot.
 *
 * Decoupling submission from any particular node is what makes multi-sink frames (present plus
 * screenshot in the same frame) and headless renderers (no swapchain, no `PresentNode`) fall
 * out of the same machinery, instead of being special cases of "the unique terminal that ends
 * the shared command buffer".
 *
 * @ingroup gpu_handles
 */

#ifndef VENG_SUBMITCONTEXT_HPP
#define VENG_SUBMITCONTEXT_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <veng/context/Context.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::gpu
{
/**
 * @brief Subset of swapchain-frame info that sinks issuing a present queue op care about.
 *
 * Holds the acquired image index and the render-finished semaphore the presentation engine
 * waits on. The acquire-wait semaphore and the slot's in-flight fence stay with the
 * `FrameExecutor` (they are submit's wait/fence, not present's). An absent optional on
 * @ref veng::gpu::SubmitContext means "this frame has no swapchain present" (headless test path, or a
 * graph with only a screenshot or encode sink).
 *
 * @ingroup gpu_handles
 * @see SubmitContext
 */
struct PresentFrame
{
	std::uint32_t image_index	 = 0;  ///< Index of the acquired swapchain image to present.
	vk::Semaphore present_signal = {}; ///< Render-finished semaphore: submit signals it, present waits on it.
};

/**
 * @brief Context passed to @ref veng::gpu::Sink::on_submitted and @ref veng::gpu::Sink::on_retired.
 *
 * Provides graph data resolution (to read input edges), the engine @ref veng::Context (for queue
 * access), the current frame slot, and optional swapchain present info. Only the sinks that
 * need it (e.g. `PresentNode`) inspect the @ref veng::gpu::PresentFrame.
 *
 * @ingroup gpu_handles
 * @see Sink
 * @see PresentFrame
 * @see GpuExecContext
 */
class SubmitContext
{
	 public:
	/**
	 * @brief Construct a post-submit context for one frame.
	 * @param graph      The graph whose data handles are resolved during sink callbacks.
	 * @param context    The engine-wide Vulkan context (device, queues).
	 * @param frame_slot The in-flight frame slot index (0..N-1).
	 * @param present    Swapchain present info, or `std::nullopt` for headless frames.
	 */
	SubmitContext(const graph::Graph& graph, const Context& context, std::size_t frame_slot,
				  std::optional<PresentFrame> present = std::nullopt) noexcept
		: m_graph(&graph)
		, m_context(&context)
		, m_frame_slot(frame_slot)
		, m_present(present)
	{
	}

	/**
	 * @brief Resolve a data handle to the underlying `graph::Data` node.
	 * @param handle The data handle to look up.
	 * @return Pointer to the data node, or `nullptr` if the handle is invalid.
	 */
	[[nodiscard]] graph::Data* data(graph::DataHandle handle) const { return m_graph->get_data(handle); }

	/// @brief Access the engine-wide Vulkan context.
	/// @return Reference to the @ref veng::Context owning the device and queues.
	[[nodiscard]] const Context& context() const noexcept { return *m_context; }

	/// @brief The in-flight frame slot index for this frame (0..N-1).
	/// @return The slot index used to select N-buffered resources.
	[[nodiscard]] std::size_t frame_slot() const noexcept { return m_frame_slot; }

	/**
	 * @brief Present-frame info for sinks that issue `vkQueuePresentKHR`.
	 *
	 * Populated by the driver when a swapchain image was acquired for this frame. `nullopt`
	 * when the frame is headless or otherwise has no present sink. `PresentNode::on_submitted`
	 * reads this to issue `vkQueuePresentKHR`.
	 *
	 * @return The @ref veng::gpu::PresentFrame for this frame, or `std::nullopt` if headless.
	 */
	[[nodiscard]] const std::optional<PresentFrame>& present_frame() const noexcept { return m_present; }

	 private:
	const graph::Graph*			m_graph;	  ///< Graph used for data handle resolution.
	const Context*				m_context;	  ///< Engine-wide Vulkan context.
	std::size_t					m_frame_slot; ///< In-flight slot index (0..N-1).
	std::optional<PresentFrame> m_present;	  ///< Swapchain present info, or `nullopt` if headless.
};
} // namespace veng::gpu

#endif // VENG_SUBMITCONTEXT_HPP
