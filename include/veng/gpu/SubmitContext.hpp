//
// Post-submit context for sink nodes ([[pass-draw-redesign]]). After the driver ends the frame's
// command buffer and submits it on the graphics queue, every `GpuNode` in the executed plan gets
// one call to `on_submitted` with this context — that is where `vkQueuePresentKHR` is issued by
// `PresentNode` (a queue op that must follow submit, not a recordable command), where a future
// screenshot or video-encode sink would enqueue its post-submit work, and where any "had to happen
// *after* submit" behaviour lives. Most nodes don't override the hook; the few that do (sinks)
// receive only what they need here: graph data resolution (to read their input edges) and the
// Vulkan `Context` (for the queue), plus the current frame slot.
//
// Decoupling submission from any particular node is what makes multi-sink frames (present +
// screenshot in the same frame) and headless renderers (no swapchain → no `PresentNode`) fall out
// of the same machinery, instead of being special cases of "the unique terminal that ends the
// shared command buffer".
//

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
/// Subset of `SwapchainManager::Frame` that sinks needing to issue a present queue op care
/// about: which acquired image to present and the render-finished semaphore the presentation
/// engine waits on. The acquire-wait semaphore + the slot's in-flight fence stay with the
/// `FrameExecutor` (they are submit's wait/fence, not present's). An empty optional on
/// `SubmitContext` means "this frame has no swapchain present" (headless test path, or a
/// future graph with only a screenshot/encode sink).
struct PresentFrame
{
	std::uint32_t image_index	 = 0;
	vk::Semaphore present_signal = {}; // render-finished — the submit signals it, present waits on it
};

class SubmitContext
{
	 public:
	SubmitContext(const graph::Graph& graph, const Context& context, std::size_t frame_slot,
				  std::optional<PresentFrame> present = std::nullopt) noexcept
		: m_graph(&graph)
		, m_context(&context)
		, m_frame_slot(frame_slot)
		, m_present(present)
	{
	}

	[[nodiscard]] graph::Data*	 data(graph::DataHandle handle) const { return m_graph->get_data(handle); }
	[[nodiscard]] const Context& context() const noexcept { return *m_context; }
	[[nodiscard]] std::size_t	 frame_slot() const noexcept { return m_frame_slot; }

	/// Present-frame info, populated by the driver/`FrameExecutor` when a swapchain image was
	/// acquired for this frame; `nullopt` when the frame is headless or otherwise has no
	/// present sink. `PresentNode::on_submitted` reads this to issue `vkQueuePresentKHR`.
	[[nodiscard]] const std::optional<PresentFrame>& present_frame() const noexcept { return m_present; }

	 private:
	const graph::Graph*			m_graph;
	const Context*				m_context;
	std::size_t					m_frame_slot;
	std::optional<PresentFrame> m_present;
};
} // namespace veng::gpu

#endif // VENG_SUBMITCONTEXT_HPP
