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
#include <veng/context/Context.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/Graph.hpp>

namespace veng::gpu
{
class SubmitContext
{
	 public:
	SubmitContext(const graph::Graph& graph, const Context& context, std::size_t frame_slot) noexcept
		: m_graph(&graph)
		, m_context(&context)
		, m_frame_slot(frame_slot)
	{
	}

	[[nodiscard]] graph::Data*	 data(graph::DataHandle handle) const { return m_graph->get_data(handle); }
	[[nodiscard]] const Context& context() const noexcept { return *m_context; }
	[[nodiscard]] std::size_t	 frame_slot() const noexcept { return m_frame_slot; }

	 private:
	const graph::Graph* m_graph;
	const Context*		m_context;
	std::size_t			m_frame_slot;
};
} // namespace veng::gpu

#endif // VENG_SUBMITCONTEXT_HPP
