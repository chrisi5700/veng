//
// Created by chris on 5/25/26.
//
// See PresentNode.hpp and design.md §L4.
//

#include <veng/gpu/ImageRef.hpp>
#include <veng/managers/QueueKind.hpp>
#include <veng/nodes/PresentNode.hpp>
#include <veng/rendergraph/data/Data.hpp>

namespace veng::nodes
{
PresentNode::PresentNode(SwapchainManager& swap, graph::DataHandle presented_image, graph::DataHandle output) noexcept
	: m_swap(&swap)
	, m_input(presented_image)
	, m_output(output)
{
}

std::expected<bool, graph::ExecError> PresentNode::record(gpu::GpuExecContext& ctx)
{
	// No GPU work to record — the blit upstream already wrote the swapchain image and left it
	// in PRESENT_SRC. The driver/executor ends + submits the shared command buffer (submission
	// is no longer a node's job), and on_submitted issues the present queue op. Returning true
	// stamps the frame-done output each frame, like a normal producer.
	(void)ctx;
	return true;
}

void PresentNode::on_submitted(gpu::SubmitContext& ctx) noexcept
{
	// Per-frame present info comes from the driver/FrameExecutor through SubmitContext — it no
	// longer rides on the input ImageRef edge (image refs don't leak queue plumbing anymore).
	// The input edge stays purely for ordering: it makes the planner schedule us after the blit.
	const auto& present = ctx.present_frame();
	if (!present.has_value() || !present->present_signal)
	{
		return;
	}
	auto presented_ok = m_swap->present(ctx.context().graphics_queue(), present->image_index, present->present_signal);
	m_out_of_date	  = presented_ok.has_value() ? presented_ok.value() : true;
	++m_present_count;
}
} // namespace veng::nodes
