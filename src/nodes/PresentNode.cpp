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
	const auto* presented = dynamic_cast<graph::ValueData<gpu::ImageRef>*>(ctx.data(m_input));
	if (presented == nullptr)
	{
		return;
	}
	const gpu::ImageRef image = presented->value();
	if (!image.image || !image.present_signal)
	{
		return;
	}
	// vkQueuePresentKHR is a queue op (not recordable), so it must run after submit — here, on
	// the post-submit hook. The render-finished semaphore the submit signalled gates the
	// presentation engine; the acquire_wait + in_flight fence were used by the submit itself.
	auto presented_ok = m_swap->present(ctx.context().graphics_queue(), image.index, image.present_signal);
	m_out_of_date	  = presented_ok.has_value() ? presented_ok.value() : true;
	++m_present_count;
}
} // namespace veng::nodes
