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
	const auto* presented = dynamic_cast<graph::ValueData<gpu::ImageRef>*>(ctx.data(m_input));
	if (presented == nullptr)
	{
		return std::unexpected(graph::ExecError::MISSING_INPUT);
	}
	const gpu::ImageRef image = presented->value();
	if (!image.image || !image.acquire_wait || !image.present_signal || !image.in_flight)
	{
		return std::unexpected(graph::ExecError::NODE_FAILED);
	}

	// Close the frame: the blit left the image in PRESENT_SRC, so all GPU work is now
	// recorded. End the buffer and submit it — the acquire semaphore gates the blit (the
	// swapchain image's first and only use here, a transfer), render-finished signals the
	// presentation engine.
	const vk::CommandBuffer cmd = ctx.command_buffer();
	if (cmd.end() != vk::Result::eSuccess)
	{
		return std::unexpected(graph::ExecError::NODE_FAILED);
	}

	const vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eTransfer;
	const auto					 submit		= vk::SubmitInfo()
												  .setWaitSemaphores(image.acquire_wait)
												  .setWaitDstStageMask(wait_stage)
												  .setCommandBuffers(cmd)
												  .setSignalSemaphores(image.present_signal);
	if (ctx.context().graphics_queue().submit(submit, image.in_flight) != vk::Result::eSuccess)
	{
		return std::unexpected(graph::ExecError::NODE_FAILED);
	}

	auto presented_ok = m_swap->present(ctx.context().graphics_queue(), image.index, image.present_signal);
	if (!presented_ok.has_value())
	{
		return std::unexpected(graph::ExecError::NODE_FAILED);
	}
	m_out_of_date = presented_ok.value();
	++m_present_count;
	return true;
}
} // namespace veng::nodes
