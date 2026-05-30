//
// Created by chris on 5/26/26.
//
// See BlitNode.hpp and design.md §L4.
//

#include <array>
#include <cstdint>
#include <veng/gpu/ImageRef.hpp>
#include <veng/managers/CommandManager.hpp>
#include <veng/nodes/BlitNode.hpp>
#include <veng/rendergraph/data/Data.hpp>

namespace veng::nodes
{
BlitNode::BlitNode(graph::DataHandle src, graph::DataHandle dst, graph::DataHandle output,
				   vk::ImageLayout final_layout) noexcept
	: m_inputs{src, dst}
	, m_output(output)
	, m_final_layout(final_layout)
{
}

std::vector<gpu::ImageUsage> BlitNode::image_usages(graph::ExecContext& ctx)
{
	// Read-side dependency for the executor: the source must be in TRANSFER_SRC before the blit
	// runs. The destination is the swapchain (or a test-owned image) — not pool-backed — and is
	// transitioned manually in record(), because nothing else in the engine tracks its layout.
	const auto* src = dynamic_cast<graph::ValueData<gpu::ImageRef>*>(ctx.data(m_inputs[0]));
	if (src == nullptr || src->value().pool_id == gpu::ImageRef::INVALID_POOL_ID)
	{
		return {};
	}
	return {gpu::ImageUsage{.id		= src->value().pool_id,
							.layout = vk::ImageLayout::eTransferSrcOptimal,
							.stage	= vk::PipelineStageFlagBits2::eTransfer,
							.access = vk::AccessFlagBits2::eTransferRead}};
}

std::expected<bool, graph::ExecError> BlitNode::record(gpu::GpuExecContext& ctx)
{
	const auto* src = dynamic_cast<graph::ValueData<gpu::ImageRef>*>(ctx.data(m_inputs[0]));
	const auto* dst = dynamic_cast<graph::ValueData<gpu::ImageRef>*>(ctx.data(m_inputs[1]));
	auto*		out = dynamic_cast<graph::ValueData<gpu::ImageRef>*>(ctx.data(m_output));
	if (src == nullptr || dst == nullptr || out == nullptr)
	{
		return std::unexpected(graph::ExecError::MISSING_INPUT);
	}
	const gpu::ImageRef source = src->value();
	const gpu::ImageRef target = dst->value();
	if (!source.image || !target.image)
	{
		return std::unexpected(graph::ExecError::NODE_FAILED);
	}

	// Retain the pooled source copy we read while this frame is in flight (the dst is the
	// swapchain / a test target — not pool-owned, so consume is a no-op for it).
	ctx.pool().consume(source);

	const vk::CommandBuffer cmd = ctx.command_buffer();

	// Target: undefined -> transfer dst. We overwrite the whole image, so its prior
	// contents and layout are discarded (the swapchain image is fresh each acquire).
	CommandManager::image_barrier(cmd, target.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
								  vk::PipelineStageFlagBits2::eTopOfPipe, vk::AccessFlagBits2::eNone,
								  vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);

	// Blit (not copy): absorbs any size/format mismatch between source and target — which
	// matters across a resize and between the scene format and the surface format.
	const auto layers = vk::ImageSubresourceLayers().setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1);
	const auto region =
		vk::ImageBlit()
			.setSrcSubresource(layers)
			.setDstSubresource(layers)
			.setSrcOffsets({vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<std::int32_t>(source.extent.width),
																static_cast<std::int32_t>(source.extent.height), 1}})
			.setDstOffsets({vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<std::int32_t>(target.extent.width),
																static_cast<std::int32_t>(target.extent.height), 1}});
	cmd.blitImage(source.image, vk::ImageLayout::eTransferSrcOptimal, target.image,
				  vk::ImageLayout::eTransferDstOptimal, region, vk::Filter::eLinear);

	// Leave the target ready for its consumer. For PRESENT_SRC the presentation engine is
	// synced by the render-finished semaphore, so no dst access scope is needed; otherwise
	// a transfer reader (a readback / a further blit) waits on the write.
	const bool to_present = m_final_layout == vk::ImageLayout::ePresentSrcKHR;
	CommandManager::image_barrier(cmd, target.image, vk::ImageLayout::eTransferDstOptimal, m_final_layout,
								  vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
								  to_present ? vk::PipelineStageFlagBits2::eBottomOfPipe
											 : vk::PipelineStageFlagBits2::eTransfer,
								  to_present ? vk::AccessFlagBits2::eNone : vk::AccessFlagBits2::eTransferRead);

	// Forward the written target so the next node is ordered after us and sees the image
	// (carries the swapchain index + present semaphores the PresentNode needs). The version bump
	// (owned by m_versioned) makes the published ImageRef compare unequal across consecutive
	// blits even when the swapchain hands us the same underlying image.
	m_versioned.publish(ctx, m_output, target);
	++m_record_count;
	return true;
}
} // namespace veng::nodes
