/**
 * @file
 * @author chris
 * @brief Implementation of @ref veng::nodes::BlitNode.
 * @ingroup graph_nodes
 */

#include <array>
#include <veng/gpu/ImageRef.hpp>
#include <veng/nodes/BlitNode.hpp>
#include <veng/rendergraph/data/Data.hpp>

namespace veng::nodes
{
BlitNode::BlitNode(graph::DataHandle src, graph::DataHandle dst, graph::DataHandle output,
				   rhi::TextureUsage final_usage) noexcept
	: m_inputs{src, dst}
	, m_output(output)
	, m_final_usage(final_usage)
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
	return {gpu::ImageUsage{.id = src->value().pool_id, .usage = rhi::TextureUsage::TRANSFER_SRC}};
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
	if (!source.texture.valid() || !target.texture.valid())
	{
		return std::unexpected(graph::ExecError::NODE_FAILED);
	}

	// Retain the pooled source copy we read while this frame is in flight (the dst is the
	// swapchain / a test target — not pool-owned, so consume is a no-op for it).
	ctx.pool().consume(source);

	rhi::CommandEncoder& enc = ctx.encoder();

	// Target: discard prior contents (undefined) -> transfer dst. We overwrite the whole image, so
	// its prior contents and layout are discarded (the swapchain image is fresh each acquire). The
	// source is already in TRANSFER_SRC (the executor transitioned it from this node's image_usages).
	enc.transition(target.texture, rhi::TextureUsage::UNDEFINED, rhi::TextureUsage::TRANSFER_DST);

	// Blit (not copy): absorbs any size/format mismatch between source and target — which
	// matters across a resize and between the scene format and the surface format.
	enc.blit(source.texture, source.extent, target.texture, target.extent, rhi::Filter::LINEAR);

	// Leave the target ready for its consumer. PRESENT means the presentation engine takes it (synced
	// by the render-finished semaphore, no dst access scope needed); otherwise a transfer reader (a
	// readback / a further blit) waits on the write.
	enc.transition(target.texture, rhi::TextureUsage::TRANSFER_DST, m_final_usage);

	// Forward the written target so the next node is ordered after us and sees the image
	// (carries the swapchain index + present semaphores the PresentNode needs). The version bump
	// (owned by m_versioned) makes the published ImageRef compare unequal across consecutive
	// blits even when the swapchain hands us the same underlying image.
	m_versioned.publish(ctx, m_output, target);
	++m_record_count;
	return true;
}
} // namespace veng::nodes
