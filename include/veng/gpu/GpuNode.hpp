//
// Created by chris on 5/25/26.
//
// Base for GPU graph nodes (design.md §L4). It localizes the one downcast from the
// abstract graph::ExecContext to GpuExecContext: concrete GPU nodes override
// `record(GpuExecContext&)` and never touch the cast themselves. This is the bridge
// rule — the rest of the L3 core stays Vulkan-agnostic; only GpuNode knows the frame
// plan it is part of was dispatched with a GPU context.
//

#ifndef VENG_GPUNODE_HPP
#define VENG_GPUNODE_HPP

#include <expected>
#include <vector>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/SubmitContext.hpp>
#include <veng/rendergraph/nodes/Node.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <veng/resources/ResourcePool.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::gpu
{
/// A pool-backed image this node touches and the layout/stage/access it needs the image in
/// before `record` runs. The driver/executor reads these from each `GpuNode` in the plan and
/// inserts the barrier transitions for them — nodes no longer record their own layout
/// transitions for pool-backed targets ([[pass-draw-redesign]]).
struct ImageUsage
{
	ImageId					id;
	vk::ImageLayout			layout;
	vk::PipelineStageFlags2 stage;
	vk::AccessFlags2		access;
};

class GpuNode : public graph::Node
{
	 public:
	// The single, localized seam. GpuExecContext and the core CPU context are siblings,
	// so a plain static_cast would be UB if a GpuNode were reached via the default
	// (CPU) execute()/frame() path. Check it: turn that misuse into a typed error
	// (WRONG_CONTEXT) — run_node then marks the node FAILED instead of corrupting (M9).
	[[nodiscard]] std::expected<bool, graph::ExecError> execute(graph::ExecContext& ctx) final
	{
		auto* gpu = dynamic_cast<GpuExecContext*>(&ctx);
		if (gpu == nullptr)
		{
			return std::unexpected(graph::ExecError::WRONG_CONTEXT);
		}
		return record(*gpu);
	}

	/// Post-submit hook for sinks. After the frame's command buffer is ended + submitted on the
	/// graphics queue, the driver/executor invokes this once per `GpuNode` in the executed plan.
	/// Default is a no-op; `PresentNode` overrides it to issue `vkQueuePresentKHR` (a queue op
	/// that must follow submit), and future screenshot/video sinks would override to enqueue
	/// their encode or schedule a fence-gated readback. Multiple sinks in one frame are peers
	/// here — no node owns submission anymore (see [[pass-draw-redesign]]).
	virtual void on_submitted(SubmitContext& ctx) noexcept { (void)ctx; }

	/// Post-retirement hook for sinks needing the GPU work *complete* (not just submitted) —
	/// e.g. a CPU readback that must wait the slot's in-flight fence before mapping the staging
	/// buffer. The driver invokes it once per `GpuNode` of the frame that just retired in that
	/// slot, at the next `acquire(slot)` (which waited the fence). Default no-op; `ScreenshotNode`
	/// overrides it to write the captured pixels to disk.
	virtual void on_retired(SubmitContext& ctx) noexcept { (void)ctx; }

	/// Declare the pool-backed images this node touches and the layout/stage/access it needs
	/// each in *before* `record` runs. The executor (`GpuExecContext::prepare_for`) reads this
	/// and inserts the matching barriers via `ResourcePool::transition_image`. Nodes that touch
	/// no pool-backed images (or only external/swapchain images) return an empty vector. Called
	/// with the live context so usages that need to resolve a graph edge's pool_id can do so.
	virtual std::vector<ImageUsage> image_usages(graph::ExecContext& ctx)
	{
		(void)ctx;
		return {};
	}

	 protected:
	// Record this node's GPU work into `ctx.command_buffer()`. Returns whether the
	// output changed (drives change-cutoff, design.md §2.4), same contract as
	// graph::Node::execute.
	[[nodiscard]] virtual std::expected<bool, graph::ExecError> record(GpuExecContext& ctx) = 0;
};

// Inline definition of GpuExecContext::prepare_for — placed here, after gpu::GpuNode is fully
// defined, to avoid circular includes (GpuNode.hpp already includes GpuExecContext.hpp).
inline void GpuExecContext::prepare_for(graph::Node& node) noexcept
{
	auto* gnode = dynamic_cast<GpuNode*>(&node);
	if (gnode == nullptr)
	{
		return;
	}
	for (const ImageUsage& usage : gnode->image_usages(*this))
	{
		m_pool->transition_image(usage.id, m_command_buffer, usage.layout, usage.stage, usage.access);
	}
}
} // namespace veng::gpu

#endif // VENG_GPUNODE_HPP
