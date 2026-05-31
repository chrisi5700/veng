/**
 * @file
 * @author chris
 * @brief Base class for GPU graph nodes.
 *
 * Localizes the one downcast from the abstract `graph::ExecContext` to `GpuExecContext`:
 * concrete GPU nodes override `record(GpuExecContext&)` and never touch the cast
 * themselves. This is the bridge rule — the rest of the L3 core stays Vulkan-agnostic;
 * only @ref veng::gpu::GpuNode knows the frame plan it is part of was dispatched with a
 * GPU context.
 *
 * @ingroup gpu_handles
 */

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
/**
 * @brief A pool-backed image a node touches, plus the layout/stage/access it must be in
 *        before `record` runs.
 *
 * The executor reads these from each @ref veng::gpu::GpuNode in the plan and inserts the barrier
 * transitions for them — nodes no longer record their own layout transitions for
 * pool-backed targets.
 *
 * @ingroup gpu_handles
 * @see GpuNode::image_usages
 */
struct ImageUsage
{
	ImageId					id;		///< Pool id of the image being declared.
	vk::ImageLayout			layout; ///< Layout the image must be in before `record`.
	vk::PipelineStageFlags2 stage;	///< Pipeline stage at which the node first accesses it.
	vk::AccessFlags2		access; ///< Access kind (read/write) the node performs.
};

/**
 * @brief Abstract base for nodes that record GPU work into a frame's command buffer.
 *
 * Subclasses override @ref record; the public @ref execute seam performs the single
 * checked downcast from the engine-neutral `graph::ExecContext` to `GpuExecContext`.
 *
 * @ingroup gpu_handles
 * @see veng::gpu::GpuExecContext
 * @see veng::gpu::Sink
 */
class GpuNode : public graph::Node
{
	 public:
	/**
	 * @brief The single, localized type seam from `graph::ExecContext` to `GpuExecContext`.
	 *
	 * `GpuExecContext` and the core CPU context are siblings, so a plain `static_cast`
	 * would be UB if a `GpuNode` were reached via the default (CPU) execute/frame path.
	 * This checks the cast and turns that misuse into a typed `WRONG_CONTEXT` error — the
	 * scheduler then marks the node `FAILED` instead of corrupting memory.
	 *
	 * @param ctx The execution context; must dynamically be a `GpuExecContext`.
	 * @return Whether the node's output changed, or `WRONG_CONTEXT` on a non-GPU context.
	 */
	[[nodiscard]] std::expected<bool, graph::ExecError> execute(graph::ExecContext& ctx) final
	{
		auto* gpu = dynamic_cast<GpuExecContext*>(&ctx);
		if (gpu == nullptr)
		{
			return std::unexpected(graph::ExecError::WRONG_CONTEXT);
		}
		return record(*gpu);
	}

	// Post-submit / post-retire hooks moved to the dedicated gpu::Sink mixin (Sink.hpp): a node
	// opts into them by inheriting Sink, instead of every GpuNode carrying no-op versions reached
	// by a downcast. The driver dispatches over Sinks in the executed plan.

	/**
	 * @brief Declare the pool-backed images this node touches and the layout/stage/access
	 *        each must be in *before* `record` runs.
	 *
	 * The executor (`GpuExecContext::prepare_for`) reads this and inserts the matching
	 * barriers via `ResourcePool::transition_image`. Nodes that touch no pool-backed images
	 * (or only external/swapchain images) return an empty vector.
	 *
	 * @param ctx The live execution context, so usages that must resolve a graph edge's
	 *            pool id can do so.
	 * @return The image usages to barrier for; empty if the node touches no pool images.
	 */
	virtual std::vector<ImageUsage> image_usages(graph::ExecContext& ctx)
	{
		(void)ctx;
		return {};
	}

	 protected:
	/**
	 * @brief Record this node's GPU work into `ctx.command_buffer()`.
	 *
	 * Same contract as `graph::Node::execute`: the returned flag drives the change-cutoff
	 * that lets downstream consumers skip re-running when this output is unchanged.
	 *
	 * @param ctx The GPU execution context for the current frame.
	 * @return Whether the node's output changed this record, or an `ExecError` on failure.
	 */
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
