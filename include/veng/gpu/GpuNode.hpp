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
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/rendergraph/nodes/Node.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>

namespace veng::gpu
{
class GpuNode : public graph::Node
{
	 public:
	// The single, localized seam: a GPU frame plan is always executed with a
	// GpuExecContext, so the cast is sound. Forwards to record().
	[[nodiscard]] std::expected<bool, graph::ExecError> execute(graph::ExecContext& ctx) final
	{
		return record(static_cast<GpuExecContext&>(ctx));
	}

	 protected:
	// Record this node's GPU work into `ctx.command_buffer()`. Returns whether the
	// output changed (drives change-cutoff, design.md §2.4), same contract as
	// graph::Node::execute.
	[[nodiscard]] virtual std::expected<bool, graph::ExecError> record(GpuExecContext& ctx) = 0;
};
} // namespace veng::gpu

#endif // VENG_GPUNODE_HPP
