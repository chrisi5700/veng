//
// Created by chris on 5/25/26.
//
// The L3 -> Vulkan seam (design.md §L4). A GpuExecContext is the graph::ExecContext
// the driver hands to Graph::execute when the frame plan contains GPU work: it
// resolves data handles against the graph (like the core CPU context) and additionally
// exposes the recording command buffer, the Vulkan Context, and the frame-in-flight
// slot. GpuNode does the single localized cast to it (see GpuNode.hpp).
//

#ifndef VENG_GPUEXECCONTEXT_HPP
#define VENG_GPUEXECCONTEXT_HPP

#include <cstddef>
#include <veng/context/Context.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/rendergraph/nodes/Node.hpp>
#include <vulkan-memory-allocator-hpp/vk_mem_alloc.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::gpu
{
class GpuExecContext final : public graph::ExecContext
{
	 public:
	GpuExecContext(const graph::Graph& graph, const Context& context, vk::CommandBuffer command_buffer,
				   std::size_t frame_slot) noexcept
		: m_graph(&graph)
		, m_context(&context)
		, m_command_buffer(command_buffer)
		, m_frame_slot(frame_slot)
	{
	}

	// graph::ExecContext — handle resolution + frame revision, same as the CPU context.
	[[nodiscard]] graph::Data*	  data(graph::DataHandle handle) const override { return m_graph->get_data(handle); }
	[[nodiscard]] graph::Revision revision() const noexcept override { return m_graph->current_revision(); }

	// GPU surface for GpuNode::record.
	[[nodiscard]] const Context&	context() const noexcept { return *m_context; }
	[[nodiscard]] vk::CommandBuffer command_buffer() const noexcept { return m_command_buffer; }
	[[nodiscard]] vk::Device		device() const noexcept { return m_context->device(); }
	[[nodiscard]] vma::Allocator	allocator() const noexcept { return m_context->allocator(); }
	[[nodiscard]] std::size_t		frame_slot() const noexcept { return m_frame_slot; }

	 private:
	const graph::Graph* m_graph;
	const Context*		m_context;
	vk::CommandBuffer	m_command_buffer;
	std::size_t			m_frame_slot;
};
} // namespace veng::gpu

#endif // VENG_GPUEXECCONTEXT_HPP
