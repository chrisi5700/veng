//
// See DynamicMeshNode.hpp and design.md §L4.
//

#include <cstring>
#include <veng/gpu/MeshRef.hpp>
#include <veng/nodes/DynamicMeshNode.hpp>

namespace veng::nodes
{
std::expected<bool, graph::ExecError> DynamicMeshNode::record(gpu::GpuExecContext& ctx)
{
	const Reading vertices = m_read_vertices(ctx);
	if (!vertices.resolved)
	{
		return std::unexpected(graph::ExecError::MISSING_INPUT);
	}

	// Declare the two transient buffers once. Vertex buffer always; index buffer only when
	// the second input edge was provided at construction.
	if (!m_declared)
	{
		m_vertex_buffer_id = ctx.pool().declare_buffer(vk::BufferUsageFlagBits::eVertexBuffer);
		if (m_index_input.valid())
		{
			m_index_buffer_id = ctx.pool().declare_buffer(vk::BufferUsageFlagBits::eIndexBuffer);
		}
		m_declared = true;
	}

	// A zero-vertex publish is legal (the consuming draw uses the published count of 0 to
	// emit no work) but VMA rejects a zero-size allocation, so we round up to one stride.
	const vk::DeviceSize vertex_bytes = static_cast<vk::DeviceSize>(vertices.count) * m_vertex_stride;
	const vk::DeviceSize vertex_alloc = vertex_bytes == 0 ? m_vertex_stride : vertex_bytes;
	auto				 vbuf		  = ctx.pool().acquire_buffer(m_vertex_buffer_id, vertex_alloc);
	if (!vbuf.has_value())
	{
		return std::unexpected(graph::ExecError::NODE_FAILED);
	}
	if (vertex_bytes > 0)
	{
		std::memcpy(vbuf.value()->mapped(), vertices.bytes, vertex_bytes);
	}

	vk::Buffer	  index_handle{};
	std::uint32_t index_count = 0;
	if (m_index_input.valid())
	{
		const Reading indices = m_read_indices(ctx);
		if (!indices.resolved)
		{
			return std::unexpected(graph::ExecError::MISSING_INPUT);
		}
		const vk::DeviceSize index_bytes = static_cast<vk::DeviceSize>(indices.count) * sizeof(std::uint32_t);
		const vk::DeviceSize index_alloc = index_bytes == 0 ? sizeof(std::uint32_t) : index_bytes;
		auto				 ibuf		 = ctx.pool().acquire_buffer(m_index_buffer_id, index_alloc);
		if (!ibuf.has_value())
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		if (index_bytes > 0)
		{
			std::memcpy(ibuf.value()->mapped(), indices.bytes, index_bytes);
		}
		index_handle = ibuf.value()->buffer();
		index_count	 = indices.count;
	}

	m_versioned.publish(ctx, m_output,
						gpu::MeshRef{.vertex_buffer = vbuf.value()->buffer(),
									 .index_buffer	= index_handle,
									 .vertex_count	= vertices.count,
									 .index_count	= index_count,
									 .index_type	= vk::IndexType::eUint32,
									 .vertex_stride = m_vertex_stride});
	return true;
}
} // namespace veng::nodes
