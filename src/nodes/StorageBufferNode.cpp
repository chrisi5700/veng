/**
 * @file
 * @author chris
 * @brief Implementation of @ref veng::nodes::StorageBufferNode.
 * @ingroup graph_nodes
 */

#include <cstring>
#include <utility>
#include <veng/gpu/BufferRef.hpp>
#include <veng/nodes/StorageBufferNode.hpp>

namespace veng::nodes
{
std::expected<bool, graph::ExecError> StorageBufferNode::record(gpu::GpuExecContext& ctx)
{
	const Reading reading = m_read(ctx);
	if (!reading.resolved)
	{
		return std::unexpected(graph::ExecError::MISSING_INPUT);
	}

	// The storage buffer is N-buffered by the pool: declare it once, then write this frame's
	// physical copy. A new array the CPU writes therefore never stomps a buffer an in-flight
	// frame is still reading — the same N-buffering rule the UniformNode relies on (the
	// previous frame's draw may still be in flight when we re-upload).
	if (!m_declared)
	{
		// `eVertexBuffer` is included so the same buffer can also feed a vertex-rate binding (a
		// future zero-copy instance-attribute path); for the descriptor-bound `StructuredBuffer`
		// path the `eStorageBuffer` usage is what matters.
		m_buffer_id = ctx.pool().declare_buffer(rhi::BufferUsageFlags::STORAGE | rhi::BufferUsageFlags::VERTEX);
		m_declared = true;
	}

	// An empty array still requires a non-zero allocation (VMA rejects size==0). One stride's
	// worth is fine and unused — the consuming draw uses `count` from the published ref as its
	// `instanceCount`, so a 0-count ref produces zero draws regardless of the underlying size.
	const std::uint64_t element_bytes = static_cast<std::uint64_t>(reading.count) * m_stride;
	const std::uint64_t alloc_bytes	  = element_bytes == 0 ? m_stride : element_bytes;
	auto				 buffer		   = ctx.pool().acquire_buffer(m_buffer_id, alloc_bytes);
	if (!buffer.has_value())
	{
		return std::unexpected(graph::ExecError::NODE_FAILED);
	}
	if (element_bytes > 0)
	{
		std::memcpy(buffer.value()->mapped(), reading.bytes, element_bytes);
	}

	// Publish the *allocated* size (>= one stride), not element_bytes: a consumer binds this as the
	// descriptor range, and a range of 0 is invalid (VUID-VkDescriptorBufferInfo-range-00341). An
	// empty array therefore binds a one-stride range the shader never reads — `count` (0) is what tells
	// the consuming draw to emit zero instances, so nothing accesses it.
	m_versioned.publish(ctx, m_output,
						gpu::BufferRef{.buffer	= buffer.value()->handle(),
									   .size	= alloc_bytes,
									   .stride	= m_stride,
									   .count	= reading.count,
									   .name	= m_name,
									   .pool_id = m_buffer_id}); // let SSBO consumers retain this copy
	return true;
}
} // namespace veng::nodes
