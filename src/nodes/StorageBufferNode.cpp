//
// See StorageBufferNode.hpp and design.md §L4.
//

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
		m_buffer_id =
			ctx.pool().declare_buffer(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer);
		m_declared = true;
	}

	// An empty array still requires a non-zero allocation (VMA rejects size==0). One stride's
	// worth is fine and unused — the consuming draw uses `count` from the published ref as its
	// `instanceCount`, so a 0-count ref produces zero draws regardless of the underlying size.
	const vk::DeviceSize element_bytes = static_cast<vk::DeviceSize>(reading.count) * m_stride;
	const vk::DeviceSize alloc_bytes   = element_bytes == 0 ? m_stride : element_bytes;
	auto				 buffer		   = ctx.pool().acquire_buffer(m_buffer_id, alloc_bytes);
	if (!buffer.has_value())
	{
		return std::unexpected(graph::ExecError::NODE_FAILED);
	}
	if (element_bytes > 0)
	{
		std::memcpy(buffer.value()->mapped(), reading.bytes, element_bytes);
	}

	if (auto* out = dynamic_cast<graph::ValueData<gpu::BufferRef>*>(ctx.data(m_output)); out != nullptr)
	{
		++m_version;
		(void)out->produce(gpu::BufferRef{.buffer  = buffer.value()->buffer(),
										  .size	   = element_bytes,
										  .stride  = m_stride,
										  .count   = reading.count,
										  .name	   = m_name,
										  .version = m_version});
	}
	return true;
}
} // namespace veng::nodes
