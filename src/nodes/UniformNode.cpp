/**
 * @file
 * @author chris
 * @brief UniformNode implementation: pool-backed uniform-buffer upload and `UniformRef` publish.
 * @ingroup graph_nodes
 */

#include <cstring>
#include <utility>
#include <veng/gpu/UniformRef.hpp>
#include <veng/nodes/UniformNode.hpp>

namespace veng::nodes
{
std::expected<bool, graph::ExecError> UniformNode::record(gpu::GpuExecContext& ctx)
{
	const void* bytes = m_read(ctx);
	if (bytes == nullptr)
	{
		return std::unexpected(graph::ExecError::MISSING_INPUT);
	}

	// The uniform buffer is N-buffered by the pool: declare it once, then write this frame's
	// physical copy. A new value the CPU writes therefore never stomps a buffer an in-flight
	// frame is still reading (the previous single-buffered design was only safe at 1 in flight).
	if (!m_declared)
	{
		m_buffer_id = ctx.pool().declare_buffer(vk::BufferUsageFlagBits::eUniformBuffer);
		m_declared	= true;
	}
	auto buffer = ctx.pool().acquire_buffer(m_buffer_id, m_size);
	if (!buffer.has_value())
	{
		return std::unexpected(graph::ExecError::NODE_FAILED);
	}
	std::memcpy(buffer.value()->mapped(), bytes, m_size);

	m_versioned.publish(ctx, m_output,
						gpu::UniformRef{.buffer = buffer.value()->buffer(), .size = m_size, .name = m_name});
	return true;
}
} // namespace veng::nodes
