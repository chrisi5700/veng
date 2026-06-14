/**
 * @file
 * @author chris
 * @brief MeshNode implementation: host-visible buffer allocation and geometry upload.
 * @ingroup graph_nodes
 */

#include <cstring>
#include <optional>
#include <span>
#include <utility>
#include <veng/gpu/MeshRef.hpp>
#include <veng/nodes/MeshNode.hpp>

namespace veng::nodes
{
namespace
{
// Host-visible, persistently-mapped buffer of `bytes`, uploaded by memcpy. Host writes made
// before the frame's submit are visible to the device, so no staging copy/barrier is needed.
// Speaks only RHI vocabulary — no `vk::`/`vma::` reaches the node. `nullopt` on allocation/map failure.
std::optional<Buffer> upload_buffer(gpu::GpuExecContext& ctx, std::span<const std::byte> bytes,
									rhi::BufferUsageFlags usage)
{
	auto buffer = Buffer::create(ctx.rhi(), bytes.size(), usage, rhi::MemoryAccess::HOST_VISIBLE);
	if (!buffer.has_value() || buffer->mapped() == nullptr)
	{
		return std::nullopt;
	}
	std::memcpy(buffer->mapped(), bytes.data(), bytes.size());
	return std::move(*buffer);
}
} // namespace

std::expected<bool, graph::ExecError> MeshNode::record(gpu::GpuExecContext& ctx)
{
	// Allocate + upload the geometry the first time we have a device. Static meshes never
	// re-enter the plan after this (no inputs), so this runs exactly once.
	if (!m_vertex_buffer.has_value())
	{
		auto vertex = upload_buffer(ctx, m_vertex_bytes, rhi::BufferUsageFlags::VERTEX);
		if (!vertex.has_value())
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		m_vertex_buffer = std::move(vertex.value());

		if (!m_index_bytes.empty())
		{
			auto index = upload_buffer(ctx, m_index_bytes, rhi::BufferUsageFlags::INDEX);
			if (!index.has_value())
			{
				return std::unexpected(graph::ExecError::NODE_FAILED);
			}
			m_index_buffer = std::move(index.value());
		}
	}

	m_versioned.publish(
		ctx, m_output,
		gpu::MeshRef{.vertex_buffer = m_vertex_buffer->handle(),
					 .index_buffer	= m_index_buffer.has_value() ? m_index_buffer->handle() : rhi::BufferHandle{},
					 .vertex_count	= m_vertex_count,
					 .index_count	= m_index_count,
					 .index_type	= rhi::IndexType::UINT32,
					 .vertex_stride = m_vertex_stride});
	return true;
}
} // namespace veng::nodes
