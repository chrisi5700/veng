//
// Created by chris on 5/26/26.
//
// L4 upload node (design.md §L4) — the "actual verts" source. You hand it CPU-side vertex
// data (and optional indices) at construction; on its first record it allocates GPU
// buffers, uploads the geometry, and publishes a `gpu::MeshRef` on its output edge. A
// `GraphicsNode::set_mesh(handle)` consumes that ref to bind + draw real buffer-backed
// geometry instead of fabricating it from SV_VertexID. This is the user's
// `IndexedVertexNode{vertices, indices}`.
//
// It has no graph inputs: the geometry is baked in, so the node is dirty exactly once
// (cold), uploads, then caches forever — the upload never re-runs unless the node is
// explicitly invalidated. The buffers are host-visible + persistently mapped, so the
// upload is a plain memcpy at record time (visible to the device on submit); no staging
// copy or barrier is needed for the small, static meshes this targets.
//

#ifndef VENG_MESHNODE_HPP
#define VENG_MESHNODE_HPP

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <vector>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <veng/resources/Buffer.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::nodes
{
class MeshNode final : public gpu::GpuNode
{
	 public:
	/// Upload `vertices` (and, if non-empty, `indices`) on first record and publish a
	/// `gpu::MeshRef` on `output` (a `ValueData<gpu::MeshRef>`). `Vertex` is the vertex
	/// struct; its layout must match the consuming vertex shader's reflected input (tightly
	/// packed, e.g. `struct { glm::vec3 position; glm::vec3 color; }` for a `float3 position;
	/// float3 color;` input). Indices are `uint32`.
	template <class Vertex>
	MeshNode(std::span<const Vertex> vertices, std::span<const std::uint32_t> indices,
			 graph::DataHandle output) noexcept
		: m_output(output)
		, m_vertex_count(static_cast<std::uint32_t>(vertices.size()))
		, m_index_count(static_cast<std::uint32_t>(indices.size()))
	{
		const auto* vbytes = static_cast<const std::byte*>(static_cast<const void*>(vertices.data()));
		m_vertex_bytes.assign(vbytes, vbytes + vertices.size_bytes());
		const auto* ibytes = static_cast<const std::byte*>(static_cast<const void*>(indices.data()));
		m_index_bytes.assign(ibytes, ibytes + indices.size_bytes());
	}

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return {}; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	 protected:
	[[nodiscard]] std::expected<bool, graph::ExecError> record(gpu::GpuExecContext& ctx) override;

	 private:
	graph::DataHandle	   m_output;
	std::uint32_t		   m_vertex_count;
	std::uint32_t		   m_index_count;
	std::vector<std::byte> m_vertex_bytes;
	std::vector<std::byte> m_index_bytes;
	std::optional<Buffer>  m_vertex_buffer; // allocated + uploaded lazily on first record
	std::optional<Buffer>  m_index_buffer;	// only if indices were given
	std::uint64_t		   m_version = 0;	// bumped on every produce for comparable MeshRef
};
} // namespace veng::nodes

#endif // VENG_MESHNODE_HPP
