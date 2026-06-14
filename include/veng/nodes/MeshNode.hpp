/**
 * @file
 * @author chris
 * @brief L4 static geometry upload node: uploads CPU vertex/index data once and publishes a
 *        `gpu::MeshRef` for @ref veng::nodes::GraphicsNode to draw.
 *
 * You hand the node CPU-side vertex data (and optional indices) at construction; on its first
 * record it allocates GPU buffers, uploads the geometry, and publishes a `gpu::MeshRef` on its
 * output edge. A @ref veng::nodes::GraphicsNode::set_mesh "set_mesh(handle)" call consumes that ref to bind
 * and draw real buffer-backed geometry instead of fabricating it from `SV_VertexID`.
 *
 * The node has no graph inputs: the geometry is baked in at construction, so the node is dirty
 * exactly once (cold), uploads, then caches forever — the upload never re-runs unless the node
 * is explicitly invalidated. The buffers are host-visible and persistently mapped, so the upload
 * is a plain `memcpy` at record time (visible to the device on submit); no staging copy or
 * barrier is needed for the small, static meshes this targets.
 *
 * @ingroup graph_nodes
 * @see DynamicMeshNode
 * @see GraphicsNode
 */

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
#include <veng/gpu/VersionedOutput.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <veng/resources/Buffer.hpp>

namespace veng::nodes
{
/**
 * @brief L4 static geometry upload node: uploads CPU vertex/index data once on first record
 *        and publishes a stable `gpu::MeshRef` for @ref veng::nodes::GraphicsNode.
 * @ingroup graph_nodes
 * @see DynamicMeshNode
 * @see GraphicsNode::set_mesh
 */
class MeshNode final : public gpu::GpuNode
{
	 public:
	/**
	 * @brief Construct a static mesh node from typed vertex and index spans.
	 *
	 * Uploads `vertices` (and, if non-empty, `indices`) on first record and publishes a
	 * `gpu::MeshRef` on `output` (a `ValueData<gpu::MeshRef>`). `Vertex` is the vertex struct;
	 * its layout must match the consuming vertex shader's reflected input (tightly packed,
	 * e.g. `struct { glm::vec3 position; glm::vec3 color; }` for a `float3 position; float3
	 * color;` input). Indices are `uint32_t`.
	 *
	 * @tparam Vertex Vertex struct type; must be tightly packed to match the shader's input.
	 * @param vertices Span of CPU-side vertex data to upload.
	 * @param indices  Span of `uint32_t` indices; empty span means non-indexed draw.
	 * @param output   `ValueData<gpu::MeshRef>` edge this node publishes its ref on.
	 */
	template <class Vertex>
	MeshNode(std::span<const Vertex> vertices, std::span<const std::uint32_t> indices,
			 graph::DataHandle output) noexcept
		: m_output(output)
		, m_vertex_count(static_cast<std::uint32_t>(vertices.size()))
		, m_index_count(static_cast<std::uint32_t>(indices.size()))
		, m_vertex_stride(static_cast<std::uint32_t>(sizeof(Vertex)))
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
	graph::DataHandle m_output;
	std::uint32_t	  m_vertex_count;
	std::uint32_t	  m_index_count;
	std::uint32_t
		m_vertex_stride; ///< `sizeof(Vertex)` captured at construction; validated by @ref veng::nodes::GraphicsNode.
	std::vector<std::byte> m_vertex_bytes;
	std::vector<std::byte> m_index_bytes;
	std::optional<Buffer>  m_vertex_buffer; ///< Allocated and uploaded lazily on first record.
	std::optional<Buffer>  m_index_buffer;	///< Present only when indices were supplied at construction.
	gpu::VersionedOutput   m_versioned;		///< Owns the per-produce version bump for the published `MeshRef`.
};
} // namespace veng::nodes

#endif // VENG_MESHNODE_HPP
