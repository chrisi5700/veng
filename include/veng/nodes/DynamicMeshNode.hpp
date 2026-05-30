//
// L4 upload node (design.md §L4) — the reactive sibling of MeshNode. You hand it a
// `TypedHandle<std::vector<Vertex>>` (and optionally a vector<uint32_t> for indices); each
// time that vector changes it re-uploads the bytes into its persistent vertex buffer and
// publishes an updated `gpu::MeshRef`. A `GraphicsNode::set_mesh(handle)` consumes the ref
// the same as for a static MeshNode. This is the "geometry changes each frame" path: debug
// lines refilled every step, dynamic particle sprites, deformable mesh — anything whose CB
// recording is the same draw call but whose vertex bytes change.
//
// Unlike MeshNode (static, uploads once cold and caches) this is genuinely reactive: the
// vector edge is a graph input, so changing it re-runs the upload and re-renders every
// consuming pass. The buffer is pool-owned (N-buffered, persistently mapped) so a CPU
// re-upload never stomps a copy frame N's GPU work is still consuming. Buffer growth
// reallocates the pool copies — the size is determined per-record from `vertices.size()`.
//

#ifndef VENG_DYNAMICMESHNODE_HPP
#define VENG_DYNAMICMESHNODE_HPP

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <span>
#include <utility>
#include <vector>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/gpu/VersionedOutput.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::nodes
{
class DynamicMeshNode final : public gpu::GpuNode
{
	 public:
	/// Upload the contents of `vertices` (a `TypedHandle<std::vector<Vertex>>`) on every
	/// change and publish a `gpu::MeshRef` on `output` (a `ValueData<gpu::MeshRef>`). The
	/// indices are an optional second reactive edge (the empty handle means non-indexed; the
	/// draw count is `vertices.size()`). `Vertex` is the vertex struct — its layout must
	/// match the consuming vertex shader's reflected input (tightly packed).
	template <class Vertex>
	DynamicMeshNode(graph::TypedHandle<std::vector<Vertex>> vertices, graph::DataHandle output) noexcept
		: DynamicMeshNode(vertices, graph::TypedHandle<std::vector<std::uint32_t>>{}, output, sizeof(Vertex))
	{
	}

	template <class Vertex>
	DynamicMeshNode(graph::TypedHandle<std::vector<Vertex>>		   vertices,
					graph::TypedHandle<std::vector<std::uint32_t>> indices, graph::DataHandle output) noexcept
		: DynamicMeshNode(vertices, indices, output, sizeof(Vertex))
	{
	}

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return m_inputs; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	 protected:
	[[nodiscard]] std::expected<bool, graph::ExecError> record(gpu::GpuExecContext& ctx) override;

	 private:
	// Internal constructor shared by the vertex-only and vertex+index template ctors above.
	// Captures the typed slot resolvers in lambdas so the cpp stays template-free.
	template <class Vertex>
	DynamicMeshNode(graph::TypedHandle<std::vector<Vertex>>		   vertices,
					graph::TypedHandle<std::vector<std::uint32_t>> indices, graph::DataHandle output,
					std::size_t vertex_size) noexcept
		: m_output(output)
		, m_vertex_stride(static_cast<std::uint32_t>(vertex_size))
		, m_read_vertices(
			  [handle = vertices.handle](graph::ExecContext& ctx) -> Reading
			  {
				  const auto* slot = dynamic_cast<graph::ValueData<std::vector<Vertex>>*>(ctx.data(handle));
				  if (slot == nullptr)
				  {
					  return Reading{};
				  }
				  const std::vector<Vertex>& v = slot->value();
				  return Reading{.resolved = true,
								 .bytes	   = static_cast<const void*>(v.data()),
								 .count	   = static_cast<std::uint32_t>(v.size())};
			  })
	{
		m_inputs.push_back(vertices.handle);
		if (indices.valid())
		{
			m_index_input = indices.handle;
			m_inputs.push_back(indices.handle);
			m_read_indices = [handle = indices.handle](graph::ExecContext& ctx) -> Reading
			{
				const auto* slot = dynamic_cast<graph::ValueData<std::vector<std::uint32_t>>*>(ctx.data(handle));
				if (slot == nullptr)
				{
					return Reading{};
				}
				const std::vector<std::uint32_t>& v = slot->value();
				return Reading{.resolved = true,
							   .bytes	 = static_cast<const void*>(v.data()),
							   .count	 = static_cast<std::uint32_t>(v.size())};
			};
		}
	}

	struct Reading
	{
		bool		  resolved = false;
		const void*	  bytes	   = nullptr;
		std::uint32_t count	   = 0;
	};

	graph::DataHandle							m_output;
	std::uint32_t								m_vertex_stride;
	std::vector<graph::DataHandle>				m_inputs;		 // [vertices(, indices)]
	graph::DataHandle							m_index_input{}; // valid => indexed draw
	std::function<Reading(graph::ExecContext&)> m_read_vertices;
	std::function<Reading(graph::ExecContext&)> m_read_indices; // empty => non-indexed
	bool										m_declared		   = false;
	BufferId									m_vertex_buffer_id = 0;
	BufferId									m_index_buffer_id  = 0;
	gpu::VersionedOutput						m_versioned; // owns the per-upload version bump for the MeshRef
};
} // namespace veng::nodes

#endif // VENG_DYNAMICMESHNODE_HPP
