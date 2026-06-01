/**
 * @file
 * @author chris
 * @brief Discrete level-of-detail mux: forwards one of N mesh edges based on a `uint` level input.
 *
 * The centrepiece of the LOD system, and deliberately pipeline-agnostic — it muxes
 * `gpu::MeshRef` edges and knows nothing about materials or shading, so any pass's
 * @ref veng::nodes::GraphicsNode can consume its output exactly like a plain @ref veng::nodes::MeshNode edge.
 *
 * The N LOD meshes are listed as the node's inputs, so the graph planner pulls in every LOD's
 * producer — i.e. all levels upload **eagerly** and stay GPU-resident, making a level switch a pure
 * edge swap (no re-upload, no hitch). The `level` index is produced by a separate metric node
 * (distance / screen coverage / …), keeping selection mechanism and policy independent.
 *
 * @ingroup graph_nodes
 * @see MeshNode
 * @see CoverageLodNode
 */

#ifndef VENG_MESHSELECTORNODE_HPP
#define VENG_MESHSELECTORNODE_HPP

#include <cstdint>
#include <expected>
#include <span>
#include <vector>
#include <veng/rendergraph/nodes/Node.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>

namespace veng::nodes
{
/**
 * @brief Forwards the LOD mesh selected by a `uint` level edge; pure CPU, no GPU work.
 * @ingroup graph_nodes
 */
class MeshSelectorNode final : public graph::Node
{
	 public:
	/**
	 * @brief Construct the mux from the LOD mesh edges (finest first), a level edge, and an output.
	 *
	 * @param lod_meshes `ValueData<gpu::MeshRef>` edges, one per LOD level (index 0 = highest detail).
	 *                   All are declared as inputs so the planner uploads every level eagerly.
	 * @param level      `ValueData<std::uint32_t>` edge selecting a level; clamped to the valid range.
	 * @param output     `ValueData<gpu::MeshRef>` edge this node publishes the selected mesh on.
	 */
	MeshSelectorNode(std::span<const graph::DataHandle> lod_meshes, graph::DataHandle level, graph::DataHandle output);

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return m_inputs; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	[[nodiscard]] std::expected<bool, graph::ExecError> execute(graph::ExecContext& ctx) override;

	 private:
	/// LOD mesh edges followed by the level edge (so the planner pulls every level's producer in).
	std::vector<graph::DataHandle> m_inputs;
	std::uint32_t	  m_mesh_count; ///< Number of LOD meshes (the rest of @ref m_inputs is the level edge).
	graph::DataHandle m_output;
};
} // namespace veng::nodes

#endif // VENG_MESHSELECTORNODE_HPP
