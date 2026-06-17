/**
 * @file
 * @author chris
 * @brief @ref veng::nodes::MeshSelectorNode implementation.
 * @ingroup graph_nodes
 */

#include <algorithm>
#include <veng/gpu/MeshRef.hpp>
#include <veng/nodes/MeshSelectorNode.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/Resolve.hpp>

namespace veng::nodes
{
MeshSelectorNode::MeshSelectorNode(std::span<const graph::DataHandle> lod_meshes, graph::DataHandle level,
								   graph::DataHandle output)
	: m_mesh_count(static_cast<std::uint32_t>(lod_meshes.size()))
	, m_output(output)
{
	m_inputs.reserve(lod_meshes.size() + 1);
	m_inputs.assign(lod_meshes.begin(), lod_meshes.end());
	m_inputs.push_back(level); // level edge last; the LOD meshes precede it
}

std::expected<bool, graph::ExecError> MeshSelectorNode::execute(graph::ExecContext& ctx)
{
	if (m_mesh_count == 0)
	{
		return std::unexpected(graph::ExecError::NODE_FAILED);
	}
	auto* level_slot = graph::resolve<std::uint32_t>(ctx, m_inputs.back());
	auto* out		 = graph::resolve<gpu::MeshRef>(ctx, m_output);
	if (level_slot == nullptr || out == nullptr)
	{
		return std::unexpected(graph::ExecError::MISSING_INPUT);
	}
	// Clamp defensively: a metric node should already keep level in range, but never index past the set.
	const std::uint32_t level	  = std::min(level_slot->value(), m_mesh_count - 1);
	auto*				mesh_slot = graph::resolve<gpu::MeshRef>(ctx, m_inputs[level]);
	if (mesh_slot == nullptr)
	{
		return std::unexpected(graph::ExecError::MISSING_INPUT);
	}
	return out->produce(mesh_slot->value());
}
} // namespace veng::nodes
