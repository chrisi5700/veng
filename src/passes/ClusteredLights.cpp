/**
 * @file
 * @author chris
 * @brief Implementation of @ref veng::passes::wire_clustered_lights — FroxelGridCpu, LightCullCpu, and the
 *        three SSBO upload nodes wired into a @ref veng::graph::Graph.
 * @ingroup render_passes
 */

#include <memory>
#include <vector>
#include <veng/gpu/BufferRef.hpp>
#include <veng/nodes/StorageBufferNode.hpp>
#include <veng/passes/ClusteredLights.hpp>
#include <veng/rendergraph/data/Data.hpp>

namespace veng::passes
{
namespace
{
// Upload a reactive vector<T> edge through a StorageBufferNode and return its BufferRef output edge.
template <class T>
graph::DataHandle upload(graph::Graph& graph, graph::TypedHandle<std::vector<T>> values, std::string name)
{
	const graph::DataHandle out = graph.add(std::make_unique<graph::ValueData<gpu::BufferRef>>(gpu::BufferRef{}));
	graph.set_producer(out, graph.add(std::make_unique<nodes::StorageBufferNode>(values, std::move(name), out)));
	return out;
}
} // namespace

ClusteredLightEdges wire_clustered_lights(graph::Graph&										 graph,
										  graph::TypedHandle<std::vector<culling::GpuLight>> lights,
										  graph::TypedHandle<glm::mat4> view, graph::TypedHandle<glm::mat4> proj,
										  const culling::ClusterGrid& grid)
{
	// FroxelGridCpu: view-space froxel AABBs from the inverse projection (recomputed only when the
	// projection changes — the reactive clock caches it otherwise).
	const auto inv_proj = graph.add_transform([](const glm::mat4& p) { return glm::inverse(p); }, proj);
	const auto froxels =
		graph.add_transform([grid](const glm::mat4& inv) { return culling::build_view_froxels(inv, grid); }, inv_proj);

	// LightCullCpu: assign the world-space lights to froxels (sphere-vs-AABB in view space). Re-runs
	// only when the lights or the camera move.
	const auto assignment = graph.add_transform(
		[grid](const std::vector<culling::GpuLight>& ls, const std::vector<culling::Aabb>& fx, const glm::mat4& v)
		{ return culling::assign_lights(fx, ls, v, grid); }, lights, froxels, view);

	// Split the assignment into its two arrays and upload all three buffers.
	const auto grid_vec = graph.add_transform([](const culling::ClusterAssignment& a) { return a.grid; }, assignment);
	const auto index_vec =
		graph.add_transform([](const culling::ClusterAssignment& a) { return a.indices; }, assignment);

	return ClusteredLightEdges{.lights		= upload(graph, lights, "lights"),
							   .light_grid	= upload(graph, grid_vec, "light_grid"),
							   .light_index = upload(graph, index_vec, "light_index")};
}
} // namespace veng::passes
