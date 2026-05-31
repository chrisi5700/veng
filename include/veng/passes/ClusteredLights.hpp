/**
 * @file
 * @author chris
 * @brief Wiring helper that assembles the clustered-forward point-lighting sub-graph from existing
 *        node primitives and exposes the three SSBO edges @ref veng::passes::PbrPass consumes.
 *
 * The sub-graph topology is:
 * @code
 *   proj ──▶ inv_proj ─(FroxelGridCpu)─▶ froxels(view-space AABBs, CPU) ┐
 *   lights(world) ───────────────────────────────────────────────┐     │
 *   view ─────────────────────────────────────────────────────┐  │     │
 *                                                  (LightCullCpu) ◀┴─────┘
 *                                                        │
 *                              ClusterAssignment{grid, indices}
 *                               │ light_grid (uvec2/cluster)   │ light_index (flat uint)
 *                          StorageBufferNode               StorageBufferNode
 *   lights(world) ─▶ StorageBufferNode ─▶ lights SSBO   (the three edges PbrPass consumes)
 * @endcode
 *
 * FroxelGridCpu and LightCullCpu are pure CPU transforms (`veng::culling`) feeding upload nodes —
 * named for where they run so a GPU implementation is a drop-in swap behind the same three SSBO
 * edges. The froxel AABBs stay CPU-side (the shader reconstructs its froxel analytically), so only
 * the lights, grid, and index arrays reach the GPU.
 *
 * @ingroup render_passes
 */

#ifndef VENG_PASSES_CLUSTEREDLIGHTS_HPP
#define VENG_PASSES_CLUSTEREDLIGHTS_HPP

#include <glm/glm.hpp>
#include <vector>
#include <veng/culling/Clusters.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>

namespace veng::passes
{
/**
 * @brief The three SSBO edges produced by the clustered-light wiring, ready for
 *        @ref veng::passes::PbrPass::set_clustered_lights.
 * @ingroup render_passes
 */
struct ClusteredLightEdges
{
	graph::DataHandle lights;	   ///< `gpu::BufferRef` edge — world-space `culling::GpuLight[]`
	graph::DataHandle light_grid;  ///< `gpu::BufferRef` edge — per-cluster (offset, count) pairs
	graph::DataHandle light_index; ///< `gpu::BufferRef` edge — flat light indices
};

/**
 * @brief Wire the FroxelGridCpu and LightCullCpu transforms plus the three SSBO upload nodes into
 *        @p graph and return the resulting buffer edges.
 *
 * `lights` is a reactive world-space light array — changing it re-culls and re-uploads. `view` and
 * `proj` are the camera matrices: a projection change rebuilds the froxels, a camera move re-culls.
 * `grid` configures the froxel grid and must be passed to @ref veng::passes::PbrPass::set_clustered_lights
 * unchanged.
 *
 * @ingroup render_passes
 * @param graph  The render graph to wire nodes into.
 * @param lights Reactive `std::vector<culling::GpuLight>` edge — world-space point lights.
 * @param view   Reactive world-to-view matrix edge; drives the cull's depth-slice placement.
 * @param proj   Reactive projection matrix edge; drives froxel AABB reconstruction.
 * @param grid   Froxel grid configuration; must be passed verbatim to @ref veng::passes::PbrPass::set_clustered_lights.
 * @return The three SSBO edges that @ref veng::passes::PbrPass::set_clustered_lights expects.
 */
ClusteredLightEdges wire_clustered_lights(graph::Graph&										 graph,
										  graph::TypedHandle<std::vector<culling::GpuLight>> lights,
										  graph::TypedHandle<glm::mat4> view, graph::TypedHandle<glm::mat4> proj,
										  const culling::ClusterGrid& grid);
} // namespace veng::passes

#endif // VENG_PASSES_CLUSTEREDLIGHTS_HPP
