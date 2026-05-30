//
// Wiring for clustered-forward point lighting — the composable decomposition discussed in the
// architecture notes, realised out of existing node primitives:
//
//   proj ──▶ inv_proj ─(FroxelGridCpu)─▶ froxels(view-space AABBs, CPU) ┐
//   lights(world) ───────────────────────────────────────────────┐     │
//   view ─────────────────────────────────────────────────────┐  │     │
//                                                  (LightCullCpu) ◀┴─────┘
//                                                        │
//                              ClusterAssignment{grid, indices}
//                               │ light_grid (uvec2/cluster)   │ light_index (flat uint)
//                          StorageBufferNode               StorageBufferNode
//   lights(world) ─▶ StorageBufferNode ─▶ lights SSBO   (the three edges PbrPass consumes)
//
// FroxelGridCpu and LightCullCpu are pure CPU transforms (veng::culling) feeding uploads — named for
// where they run so a GPU implementation is a drop-in swap behind the same three SSBO edges. The
// froxel AABBs stay CPU-side (the shader reconstructs its froxel analytically), so only lights, grid
// and index reach the GPU.
//

#ifndef VENG_PASSES_CLUSTEREDLIGHTS_HPP
#define VENG_PASSES_CLUSTEREDLIGHTS_HPP

#include <glm/glm.hpp>
#include <vector>
#include <veng/culling/Clusters.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>

namespace veng::passes
{
/// The three SSBO edges produced by the clustered-light wiring, ready for `PbrPass::set_clustered_lights`.
struct ClusteredLightEdges
{
	graph::DataHandle lights;	   // gpu::BufferRef — world-space culling::GpuLight[]
	graph::DataHandle light_grid;  // gpu::BufferRef — per-cluster (offset, count)
	graph::DataHandle light_index; // gpu::BufferRef — flat light indices
};

/// Wire FroxelGridCpu + LightCullCpu + the three uploads into `graph`. `lights` is a reactive
/// world-space light array (changing it re-culls and re-uploads); `view`/`proj` are the camera
/// matrices (a projection change rebuilds the froxels, a camera move re-culls). `grid` configures
/// the froxel grid and must be passed to `PbrPass::set_clustered_lights` unchanged.
ClusteredLightEdges wire_clustered_lights(graph::Graph& graph, graph::TypedHandle<std::vector<culling::GpuLight>> lights,
										  graph::TypedHandle<glm::mat4> view, graph::TypedHandle<glm::mat4> proj,
										  const culling::ClusterGrid& grid);
} // namespace veng::passes

#endif // VENG_PASSES_CLUSTEREDLIGHTS_HPP
