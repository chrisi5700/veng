//
// Clustered-shading culling primitives (the composable "cull from the light's POV" core discussed
// in review/architecture notes). Deliberately Vulkan-agnostic — pure glm, no GPU types — so it unit
// tests on the CPU without a device, the same discipline as the reactive core. The graph wires these
// pure functions behind nodes named for where they run (FroxelGridCpu / LightCullCpu) so a future GPU
// implementation is a drop-in swap behind the same data edges.
//
// The model: the view frustum is diced into a grid of "froxels" (screen tiles × exponential depth
// slices). `build_view_froxels` computes each froxel's view-space AABB (depends only on the
// projection + grid, so it recomputes only on resize/FOV change). `assign_lights` tests each light's
// bounding sphere against every froxel and emits a per-cluster (offset,count) grid + a flat index
// list — the structure a forward shader reads to loop only the lights touching its froxel. The
// shader reconstructs its froxel index *analytically* from gl_FragCoord + view depth (see
// `cluster_index` / `depth_to_slice`), so the froxel AABBs never need to reach the GPU.
//

#ifndef VENG_CULLING_CLUSTERS_HPP
#define VENG_CULLING_CLUSTERS_HPP

#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <vector>

namespace veng::culling
{
/// A point light, laid out to match the shader's `Light` (std430: two 16-byte vec4s). Position is in
/// world space; `assign_lights` transforms it to view space for the froxel test, and the shader
/// shades from the world-space copy.
struct GpuLight
{
	glm::vec4 position; // xyz = world position, w = radius (the cull sphere; 0 disables the light)
	glm::vec4 color;	// rgb = colour, w = intensity

	friend bool operator==(const GpuLight&, const GpuLight&) = default;
};
static_assert(sizeof(GpuLight) == 32, "GpuLight must match the shader's std430 Light (2x vec4)");

/// The froxel grid: screen tiles (dims.xy) × depth slices (dims.z), bounded by [z_near, z_far] in
/// positive view-space depth. Depth slicing is exponential (thin near the camera).
struct ClusterGrid
{
	glm::uvec3 dims	  = {16, 9, 24};
	float	   z_near = 0.1F;
	float	   z_far  = 100.0F;

	[[nodiscard]] std::uint32_t count() const noexcept { return dims.x * dims.y * dims.z; }
};

/// A view-space axis-aligned box (a froxel's bounds).
struct Aabb
{
	glm::vec3 min;
	glm::vec3 max;

	friend bool operator==(const Aabb&, const Aabb&) = default;
};

/// Per-cluster light lists: `grid[c]` = (offset, count) into `indices`; `indices` is the flattened
/// run of light indices, cluster by cluster. `grid` always has `ClusterGrid::count()` entries;
/// `indices` has at least one entry (padded so the GPU buffer is never zero-sized).
struct ClusterAssignment
{
	std::vector<glm::uvec2>	   grid;
	std::vector<std::uint32_t> indices;

	friend bool operator==(const ClusterAssignment&, const ClusterAssignment&) = default;
};

/// Positive view-space depth of slice boundary `slice` in [0, dims.z] (exponential distribution).
[[nodiscard]] float slice_depth(std::uint32_t slice, const ClusterGrid& grid) noexcept;

/// The depth slice a positive view-space depth falls in, clamped to [0, dims.z - 1]. The shader uses
/// the identical mapping so its froxel matches the one the cull assigned lights to.
[[nodiscard]] std::uint32_t depth_to_slice(float view_depth, const ClusterGrid& grid) noexcept;

/// Linear froxel index from grid coordinates (x + y*dimX + slice*dimX*dimY).
[[nodiscard]] std::uint32_t cluster_index(std::uint32_t x, std::uint32_t y, std::uint32_t slice,
										  const ClusterGrid& grid) noexcept;

/// View-space AABB of every froxel (size == grid.count()), from the inverse projection. Depends only
/// on (inv_proj, grid) — the FroxelGridCpu node's pure body.
[[nodiscard]] std::vector<Aabb> build_view_froxels(const glm::mat4& inv_proj, const ClusterGrid& grid);

/// Does the sphere (view-space `center`, `radius`) intersect `box`? Standard clamped-distance test.
[[nodiscard]] bool intersects_sphere(const Aabb& box, glm::vec3 center, float radius) noexcept;

/// Assign world-space `lights` to `froxels` (sphere-vs-AABB in view space via `view`), capping each
/// cluster at `max_per_cluster`. The LightCullCpu node's pure body.
[[nodiscard]] ClusterAssignment assign_lights(const std::vector<Aabb>& froxels, std::span<const GpuLight> lights,
											  const glm::mat4& view, const ClusterGrid& grid,
											  std::uint32_t max_per_cluster = 100);
} // namespace veng::culling

#endif // VENG_CULLING_CLUSTERS_HPP
