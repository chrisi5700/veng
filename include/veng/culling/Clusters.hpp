/**
 * @file
 * @author chris
 * @brief Clustered-shading culling primitives: froxel grid construction and CPU-side
 *        light assignment for clustered-forward rendering.
 *
 * Deliberately Vulkan-agnostic — pure glm, no GPU types — so the algorithms unit-test
 * on the CPU without a device, following the same discipline as the reactive core. The
 * render graph wires these pure functions behind nodes named for where they run
 * (`FroxelGridCpu` / `LightCullCpu`); a future GPU implementation is a drop-in swap
 * behind the same data edges.
 *
 * The model: the view frustum is diced into a grid of "froxels" (screen tiles ×
 * exponential depth slices). @ref veng::culling::build_view_froxels computes each froxel's view-space
 * AABB (depends only on the projection and grid dimensions, so it recomputes only on
 * resize or FOV change). @ref veng::culling::assign_lights tests each light's bounding sphere against
 * every froxel and emits a per-cluster `(offset, count)` grid plus a flat index list —
 * the structure a forward shader reads to loop only the lights touching its froxel. The
 * shader reconstructs its froxel index analytically from `gl_FragCoord` and view depth
 * (see @ref veng::culling::cluster_index / @ref veng::culling::depth_to_slice), so the froxel AABBs never need to
 * reach the GPU.
 *
 * @ingroup culling
 */

#ifndef VENG_CULLING_CLUSTERS_HPP
#define VENG_CULLING_CLUSTERS_HPP

#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <vector>

namespace veng::culling
{
/**
 * @brief A point light laid out to match the shader's `Light` struct (std430: two 16-byte vec4s).
 *
 * Position is in world space; @ref veng::culling::assign_lights transforms it to view space for the
 * sphere-vs-AABB froxel test, and the forward shader shades from the world-space copy.
 *
 * @ingroup culling
 */
struct GpuLight
{
	glm::vec4 position; ///< `xyz` = world position; `w` = cull sphere radius (0 disables the light).
	glm::vec4 color;	///< `rgb` = colour; `w` = intensity.

	friend bool operator==(const GpuLight&, const GpuLight&) = default;
};
static_assert(sizeof(GpuLight) == 32, "GpuLight must match the shader's std430 Light (2x vec4)");

/**
 * @brief Froxel grid parameters: screen tiles × depth slices, bounded by `[z_near, z_far]`.
 *
 * Depth slicing is exponential so slices are thin near the camera and thick in the
 * distance.
 *
 * @ingroup culling
 */
struct ClusterGrid
{
	glm::uvec3 dims	  = {16, 9, 24}; ///< `xy` = tile columns/rows; `z` = depth slice count.
	float	   z_near = 0.1F;		 ///< Near clip plane in positive view-space depth.
	float	   z_far  = 100.0F;		 ///< Far clip plane in positive view-space depth.

	/** @brief Total number of froxels (`dims.x * dims.y * dims.z`). */
	[[nodiscard]] std::uint32_t count() const noexcept { return dims.x * dims.y * dims.z; }
};

/**
 * @brief A view-space axis-aligned bounding box representing one froxel's volume.
 * @ingroup culling
 */
struct Aabb
{
	glm::vec3 min; ///< Minimum corner in view space.
	glm::vec3 max; ///< Maximum corner in view space.

	friend bool operator==(const Aabb&, const Aabb&) = default;
};

/**
 * @brief Per-cluster light index lists produced by @ref veng::culling::assign_lights.
 *
 * `grid[c]` is a `(offset, count)` pair into `indices`; `indices` is the flattened
 * run of light indices, cluster by cluster. `grid` always has exactly
 * `ClusterGrid::count()` entries. `indices` has at least one entry (padded so the
 * GPU buffer is never zero-sized).
 *
 * @ingroup culling
 */
struct ClusterAssignment
{
	std::vector<glm::uvec2>	   grid;	///< Per-cluster `(offset, count)` pairs into `indices`.
	std::vector<std::uint32_t> indices; ///< Flat list of light indices, cluster by cluster.

	friend bool operator==(const ClusterAssignment&, const ClusterAssignment&) = default;
};

/**
 * @brief Positive view-space depth of slice boundary `slice` in `[0, dims.z]` (exponential distribution).
 * @param slice The slice boundary index (0 = near, dims.z = far).
 * @param grid  The cluster grid parameters.
 * @return View-space depth of the boundary.
 */
[[nodiscard]] float slice_depth(std::uint32_t slice, const ClusterGrid& grid) noexcept;

/**
 * @brief The depth slice a positive view-space depth falls in, clamped to `[0, dims.z - 1]`.
 *
 * The shader uses the identical mapping so its froxel index matches the one the CPU
 * cull assigned lights to.
 *
 * @param view_depth Positive view-space depth of the fragment.
 * @param grid       The cluster grid parameters.
 * @return Slice index in `[0, dims.z - 1]`.
 */
[[nodiscard]] std::uint32_t depth_to_slice(float view_depth, const ClusterGrid& grid) noexcept;

/**
 * @brief Linear froxel index from grid tile coordinates (`x + y*dimX + slice*dimX*dimY`).
 * @param x     Tile column index.
 * @param y     Tile row index.
 * @param slice Depth slice index.
 * @param grid  The cluster grid parameters.
 * @return Linear index into a flat froxel array.
 */
[[nodiscard]] std::uint32_t cluster_index(std::uint32_t x, std::uint32_t y, std::uint32_t slice,
										  const ClusterGrid& grid) noexcept;

/**
 * @brief Compute the view-space AABB of every froxel from the inverse projection matrix.
 *
 * The result has exactly `grid.count()` entries. This depends only on `inv_proj` and `grid`,
 * so it only needs to recompute on resize or FOV change. This is the `FroxelGridCpu` node's
 * pure body.
 *
 * @param inv_proj Inverse projection matrix (GLM, depth in [0,1] under `GLM_FORCE_DEPTH_ZERO_TO_ONE`).
 * @param grid     The cluster grid parameters.
 * @return View-space AABBs for every froxel.
 */
[[nodiscard]] std::vector<Aabb> build_view_froxels(const glm::mat4& inv_proj, const ClusterGrid& grid);

/**
 * @brief Test whether a sphere (view-space `center`, `radius`) intersects `box`.
 *
 * Uses the standard clamped-distance test.
 *
 * @param box    The froxel's view-space AABB.
 * @param center Sphere center in view space.
 * @param radius Sphere radius.
 * @return `true` if the sphere and box overlap.
 */
[[nodiscard]] bool intersects_sphere(const Aabb& box, glm::vec3 center, float radius) noexcept;

/**
 * @brief Assign world-space lights to froxels via a sphere-vs-AABB test in view space.
 *
 * Each light is transformed to view space via `view`, then tested against every froxel
 * AABB. Each cluster is capped at `max_per_cluster` lights. This is the `LightCullCpu`
 * node's pure body.
 *
 * @param froxels         View-space AABBs from @ref veng::culling::build_view_froxels.
 * @param lights          World-space point lights to assign.
 * @param view            View matrix (world → view transform).
 * @param grid            The cluster grid parameters.
 * @param max_per_cluster Maximum lights recorded per cluster (default: 100).
 * @return A @ref veng::culling::ClusterAssignment with per-cluster `(offset,count)` and a flat index list.
 */
[[nodiscard]] ClusterAssignment assign_lights(const std::vector<Aabb>& froxels, std::span<const GpuLight> lights,
											  const glm::mat4& view, const ClusterGrid& grid,
											  std::uint32_t max_per_cluster = 100);
} // namespace veng::culling

#endif // VENG_CULLING_CLUSTERS_HPP
