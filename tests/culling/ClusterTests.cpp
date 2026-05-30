//
// Unit tests for the clustered-shading culling core (veng::culling). Pure CPU — no Vulkan device.
// The load-bearing property is *consistency*: a light placed at a view-space point must be assigned
// to the same froxel that the shader's analytic mapping (project to screen tile + exponential depth
// slice) computes for a fragment at that point. If the froxel AABBs and the analytic mapping ever
// disagree, lights would light the wrong pixels — so that round-trip is the central test here.
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <veng/culling/Clusters.hpp>

using namespace veng::culling;

namespace
{
constexpr float NEAR = 0.1F;
constexpr float FAR	 = 100.0F;

// The engine's projection: perspective with the Vulkan clip-space y-flip. GLM_FORCE_DEPTH_ZERO_TO_ONE
// (set on veng_context) only changes the NDC z-range, which the froxel build is invariant to (it
// keeps the through-origin ray, not the unprojected depth), so the math holds either way.
glm::mat4 make_proj()
{
	glm::mat4 proj = glm::perspective(glm::radians(55.0F), 16.0F / 9.0F, NEAR, FAR);
	proj[1][1] *= -1.0F;
	return proj;
}

// The shader-side mapping, replicated: which linear cluster a view-space point belongs to.
std::uint32_t analytic_cluster(glm::vec3 view_pos, const glm::mat4& proj, const ClusterGrid& grid)
{
	const glm::vec4 clip = proj * glm::vec4(view_pos, 1.0F);
	const glm::vec3 ndc	 = glm::vec3(clip) / clip.w;
	const auto		tx = static_cast<std::uint32_t>(glm::clamp((ndc.x * 0.5F + 0.5F) * static_cast<float>(grid.dims.x),
																0.0F, static_cast<float>(grid.dims.x - 1)));
	const auto		ty = static_cast<std::uint32_t>(glm::clamp((ndc.y * 0.5F + 0.5F) * static_cast<float>(grid.dims.y),
																0.0F, static_cast<float>(grid.dims.y - 1)));
	const std::uint32_t slice = depth_to_slice(-view_pos.z, grid);
	return cluster_index(tx, ty, slice, grid);
}
} // namespace

TEST_CASE("build_view_froxels covers the grid with sane view-space boxes", "[culling][clusters]")
{
	const ClusterGrid grid{.dims = {16, 9, 24}, .z_near = NEAR, .z_far = FAR};
	const auto		  froxels = build_view_froxels(glm::inverse(make_proj()), grid);

	REQUIRE(froxels.size() == grid.count());
	for (const Aabb& box : froxels)
	{
		REQUIRE(box.min.x <= box.max.x);
		REQUIRE(box.min.y <= box.max.y);
		REQUIRE(box.min.z <= box.max.z);
		REQUIRE(box.max.z <= 0.0F); // entirely in front of the camera (view space looks down -Z)
	}

	// Nested slices recede: a central column's near slice is closer than its far slice.
	const Aabb near_box = froxels[cluster_index(8, 4, 0, grid)];
	const Aabb far_box	= froxels[cluster_index(8, 4, 20, grid)];
	REQUIRE(far_box.min.z < near_box.min.z); // the far slice's box is deeper (more negative)
}

TEST_CASE("depth slicing round-trips and clamps", "[culling][clusters]")
{
	const ClusterGrid grid{.dims = {16, 9, 24}, .z_near = NEAR, .z_far = FAR};

	REQUIRE(slice_depth(0, grid) == NEAR);
	REQUIRE(slice_depth(grid.dims.z, grid) == FAR);
	for (std::uint32_t k = 0; k < grid.dims.z; ++k)
	{
		REQUIRE(slice_depth(k + 1, grid) > slice_depth(k, grid)); // monotonic
		// A depth just inside slice k maps back to k.
		const float mid = (slice_depth(k, grid) + slice_depth(k + 1, grid)) * 0.5F;
		REQUIRE(depth_to_slice(mid, grid) == k);
	}
	REQUIRE(depth_to_slice(NEAR * 0.5F, grid) == 0);			 // closer than near -> first slice
	REQUIRE(depth_to_slice(FAR * 2.0F, grid) == grid.dims.z - 1); // beyond far -> last slice
}

TEST_CASE("assign_lights places a light in the cluster the shader will read", "[culling][clusters]")
{
	const glm::mat4	  proj	  = make_proj();
	const ClusterGrid grid{.dims = {16, 9, 24}, .z_near = NEAR, .z_far = FAR};
	const auto		  froxels = build_view_froxels(glm::inverse(proj), grid);

	// A point light on the view axis at a mid depth, identity view (world == view space). A fragment
	// there reads cluster `expected`; the cull must have put the light in exactly that cluster.
	const float		light_depth = 6.0F;
	const glm::vec3	view_pos{0.0F, 0.0F, -light_depth};
	const std::uint32_t expected = analytic_cluster(view_pos, proj, grid);

	const std::array<GpuLight, 1> lights{
		GpuLight{.position = glm::vec4(view_pos, 0.4F), .color = glm::vec4(1.0F)}}; // small radius
	const ClusterAssignment a = assign_lights(froxels, lights, glm::mat4(1.0F), grid);

	REQUIRE(a.grid.size() == grid.count());
	const glm::uvec2 cell = a.grid[expected];
	REQUIRE(cell.y >= 1);						   // the expected cluster has at least one light
	REQUIRE(a.indices[cell.x] == 0);			   // and it is light 0
	REQUIRE(intersects_sphere(froxels[expected], view_pos, 0.4F)); // the froxel really contains it
}

TEST_CASE("assign_lights ignores lights behind the camera", "[culling][clusters]")
{
	const glm::mat4	  proj	  = make_proj();
	const ClusterGrid grid{.dims = {16, 9, 24}, .z_near = NEAR, .z_far = FAR};
	const auto		  froxels = build_view_froxels(glm::inverse(proj), grid);

	// A small light behind the camera (+Z is behind, looking down -Z) touches no froxel.
	const std::array<GpuLight, 1> lights{GpuLight{.position = glm::vec4(0.0F, 0.0F, 5.0F, 0.5F), .color = glm::vec4(1.0F)}};
	const ClusterAssignment		  a = assign_lights(froxels, lights, glm::mat4(1.0F), grid);

	std::uint32_t total = 0;
	for (const glm::uvec2& cell : a.grid)
	{
		total += cell.y;
	}
	REQUIRE(total == 0); // no cluster references the behind-camera light
}

TEST_CASE("assign_lights flattens grid offsets consistently", "[culling][clusters]")
{
	const glm::mat4	  proj	  = make_proj();
	const ClusterGrid grid{.dims = {8, 8, 8}, .z_near = NEAR, .z_far = FAR};
	const auto		  froxels = build_view_froxels(glm::inverse(proj), grid);

	// A big light at the near-centre spilling across many froxels.
	const std::array<GpuLight, 1> lights{GpuLight{.position = glm::vec4(0.0F, 0.0F, -2.0F, 3.0F), .color = glm::vec4(1.0F)}};
	const ClusterAssignment		  a = assign_lights(froxels, lights, glm::mat4(1.0F), grid);

	// Offsets must march in lockstep with counts (offset[c] + count[c] == offset[c+1]).
	std::uint32_t running = 0;
	std::uint32_t touched = 0;
	for (const glm::uvec2& cell : a.grid)
	{
		REQUIRE(cell.x == running);
		running += cell.y;
		touched += cell.y > 0 ? 1U : 0U;
	}
	REQUIRE(touched > 1);				   // a big light really did hit several froxels
	REQUIRE(a.indices.size() == running); // the flat list holds exactly the assigned references
}
