//
// See veng/culling/Clusters.hpp.
//

#include <veng/culling/Clusters.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace veng::culling
{
namespace
{
// Unproject an NDC point (Vulkan: x,y in [-1,1], z in [0,1] under GLM_FORCE_DEPTH_ZERO_TO_ONE) to
// view space via the inverse projection.
glm::vec3 unproject(glm::vec2 ndc, float ndc_z, const glm::mat4& inv_proj) noexcept
{
	const glm::vec4 v = inv_proj * glm::vec4(ndc.x, ndc.y, ndc_z, 1.0F);
	return glm::vec3(v) / v.w;
}

// The point where the ray origin->p (origin at the eye) crosses the view-space plane z = target_z.
// Since the eye is at the origin the ray is t*p, so t = target_z / p.z. The froxel's eight corners
// are its four screen-tile rays scaled to the slice's near and far z-planes.
glm::vec3 ray_to_z(glm::vec3 p, float target_z) noexcept
{
	return p * (target_z / p.z);
}
} // namespace

float slice_depth(std::uint32_t slice, const ClusterGrid& grid) noexcept
{
	// Exponential split: depth(k) = near * (far/near)^(k/numZ). Slices are thin near the camera.
	const float ratio = grid.z_far / grid.z_near;
	return grid.z_near * std::pow(ratio, static_cast<float>(slice) / static_cast<float>(grid.dims.z));
}

std::uint32_t depth_to_slice(float view_depth, const ClusterGrid& grid) noexcept
{
	if (view_depth <= grid.z_near)
	{
		return 0;
	}
	const float ratio = grid.z_far / grid.z_near;
	const float k	  = std::log(view_depth / grid.z_near) / std::log(ratio) * static_cast<float>(grid.dims.z);
	const auto	slice = static_cast<std::int32_t>(k);
	return static_cast<std::uint32_t>(std::clamp(slice, 0, static_cast<std::int32_t>(grid.dims.z) - 1));
}

std::uint32_t cluster_index(std::uint32_t x, std::uint32_t y, std::uint32_t slice, const ClusterGrid& grid) noexcept
{
	return x + (y * grid.dims.x) + (slice * grid.dims.x * grid.dims.y);
}

std::vector<Aabb> build_view_froxels(const glm::mat4& inv_proj, const ClusterGrid& grid)
{
	std::vector<Aabb> froxels(grid.count());
	const auto		  dim_x = static_cast<float>(grid.dims.x);
	const auto		  dim_y = static_cast<float>(grid.dims.y);

	for (std::uint32_t y = 0; y < grid.dims.y; ++y)
	{
		for (std::uint32_t x = 0; x < grid.dims.x; ++x)
		{
			// Screen-tile corners in NDC. Vulkan NDC y points down, the same way gl_FragCoord.y does,
			// so froxel (x,y) maps directly onto the shader's gl_FragCoord tile — no flip here.
			const glm::vec2 ndc_min{(2.0F * static_cast<float>(x) / dim_x) - 1.0F,
									(2.0F * static_cast<float>(y) / dim_y) - 1.0F};
			const glm::vec2 ndc_max{(2.0F * static_cast<float>(x + 1) / dim_x) - 1.0F,
									(2.0F * static_cast<float>(y + 1) / dim_y) - 1.0F};
			// The four tile-corner rays (unprojected at the near plane; any z works — the ray through
			// the origin is what we keep).
			const std::array<glm::vec3, 4> corners{unproject({ndc_min.x, ndc_min.y}, 0.0F, inv_proj),
												   unproject({ndc_max.x, ndc_min.y}, 0.0F, inv_proj),
												   unproject({ndc_min.x, ndc_max.y}, 0.0F, inv_proj),
												   unproject({ndc_max.x, ndc_max.y}, 0.0F, inv_proj)};

			for (std::uint32_t z = 0; z < grid.dims.z; ++z)
			{
				// View-space z is negative in front of the camera; the slice spans [-far, -near].
				const float z_near = -slice_depth(z, grid);
				const float z_far  = -slice_depth(z + 1, grid);

				glm::vec3 lo(std::numeric_limits<float>::max());
				glm::vec3 hi(std::numeric_limits<float>::lowest());
				for (const glm::vec3& corner : corners)
				{
					const glm::vec3 at_near = ray_to_z(corner, z_near);
					const glm::vec3 at_far	= ray_to_z(corner, z_far);
					lo = glm::min(lo, glm::min(at_near, at_far));
					hi = glm::max(hi, glm::max(at_near, at_far));
				}
				froxels[cluster_index(x, y, z, grid)] = Aabb{.min = lo, .max = hi};
			}
		}
	}
	return froxels;
}

bool intersects_sphere(const Aabb& box, glm::vec3 center, float radius) noexcept
{
	const glm::vec3 closest	 = glm::clamp(center, box.min, box.max);
	const glm::vec3 delta	 = closest - center;
	return glm::dot(delta, delta) <= radius * radius;
}

ClusterAssignment assign_lights(const std::vector<Aabb>& froxels, std::span<const GpuLight> lights,
								const glm::mat4& view, const ClusterGrid& grid, std::uint32_t max_per_cluster)
{
	const std::uint32_t cluster_count = grid.count();
	std::vector<std::vector<std::uint32_t>> per_cluster(cluster_count);

	for (std::uint32_t li = 0; li < lights.size(); ++li)
	{
		const GpuLight& light  = lights[li];
		const float		radius = light.position.w;
		if (radius <= 0.0F)
		{
			continue; // a zero-radius light culls to nothing (the disabled / directional-only case)
		}
		const glm::vec3 center = glm::vec3(view * glm::vec4(glm::vec3(light.position), 1.0F));
		for (std::uint32_t ci = 0; ci < cluster_count; ++ci)
		{
			if (intersects_sphere(froxels[ci], center, radius) && per_cluster[ci].size() < max_per_cluster)
			{
				per_cluster[ci].push_back(li);
			}
		}
	}

	ClusterAssignment out;
	out.grid.resize(cluster_count);
	std::uint32_t offset = 0;
	for (std::uint32_t ci = 0; ci < cluster_count; ++ci)
	{
		const auto count = static_cast<std::uint32_t>(per_cluster[ci].size());
		out.grid[ci]	 = glm::uvec2(offset, count);
		out.indices.insert(out.indices.end(), per_cluster[ci].begin(), per_cluster[ci].end());
		offset += count;
	}
	if (out.indices.empty())
	{
		out.indices.push_back(0); // never publish a zero-sized buffer; every grid count is 0 here anyway
	}
	return out;
}
} // namespace veng::culling
