/**
 * @file
 * @author chris
 * @brief Implementation of @ref veng::assets::detail::repair_orientation.
 * @ingroup assets
 */

#include "MeshProcessing.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <meshoptimizer.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace veng::assets::detail
{
namespace
{
// Undirected edge key (endpoints sorted) so the two triangles sharing an edge hash together.
struct EdgeKey
{
	std::uint32_t lo;
	std::uint32_t hi;
	friend bool	  operator==(const EdgeKey&, const EdgeKey&) noexcept = default;
};
struct EdgeHash
{
	std::size_t operator()(const EdgeKey& e) const noexcept
	{
		return (static_cast<std::size_t>(e.lo) << 32U) ^ static_cast<std::size_t>(e.hi);
	}
};
EdgeKey edge_key(std::uint32_t a, std::uint32_t b) noexcept
{
	return a < b ? EdgeKey{.lo = a, .hi = b} : EdgeKey{.lo = b, .hi = a};
}

// Sorted index triple, to detect exact-duplicate triangles regardless of winding/rotation.
struct TriKey
{
	std::array<std::uint32_t, 3> v;
	friend bool					 operator==(const TriKey&, const TriKey&) noexcept = default;
};
struct TriHash
{
	std::size_t operator()(const TriKey& t) const noexcept
	{
		std::size_t h = 1469598103934665603ULL;
		for (const std::uint32_t x : t.v)
		{
			h = (h ^ x) * 1099511628211ULL;
		}
		return h;
	}
};
TriKey tri_key(const Face& f)
{
	std::array<std::uint32_t, 3> s = f;
	std::sort(s.begin(), s.end());
	return TriKey{.v = s};
}
} // namespace

RepairStats repair_orientation(const std::vector<glm::vec3>& positions, std::vector<Face>& faces)
{
	RepairStats stats;

	// 1) Drop topologically-degenerate triangles (a repeated index) and exact duplicates.
	{
		std::vector<Face>					unique;
		std::unordered_set<TriKey, TriHash> seen;
		unique.reserve(faces.size());
		for (const Face& f : faces)
		{
			if (f[0] == f[1] || f[1] == f[2] || f[0] == f[2])
			{
				++stats.duplicate_faces_removed;
				continue;
			}
			if (!seen.insert(tri_key(f)).second)
			{
				++stats.duplicate_faces_removed;
				continue;
			}
			unique.push_back(f);
		}
		faces.swap(unique);
	}
	if (faces.empty())
	{
		return stats;
	}

	// 2) Edge adjacency: each undirected edge maps to the faces touching it, paired with the
	//    *directed* endpoints in that face's current winding (so we can test consistency).
	using DirEntry = std::pair<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>>; // (face, (u,v))
	std::unordered_map<EdgeKey, std::vector<DirEntry>, EdgeHash> adjacency;
	adjacency.reserve(faces.size() * 3);
	for (std::uint32_t f = 0; f < faces.size(); ++f)
	{
		const Face& tri = faces[f];
		for (int k = 0; k < 3; ++k)
		{
			const std::uint32_t u = tri[static_cast<std::size_t>(k)];
			const std::uint32_t v = tri[static_cast<std::size_t>((k + 1) % 3)];
			adjacency[edge_key(u, v)].push_back({f, {u, v}});
		}
	}

	// 3) Flood-fill a consistent winding across shared edges. flip[f] = winding inverted vs. its
	//    component seed. Two faces agree on a shared edge when they traverse it in opposite
	//    directions; if they don't, the freshly-reached face is flipped.
	std::vector<char>		   visited(faces.size(), 0);
	std::vector<char>		   flip(faces.size(), 0);
	std::vector<std::uint32_t> component(faces.size(), 0);
	std::uint32_t			   component_count = 0;

	const auto effective = [&](std::uint32_t f, std::pair<std::uint32_t, std::uint32_t> dir)
	{ return (flip[f] != 0) ? std::pair<std::uint32_t, std::uint32_t>{dir.second, dir.first} : dir; };

	std::vector<std::uint32_t> stack;
	for (std::uint32_t start = 0; start < faces.size(); ++start)
	{
		if (visited[start] != 0)
		{
			continue;
		}
		const std::uint32_t comp = component_count++;
		visited[start]			 = 1;
		component[start]		 = comp;
		stack.push_back(start);
		while (!stack.empty())
		{
			const std::uint32_t cf = stack.back();
			stack.pop_back();
			const Face& tri = faces[cf];
			for (int k = 0; k < 3; ++k)
			{
				const std::uint32_t u	 = tri[static_cast<std::size_t>(k)];
				const std::uint32_t v	 = tri[static_cast<std::size_t>((k + 1) % 3)];
				const auto			cf_e = effective(cf, {u, v});
				for (const auto& [nf, ndir] : adjacency[edge_key(u, v)])
				{
					if (nf == cf || visited[nf] != 0)
					{
						continue;
					}
					const auto nf_e = effective(nf, ndir);
					// Consistent ⇔ the two traverse the shared edge in opposite directions.
					const bool consistent = (cf_e.first == nf_e.second && cf_e.second == nf_e.first);
					flip[nf]			  = consistent ? 0 : 1;
					visited[nf]			  = 1;
					component[nf]		  = comp;
					stack.push_back(nf);
				}
			}
		}
	}

	// 4) Orient each component outward: a closed shell wound outward has positive signed volume
	//    (∑ p0·(p1×p2)); if negative, invert the whole component. Open shells sum to ≈0 → left as-is.
	std::vector<double> volume(component_count, 0.0);
	for (std::uint32_t f = 0; f < faces.size(); ++f)
	{
		const Face&		tri = faces[f];
		std::uint32_t	i0	= tri[0];
		std::uint32_t	i1	= (flip[f] != 0) ? tri[2] : tri[1];
		std::uint32_t	i2	= (flip[f] != 0) ? tri[1] : tri[2];
		const glm::vec3 p0	= positions[i0];
		const glm::vec3 p1	= positions[i1];
		const glm::vec3 p2	= positions[i2];
		volume[component[f]] += glm::dot(p0, glm::cross(p1, p2));
	}

	// 5) Apply the final winding (flood orientation, inverted for inward components).
	for (std::uint32_t f = 0; f < faces.size(); ++f)
	{
		const bool invert_component = volume[component[f]] < 0.0;
		const bool final_flip		= (flip[f] != 0) != invert_component;
		if (final_flip)
		{
			std::swap(faces[f][1], faces[f][2]);
			++stats.faces_reoriented;
		}
	}
	return stats;
}

DecimateResult decimate(const std::vector<glm::vec3>& positions, const std::vector<glm::vec3>& normals,
						const std::vector<std::uint32_t>& indices, const DecimateOptions& opts,
						std::span<const glm::vec2> uvs)
{
	const std::size_t vertex_count = positions.size();
	const float		  ratio		   = std::clamp(opts.target_ratio, 0.0F, 1.0F);

	// Target index count, kept a multiple of 3 and at least one triangle.
	std::size_t target = static_cast<std::size_t>(static_cast<float>(indices.size()) * ratio);
	target			   = (target / 3) * 3;
	target			   = std::max<std::size_t>(target, 3);

	// Simplify into a scratch buffer; both algorithms write indices that reference the *original*
	// vertices (a subset). The precise path feeds normals (and authored UVs, if given) into the metric
	// so creases/seams survive; "aggressive" trades that for the faster topology-agnostic simplifier.
	std::vector<std::uint32_t> simplified(indices.size());
	float					   error = 0.0F;
	std::size_t				   count = 0;
	if (opts.aggressive)
	{
		count = meshopt_simplifySloppy(simplified.data(), indices.data(), indices.size(), &positions[0].x, vertex_count,
									   sizeof(glm::vec3), target, opts.max_error, &error);
	}
	else
	{
		// Interleave the attributes meshopt scores: normal.xyz, then uv.xy when UVs are supplied.
		const bool		   have_uv	  = !uvs.empty();
		const std::size_t  attr_count = have_uv ? 5 : 3;
		std::vector<float> attribs(vertex_count * attr_count);
		for (std::size_t v = 0; v < vertex_count; ++v)
		{
			float* a = &attribs[v * attr_count];
			a[0]	 = normals[v].x;
			a[1]	 = normals[v].y;
			a[2]	 = normals[v].z;
			if (have_uv)
			{
				a[3] = uvs[v].x;
				a[4] = uvs[v].y;
			}
		}
		const std::array<float, 5> weights{1.0F, 1.0F, 1.0F, 1.0F, 1.0F}; // normal ×3, uv ×2
		unsigned int			   options = 0;
		if (opts.lock_boundary)
		{
			options |= meshopt_SimplifyLockBorder;
		}
		count = meshopt_simplifyWithAttributes(simplified.data(), indices.data(), indices.size(), &positions[0].x,
											   vertex_count, sizeof(glm::vec3), attribs.data(),
											   attr_count * sizeof(float), weights.data(), attr_count, nullptr, target,
											   opts.max_error, options, &error);
	}
	simplified.resize(count);

	// Compact: build an input→output vertex remap (drops now-unused vertices, improves fetch order)
	// and rewrite the indices over the compacted set. The caller applies the remap to its attribute
	// arrays — meshoptimizer's types never escape this function.
	DecimateResult result;
	result.remap.assign(vertex_count, 0);
	const std::size_t unique =
		meshopt_optimizeVertexFetchRemap(result.remap.data(), simplified.data(), simplified.size(), vertex_count);
	result.indices.resize(simplified.size());
	meshopt_remapIndexBuffer(result.indices.data(), simplified.data(), simplified.size(), result.remap.data());

	result.vertex_count	  = static_cast<std::uint32_t>(unique);
	result.triangle_count = static_cast<std::uint32_t>(simplified.size() / 3);
	result.error		  = error;
	return result;
}
} // namespace veng::assets::detail
