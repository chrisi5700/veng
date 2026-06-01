/**
 * @file
 * @author chris
 * @brief Internal, format-agnostic mesh-repair helpers shared by the asset loaders.
 *
 * Not a public header — this lives under `src/` so the repair logic stays an implementation detail
 * of `veng_assets` (the loaders expose only engine types, never these). Operates on welded position
 * indices so any loader (STL today; OBJ/PLY/3MF later) can route its raw triangles through the same
 * cleanup before attribute synthesis. Kept here (rather than in an anonymous namespace) purely so it
 * can be unit-tested directly.
 *
 * @ingroup assets
 */

#ifndef VENG_ASSETS_MESHPROCESSING_HPP
#define VENG_ASSETS_MESHPROCESSING_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <vector>

namespace veng::assets::detail
{
/// A triangle as three indices into a shared welded-position array.
using Face = std::array<std::uint32_t, 3>;

/// What @ref repair_orientation changed, for logging / tests.
struct RepairStats
{
	std::size_t duplicate_faces_removed = 0; ///< Exact-duplicate or topologically-degenerate triangles dropped.
	std::size_t faces_reoriented		= 0; ///< Triangles whose winding was flipped to make the mesh consistent.
};

/**
 * @brief Clean a welded triangle set in place: drop duplicate/degenerate faces, make winding
 *        consistent within each connected component, and orient each closed shell outward.
 *
 * Winding is unified by flooding a consistent orientation across shared edges; each component is
 * then flipped as a whole if its signed volume is negative (i.e. it faced inward). This is what lets
 * a downstream pass trust face normals and use backface culling. Open or non-manifold input is
 * handled best-effort — the function never throws and leaves ambiguous cases as flooded.
 *
 * @param positions Welded vertex positions the face indices reference.
 * @param faces     Triangle index triples; reordered in place (winding may be flipped, duplicates removed).
 * @return Counts of what was changed.
 */
RepairStats repair_orientation(const std::vector<glm::vec3>& positions, std::vector<Face>& faces);

/// Knobs for @ref decimate, in engine terms (meshoptimizer's vocabulary stays hidden in the .cpp).
struct DecimateOptions
{
	float target_ratio	= 0.5F;	 ///< Fraction of triangles to keep (0..1]. 0.25 ⇒ a quarter.
	float max_error		= 0.02F; ///< Quality cap: max deviation as a fraction of the mesh's size.
	bool  lock_boundary = false; ///< Hold open edges/holes fixed (don't erode borders).
	bool  aggressive	= false; ///< Use the faster, lower-fidelity "sloppy" simplifier.
};

/// What @ref decimate produced. @ref remap maps each input vertex to its compacted output index
/// (or `0xFFFFFFFF` if the vertex was dropped); apply it to every per-vertex array the caller holds
/// (see @ref apply_vertex_remap). @ref indices already references the compacted output vertices.
struct DecimateResult
{
	std::vector<std::uint32_t> remap;				  ///< input vertex index → output index (or 0xFFFFFFFF).
	std::vector<std::uint32_t> indices;				  ///< Simplified index buffer over the compacted vertices.
	std::uint32_t			   vertex_count	  = 0;	  ///< Surviving (compacted) vertex count.
	std::uint32_t			   triangle_count = 0;	  ///< Surviving triangle count.
	float					   error		  = 0.0F; ///< Achieved deviation, as a fraction of the mesh size.
};

/**
 * @brief Simplify an indexed mesh via edge collapse (meshoptimizer), feeding @p normals into the
 *        error metric so hard edges survive. Vertices are only ever removed, never created — so all
 *        surviving attributes stay exact and the caller re-compacts its arrays via the returned remap.
 *
 * @param positions Vertex positions (geometry + scale for the normalized error).
 * @param normals   Per-vertex normals, factored into the collapse metric to preserve creases/seams.
 * @param indices   Triangle index buffer to simplify.
 * @param opts       Target ratio, error budget, and algorithm options.
 * @param uvs        Optional per-vertex UVs; when non-empty they are added to the collapse metric so
 *                   *authored* texture seams survive (glTF/OBJ). Leave empty when UVs are synthesised
 *                   after decimation (STL), where there is nothing yet to preserve.
 * @return The remap, simplified indices, and achieved counts/error.
 */
DecimateResult decimate(const std::vector<glm::vec3>& positions, const std::vector<glm::vec3>& normals,
						const std::vector<std::uint32_t>& indices, const DecimateOptions& opts,
						std::span<const glm::vec2> uvs = {});

/// Compact a per-vertex array with a @ref DecimateResult::remap (entries with `0xFFFFFFFF` are dropped).
template <class T>
std::vector<T> apply_vertex_remap(const std::vector<T>& src, const std::vector<std::uint32_t>& remap,
								  std::uint32_t out_count)
{
	std::vector<T> out(out_count);
	for (std::size_t i = 0; i < src.size(); ++i)
	{
		if (remap[i] != 0xFFFFFFFFU)
		{
			out[remap[i]] = src[i];
		}
	}
	return out;
}
} // namespace veng::assets::detail

#endif // VENG_ASSETS_MESHPROCESSING_HPP
