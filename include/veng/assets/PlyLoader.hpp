/**
 * @file
 * @author chris
 * @brief Stanford PLY loader (via tinyply) → @ref MeshData → shared geometry pipeline.
 *
 * PLY is common for scans/photogrammetry: positions, optional normals/UVs, and notably optional
 * per-vertex colour. Geometry-only, like @ref load_stl (the caller supplies the material); per-vertex
 * colour is carried into @ref MeshData::colors but not yet consumed at pack time (PbrVertex has no
 * colour slot). Missing normals/UVs are synthesised by the pipeline.
 *
 * @ingroup assets
 * @see load_stl
 */

#ifndef VENG_PLYLOADER_HPP
#define VENG_PLYLOADER_HPP

#include <cstdint>
#include <expected>
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <veng/assets/Lod.hpp>
#include <veng/assets/MeshData.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>

namespace veng::assets
{
/// Error codes returned by @ref load_ply / @ref load_ply_lods.
enum class PlyError : std::uint8_t
{
	FileUnreadable, ///< The file could not be opened.
	ParseFailed,	///< tinyply rejected the header/body, or positions/faces were missing.
	NoGeometry,		///< The file parsed but produced no triangles.
};

/// Stringify a @ref PlyError.
[[nodiscard]] constexpr std::string_view to_string(PlyError error) noexcept
{
	switch (error)
	{
		case PlyError::FileUnreadable: return "PLY file could not be read";
		case PlyError::ParseFailed: return "PLY document failed to parse";
		case PlyError::NoGeometry: return "PLY file has no triangle geometry";
	}
	return "unknown PLY error";
}

/// A loaded PLY mesh: one `MeshNode` output edge plus object-space bounds.
struct PlyMesh
{
	graph::DataHandle mesh;
	std::uint32_t	  vertex_count = 0;
	std::uint32_t	  index_count  = 0;
	glm::vec3		  bounds_min   = glm::vec3(0.0F);
	glm::vec3		  bounds_max   = glm::vec3(0.0F);

	[[nodiscard]] glm::vec3 center() const noexcept { return (bounds_min + bounds_max) * 0.5F; }
	[[nodiscard]] float		radius() const noexcept { return glm::length(bounds_max - bounds_min) * 0.5F; }
};

/// A loaded PLY LOD chain (one eagerly-uploaded mesh edge per level, finest first).
struct PlyLodSet
{
	std::vector<graph::DataHandle> meshes;
	glm::vec3					   bounds_min = glm::vec3(0.0F);
	glm::vec3					   bounds_max = glm::vec3(0.0F);

	[[nodiscard]] glm::vec3 center() const noexcept { return (bounds_min + bounds_max) * 0.5F; }
	[[nodiscard]] float		radius() const noexcept { return glm::length(bounds_max - bounds_min) * 0.5F; }
};

/// Load a `.ply` (ASCII or binary) and add a `MeshNode` to `graph`.
[[nodiscard]] std::expected<PlyMesh, PlyError> load_ply(graph::Graph& graph, const std::string& path,
														const ProcessOptions& opts = {});

/// Load a `.ply` once and decimate it into a GPU-resident LOD chain (empty levels ⇒ one full mesh).
[[nodiscard]] std::expected<PlyLodSet, PlyError> load_ply_lods(graph::Graph& graph, const std::string& path,
															   std::span<const LodLevel> levels,
															   const ProcessOptions&	 base = {});
} // namespace veng::assets

#endif // VENG_PLYLOADER_HPP
