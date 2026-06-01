/**
 * @file
 * @author chris
 * @brief 3MF loader (via lib3mf) → @ref MeshData → shared geometry pipeline.
 *
 * 3MF is the modern, clean STL replacement: indexed, manifold, with explicit units. This loader
 * consumes its geometry (all mesh objects merged) and, like @ref load_stl, feeds it as a triangle
 * soup so the pipeline welds and crease-smooths it; the caller supplies the material. (3MF's optional
 * colour/material/texture extensions are not consumed yet.)
 *
 * @ingroup assets
 * @see load_stl
 */

#ifndef VENG_THREEMFLOADER_HPP
#define VENG_THREEMFLOADER_HPP

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
/// Error codes returned by @ref load_3mf / @ref load_3mf_lods.
enum class ThreemfError : std::uint8_t
{
	FileUnreadable, ///< The file could not be opened, or lib3mf could not be loaded.
	ParseFailed,	///< lib3mf rejected the document.
	NoGeometry,		///< The file parsed but contained no triangles.
};

/// Stringify a @ref ThreemfError.
[[nodiscard]] constexpr std::string_view to_string(ThreemfError error) noexcept
{
	switch (error)
	{
		case ThreemfError::FileUnreadable: return "3MF file could not be read (or lib3mf unavailable)";
		case ThreemfError::ParseFailed: return "3MF document failed to parse";
		case ThreemfError::NoGeometry: return "3MF file has no triangle geometry";
	}
	return "unknown 3MF error";
}

/// A loaded 3MF mesh: one `MeshNode` output edge plus object-space bounds.
struct ThreemfMesh
{
	graph::DataHandle mesh;
	std::uint32_t	  vertex_count = 0;
	std::uint32_t	  index_count  = 0;
	glm::vec3		  bounds_min   = glm::vec3(0.0F);
	glm::vec3		  bounds_max   = glm::vec3(0.0F);

	[[nodiscard]] glm::vec3 center() const noexcept { return (bounds_min + bounds_max) * 0.5F; }
	[[nodiscard]] float		radius() const noexcept { return glm::length(bounds_max - bounds_min) * 0.5F; }
};

/// A loaded 3MF LOD chain (one eagerly-uploaded mesh edge per level, finest first).
struct ThreemfLodSet
{
	std::vector<graph::DataHandle> meshes;
	glm::vec3					   bounds_min = glm::vec3(0.0F);
	glm::vec3					   bounds_max = glm::vec3(0.0F);

	[[nodiscard]] glm::vec3 center() const noexcept { return (bounds_min + bounds_max) * 0.5F; }
	[[nodiscard]] float		radius() const noexcept { return glm::length(bounds_max - bounds_min) * 0.5F; }
};

/// Load a `.3mf` and add a `MeshNode` to `graph`.
[[nodiscard]] std::expected<ThreemfMesh, ThreemfError> load_3mf(graph::Graph& graph, const std::string& path,
																const ProcessOptions& opts = {});

/// Load a `.3mf` once and decimate it into a GPU-resident LOD chain (empty levels ⇒ one full mesh).
[[nodiscard]] std::expected<ThreemfLodSet, ThreemfError> load_3mf_lods(graph::Graph& graph, const std::string& path,
																	   std::span<const LodLevel> levels,
																	   const ProcessOptions&	 base = {});
} // namespace veng::assets

#endif // VENG_THREEMFLOADER_HPP
