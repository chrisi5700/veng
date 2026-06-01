/**
 * @file
 * @author chris
 * @brief Wavefront OBJ loader: parses (via tinyobjloader) into a @ref MeshData and runs the shared
 *        geometry pipeline. Authored UVs/normals are preserved; missing attributes are synthesised.
 *
 * Geometry-only, like @ref load_stl — all of a file's shapes are merged into one mesh and the caller
 * supplies the material. (OBJ's `.mtl` materials are not consumed yet.)
 *
 * @ingroup assets
 * @see load_stl
 * @see MeshData
 */

#ifndef VENG_OBJLOADER_HPP
#define VENG_OBJLOADER_HPP

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
/// Error codes returned by @ref load_obj / @ref load_obj_lods.
enum class ObjError : std::uint8_t
{
	FileUnreadable, ///< The file could not be opened.
	ParseFailed,	///< tinyobjloader rejected the document.
	NoGeometry,		///< The file parsed but produced no triangles.
};

/// Stringify an @ref ObjError.
[[nodiscard]] constexpr std::string_view to_string(ObjError error) noexcept
{
	switch (error)
	{
		case ObjError::FileUnreadable: return "OBJ file could not be read";
		case ObjError::ParseFailed: return "OBJ document failed to parse";
		case ObjError::NoGeometry: return "OBJ file has no triangle geometry";
	}
	return "unknown OBJ error";
}

/// A loaded OBJ mesh: one `MeshNode` output edge plus object-space bounds for framing.
struct ObjMesh
{
	graph::DataHandle mesh;
	std::uint32_t	  vertex_count = 0;
	std::uint32_t	  index_count  = 0;
	glm::vec3		  bounds_min   = glm::vec3(0.0F);
	glm::vec3		  bounds_max   = glm::vec3(0.0F);

	[[nodiscard]] glm::vec3 center() const noexcept { return (bounds_min + bounds_max) * 0.5F; }
	[[nodiscard]] float		radius() const noexcept { return glm::length(bounds_max - bounds_min) * 0.5F; }
};

/// A loaded OBJ LOD chain (one eagerly-uploaded mesh edge per level, finest first).
struct ObjLodSet
{
	std::vector<graph::DataHandle> meshes;
	glm::vec3					   bounds_min = glm::vec3(0.0F);
	glm::vec3					   bounds_max = glm::vec3(0.0F);

	[[nodiscard]] glm::vec3 center() const noexcept { return (bounds_min + bounds_max) * 0.5F; }
	[[nodiscard]] float		radius() const noexcept { return glm::length(bounds_max - bounds_min) * 0.5F; }
};

/**
 * @brief Load an `.obj`, merge its shapes into one mesh, and add a `MeshNode` to `graph`.
 * @param graph The render graph the mesh node + edge are added to.
 * @param path  Filesystem path to the `.obj` file.
 * @param opts  Geometry pipeline options (UV synthesis density, crease angle, repair).
 */
[[nodiscard]] std::expected<ObjMesh, ObjError> load_obj(graph::Graph& graph, const std::string& path,
														const ProcessOptions& opts = {});

/**
 * @brief Load an `.obj` once and decimate it into a GPU-resident LOD chain.
 * @param levels LOD levels (finest first); empty ⇒ one full-detail mesh.
 */
[[nodiscard]] std::expected<ObjLodSet, ObjError> load_obj_lods(graph::Graph& graph, const std::string& path,
															   std::span<const LodLevel> levels,
															   const ProcessOptions&	 base = {});
} // namespace veng::assets

#endif // VENG_OBJLOADER_HPP
