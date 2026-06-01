/**
 * @file
 * @author chris
 * @brief Format-agnostic mesh intermediate and the shared geometry pipeline every loader feeds.
 *
 * A loader's only job is to parse bytes into a @ref MeshData; @ref process (or @ref prepare +
 * @ref finalize for LOD chains) turns that into renderable `gpu::PbrVertex` geometry — welding,
 * synthesising the attributes the format omitted, decimating, and packing. This is why STL, glTF,
 * OBJ, PLY and 3MF share one geometry path and one LOD path: the only thing that differs per format
 * is which @ref MeshData fields the parser fills.
 *
 * @ingroup assets
 */

#ifndef VENG_ASSETS_MESHDATA_HPP
#define VENG_ASSETS_MESHDATA_HPP

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>
#include <veng/assets/Lod.hpp>
#include <veng/gpu/Vertex.hpp>

namespace veng::assets
{
/**
 * @brief Parser-produced geometry for one submesh; the sole input to the shared pipeline.
 *
 * @ref positions is mandatory. Every other per-vertex array is either empty (the format didn't
 * provide it — the pipeline synthesises it) or exactly `positions.size()` long (authored — the
 * pipeline preserves it). @ref indices empty means a non-indexed triangle soup (positions in groups
 * of three, e.g. STL); non-empty is an indexed triangle list. Materials are NOT here — a loader
 * returns its materials separately, one @ref MeshData per primitive/submesh.
 *
 * @ingroup assets
 */
struct MeshData
{
	std::vector<glm::vec3>	   positions; ///< Required.
	std::vector<glm::vec3>	   normals;	  ///< Empty ⇒ synthesise (weld + crease for soup, smooth otherwise).
	std::vector<glm::vec2>	   uvs;		  ///< Empty ⇒ synthesise (box / triplanar projection).
	std::vector<glm::vec4>	   tangents;  ///< Empty ⇒ compute from UVs.
	std::vector<glm::vec4>	   colors;	  ///< Optional per-vertex colour (PLY/3MF); currently unused at pack time.
	std::vector<std::uint32_t> indices;	  ///< Empty ⇒ non-indexed triangle soup.

	[[nodiscard]] bool indexed() const noexcept { return !indices.empty(); }
	[[nodiscard]] bool has_normals() const noexcept { return !normals.empty(); }
	[[nodiscard]] bool has_uvs() const noexcept { return !uvs.empty(); }
	[[nodiscard]] bool has_tangents() const noexcept { return !tangents.empty(); }
	[[nodiscard]] bool has_colors() const noexcept { return !colors.empty(); }
};

/**
 * @brief Knobs for attribute synthesis and repair (the per-format-independent geometry policy).
 * @ingroup assets
 */
struct ProcessOptions
{
	/// Run the repair pass (drop duplicate/degenerate triangles, unify winding, orient outward).
	bool repair = true;
	/// Crease threshold for *synthesised* normals (faces below it smooth together; sharper stay hard).
	float crease_angle_deg = 40.0F;
	/// Density for *synthesised* UVs: texture tiles across the mesh's longest extent (unit-agnostic).
	float texture_tiles = 1.5F;
	/// Absolute UV-density override: real-world units per tile. > 0 takes precedence over @ref texture_tiles.
	float world_units_per_tile = 0.0F;
};

/// Renderable result of the pipeline: a `gpu::PbrVertex` mesh plus its object-space bounds.
struct ProcessedMesh
{
	std::vector<gpu::PbrVertex> vertices;
	std::vector<std::uint32_t>	indices;
	glm::vec3					bounds_min = glm::vec3(0.0F);
	glm::vec3					bounds_max = glm::vec3(0.0F);
};

/**
 * @brief LOD-invariant intermediate: welded, normals guaranteed present, authored attrs carried.
 *
 * Produced once by @ref prepare; @ref finalize runs the per-level tail (decimate → UVs → tangents →
 * pack) over it, so a LOD chain is "prepare once, finalize per level".
 *
 * @ingroup assets
 */
struct PreparedMesh
{
	MeshData  mesh;							///< Welded geometry with normals filled (UVs/colours carried if authored).
	glm::vec3 bounds_min = glm::vec3(0.0F); ///< Object-space AABB minimum.
	glm::vec3 bounds_max = glm::vec3(0.0F); ///< Object-space AABB maximum.
};

/**
 * @brief Run the LOD-invariant front of the pipeline: weld a soup, guarantee normals (crease-split
 *        for soup, smooth for indexed-without-normals, kept when authored), repair, carry UVs/colours.
 */
[[nodiscard]] PreparedMesh prepare(const MeshData& data, const ProcessOptions& opts);

/**
 * @brief Run the per-level tail over a @ref PreparedMesh: decimate (attribute-aware, preserving
 *        authored UVs), synthesise UVs when absent, compute tangents when absent, pack `PbrVertex`.
 *
 * @param prep  The prepared mesh from @ref prepare.
 * @param level Decimation for this level (`target_ratio = 1.0` ⇒ full detail, no decimation).
 * @param opts  Synthesis options (UV density for synthesised UVs).
 */
[[nodiscard]] ProcessedMesh finalize(const PreparedMesh& prep, const LodLevel& level, const ProcessOptions& opts);

/// Convenience: @ref prepare then @ref finalize at full detail — the non-LOD case.
[[nodiscard]] ProcessedMesh process(const MeshData& data, const ProcessOptions& opts);
} // namespace veng::assets

#endif // VENG_ASSETS_MESHDATA_HPP
