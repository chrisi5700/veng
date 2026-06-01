/**
 * @file
 * @author chris
 * @brief STL mesh loader producing graph-ready `gpu::PbrVertex` geometry for a PBR pass.
 *
 * STL files carry only triangle positions (and a redundant per-facet normal) — no shared
 * vertices, no normals worth trusting, no texture coordinates and no tangents. To render an STL
 * through @ref veng::passes::PbrPass (whose `PbrVertex` needs position, normal, tangent and uv),
 * this loader synthesises the missing attributes:
 *   - **welds** coincident facet corners into shared vertices,
 *   - derives **crease-angle smoothed normals** (faces meeting below the crease angle are
 *     averaged; sharper edges stay hard), so threads shade smoothly while machined edges stay crisp,
 *   - generates **box / triplanar UVs** at a configurable world scale — STL has no UVs, so the
 *     surface is projected onto whichever axis plane each face most faces. Seams are invisible for
 *     the stochastic, tiling materials this targets (e.g. bare steel),
 *   - computes a **tangent frame** from those UVs for normal mapping.
 *
 * The result is a single `MeshNode` added to the graph plus its world-space bounds (for framing a
 * camera). Materials are NOT part of an STL file — the caller supplies the PBR material/textures,
 * exactly as @ref veng::assets::load_gltf's primitives are wired into a pass.
 *
 * @ingroup assets
 * @see load_gltf
 */

#ifndef VENG_STLLOADER_HPP
#define VENG_STLLOADER_HPP

#include <cstdint>
#include <expected>
#include <glm/glm.hpp>
#include <string>
#include <string_view>
#include <veng/rendergraph/Graph.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>

namespace veng::assets
{
/**
 * @brief Error codes returned by @ref veng::assets::load_stl.
 * @ingroup assets
 */
enum class StlError : std::uint8_t
{
	FileUnreadable, ///< The file could not be opened or read.
	ParseFailed,	///< The bytes are neither valid ASCII nor binary STL.
	NoGeometry,		///< The file parsed but contained no non-degenerate triangles.
};

/**
 * @brief Stringify an @ref StlError for logging and error reporting.
 * @param error The error code to render.
 * @return A human-readable description of the error.
 */
[[nodiscard]] constexpr std::string_view to_string(StlError error) noexcept
{
	switch (error)
	{
		case StlError::FileUnreadable: return "STL file could not be read";
		case StlError::ParseFailed: return "STL bytes are not valid ASCII or binary STL";
		case StlError::NoGeometry: return "STL file has no non-degenerate triangles";
	}
	return "unknown STL error";
}

/**
 * @brief Knobs controlling how the missing PBR attributes are synthesised from raw STL triangles.
 * @ingroup assets
 */
struct StlOptions
{
	/// Texture-coordinate density: UV units per world unit for the box projection. Larger ⇒ the
	/// material tiles more often across the surface (finer apparent detail). Tune to the mesh's
	/// units — e.g. ~0.15 reads well on a part measured in millimetres.
	float uv_scale = 0.15F;

	/// Crease threshold in degrees. Adjacent faces whose normals differ by less than this are
	/// smoothed together; sharper transitions become hard edges. ~40° keeps machined edges crisp
	/// while smoothing curved/threaded surfaces.
	float crease_angle_deg = 40.0F;
};

/**
 * @brief A loaded STL mesh: one `MeshNode` output edge plus world-space bounds for framing.
 *
 * The mesh geometry lives in the graph (valid as long as the graph is alive). Hand @ref mesh to
 * `passes::PbrPass::add_object` with a model-matrix source edge and a caller-chosen material.
 *
 * @ingroup assets
 */
struct StlMesh
{
	graph::DataHandle mesh;							  ///< Output edge of the `MeshNode` holding the welded geometry.
	std::uint32_t	  vertex_count = 0;				  ///< Welded/split vertex count (after crease smoothing).
	std::uint32_t	  index_count  = 0;				  ///< Triangle index count (== 3 × triangle count).
	glm::vec3		  bounds_min   = glm::vec3(0.0F); ///< Object-space AABB minimum.
	glm::vec3		  bounds_max   = glm::vec3(0.0F); ///< Object-space AABB maximum.

	/** @brief Geometric centre of the object-space bounding box. */
	[[nodiscard]] glm::vec3 center() const noexcept { return (bounds_min + bounds_max) * 0.5F; }
	/** @brief Half the diagonal length of the object-space bounding box. */
	[[nodiscard]] float radius() const noexcept { return glm::length(bounds_max - bounds_min) * 0.5F; }
};

/**
 * @brief Load an ASCII or binary `.stl` file, synthesise PBR vertex attributes, and add the
 *        resulting `MeshNode` (and its `MeshRef` output edge) to `graph`.
 *
 * Format is detected by content (binary is recognised by the exact `84 + 50·triangles` file size),
 * so a binary file whose 80-byte header happens to begin with "solid" still parses correctly.
 *
 * @param graph The render graph to which the mesh node and output edge are added.
 * @param path  Filesystem path to the `.stl` file.
 * @param opts  Attribute-synthesis options (UV scale, crease angle).
 * @return A loaded @ref StlMesh, or an @ref StlError on failure.
 */
[[nodiscard]] std::expected<StlMesh, StlError> load_stl(graph::Graph& graph, const std::string& path,
														const StlOptions& opts = {});
} // namespace veng::assets

#endif // VENG_STLLOADER_HPP
