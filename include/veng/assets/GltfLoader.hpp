/**
 * @file
 * @author chris
 * @brief glTF 2.0 scene loader built on fastgltf, producing graph-ready data wired directly
 *        into a PBR render pass.
 *
 * Each mesh primitive becomes a `MeshNode` (`gpu::PbrVertex` geometry) added to the graph;
 * each material's textures are decoded and uploaded as @ref veng::assets::Texture objects and exposed as
 * graph source edges; and the node hierarchy is flattened to a per-primitive world matrix
 * (a model-matrix source edge). The result wires directly into a `passes::PbrPass`:
 * @code
 *   for (auto& m : model.materials) pass.add_material({...m...});
 *   for (auto& p : model.primitives) pass.add_object(p.mesh, p.model, p.material);
 * @endcode
 *
 * Scope: static geometry, materials, and node transforms. No skinning, animation, morph
 * targets, cameras, or lights. Tangents are read from the asset when present and computed
 * from UVs otherwise; normals are required.
 *
 * @ingroup assets
 */

#ifndef VENG_GLTFLOADER_HPP
#define VENG_GLTFLOADER_HPP

#include <cstdint>
#include <expected>
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <veng/assets/Lod.hpp>
#include <veng/assets/Texture.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>

namespace veng
{
class Context;
}

namespace veng::assets
{
/**
 * @brief Error codes returned by @ref veng::assets::load_gltf.
 * @ingroup assets
 */
enum class GltfError : std::uint8_t
{
	FileUnreadable,	 ///< The file could not be opened or read.
	ParseFailed,	 ///< fastgltf rejected the glTF/GLB document.
	NoGeometry,		 ///< The asset has no renderable triangle primitives.
	MissingPosition, ///< A primitive lacked the required `POSITION` attribute.
	MissingNormal,	 ///< A primitive lacked `NORMAL` (normals are not synthesized).
	TextureLoad,	 ///< An image failed to decode or upload.
	GpuUpload,		 ///< A mesh or texture GPU upload failed.
};

/**
 * @brief Stringify a @ref GltfError for logging and error reporting.
 * @param error The error code to render.
 * @return A human-readable description of the error.
 */
[[nodiscard]] constexpr std::string_view to_string(GltfError error) noexcept
{
	switch (error)
	{
		case GltfError::FileUnreadable: return "glTF file could not be read";
		case GltfError::ParseFailed: return "glTF document failed to parse";
		case GltfError::NoGeometry: return "glTF asset has no triangle geometry";
		case GltfError::MissingPosition: return "glTF primitive is missing POSITION";
		case GltfError::MissingNormal: return "glTF primitive is missing NORMAL";
		case GltfError::TextureLoad: return "glTF texture failed to load";
		case GltfError::GpuUpload: return "glTF GPU upload failed";
	}
	return "unknown glTF error";
}

/**
 * @brief glTF 2.0 alpha mode (spec §3.9.3).
 *
 * Mirrors `passes::AlphaMode` value-for-value; kept in the assets layer so the loader
 * stays decoupled from the passes library. The caller maps one to the other when wiring
 * materials into a pass.
 *
 * @ingroup assets
 */
enum class AlphaMode : std::uint8_t
{
	Opaque, ///< Fully opaque; alpha is ignored.
	Mask,	///< Alpha-test: fragments below `alpha_cutoff` are discarded.
	Blend,	///< Straight-alpha blending (src*srcA + dst*(1-srcA)).
};

/**
 * @brief A glTF material resolved to graph texture-source edges and scalar factors.
 *
 * Maps one-to-one onto `passes::PbrMaterial`. Missing glTF texture channels point at
 * the model's shared default textures (white/flat-normal/etc.) so every channel is
 * always valid.
 *
 * @ingroup assets
 */
struct GltfMaterialDesc
{
	graph::DataHandle base_color;  ///< Graph source edge for the base-color texture.
	graph::DataHandle normal;	   ///< Graph source edge for the normal map.
	graph::DataHandle metal_rough; ///< Graph source edge for the metallic-roughness texture.
	graph::DataHandle emissive;	   ///< Graph source edge for the emissive texture.
	graph::DataHandle occlusion;   ///< Graph source edge for the occlusion texture.

	glm::vec4 base_color_factor	 = glm::vec4(1.0F); ///< Multiplied with the base-color texture sample.
	float	  metallic_factor	 = 1.0F;			///< Metallic scale factor.
	float	  roughness_factor	 = 1.0F;			///< Roughness scale factor.
	float	  normal_scale		 = 1.0F;			///< Normal-map intensity scale.
	float	  occlusion_strength = 1.0F;			///< Occlusion intensity scale.
	glm::vec3 emissive_factor	 = glm::vec3(0.0F); ///< Emissive tint/strength multiplier.

	AlphaMode alpha_mode   = AlphaMode::Opaque; ///< Alpha blending mode for this material.
	float	  alpha_cutoff = 0.5F;				///< Discard threshold for `AlphaMode::Mask`.
};

/**
 * @brief One drawable primitive: a mesh edge, a per-primitive world-matrix source edge,
 *        and an index into @ref veng::assets::GltfModel::materials.
 * @ingroup assets
 */
struct GltfPrimitive
{
	graph::DataHandle mesh;			///< Output edge of the `MeshNode` holding this primitive's geometry.
	graph::DataHandle model;		///< Source edge for the per-primitive world transform matrix.
	std::uint32_t	  material = 0; ///< Index into `GltfModel::materials`.
};

/**
 * @brief The fully loaded model returned by @ref veng::assets::load_gltf.
 *
 * Owns all GPU @ref veng::assets::Texture objects — keep it alive as long as anything renders the
 * model, because the material edges reference these textures' image handles. The meshes
 * and model-matrix source nodes live in the graph and are valid as long as the graph is
 * alive.
 *
 * @ingroup assets
 */
struct GltfModel
{
	std::vector<Texture>		  textures;	  ///< Owns every uploaded texture including shared defaults.
	std::vector<GltfMaterialDesc> materials;  ///< One entry per glTF material.
	std::vector<GltfPrimitive>	  primitives; ///< One entry per renderable triangle primitive.

	/// World-space axis-aligned bounding box of all geometry (for framing a camera).
	glm::vec3 bounds_min = glm::vec3(0.0F);
	/// @copydoc bounds_min
	glm::vec3 bounds_max = glm::vec3(0.0F);

	/** @brief Geometric centre of the world-space bounding box. */
	[[nodiscard]] glm::vec3 center() const noexcept { return (bounds_min + bounds_max) * 0.5F; }
	/** @brief Half the diagonal length of the world-space bounding box. */
	[[nodiscard]] float radius() const noexcept { return glm::length(bounds_max - bounds_min) * 0.5F; }
};

/**
 * @brief Load a `.gltf` or `.glb` file, uploading geometry and textures to the GPU and
 *        adding the resulting `MeshNode`s / source edges to `graph`.
 *
 * The returned @ref veng::assets::GltfModel is wired into a `passes::PbrPass` by the caller.
 *
 * @param ctx   The engine context providing the Vulkan device and allocator.
 * @param graph The render graph to which mesh nodes and source edges are added.
 * @param path  Filesystem path to the `.gltf` or `.glb` file.
 * @return A fully loaded @ref veng::assets::GltfModel, or a @ref GltfError on failure.
 */
[[nodiscard]] std::expected<GltfModel, GltfError> load_gltf(const Context& ctx, graph::Graph& graph,
															const std::string& path);

/**
 * @brief One drawable primitive as a LOD chain: N GPU-resident mesh edges (finest first) plus the
 *        per-primitive transform and material index.
 * @ingroup assets
 */
struct GltfLodPrimitive
{
	std::vector<graph::DataHandle> lods;		 ///< One `MeshRef` edge per LOD level (index 0 = finest).
	graph::DataHandle			   model;		 ///< Source edge for the per-primitive world transform.
	std::uint32_t				   material = 0; ///< Index into @ref GltfLodModel::materials.
};

/**
 * @brief A glTF model whose every primitive is a decimated LOD chain (see @ref load_gltf_lods).
 *
 * Mirrors @ref GltfModel but with @ref GltfLodPrimitive in place of single-mesh primitives. Owns the
 * uploaded textures — keep it alive while rendering. A single screen-coverage metric on @ref center /
 * @ref radius can drive one @ref veng::nodes::MeshSelectorNode per primitive (all switching together).
 *
 * @ingroup assets
 */
struct GltfLodModel
{
	std::vector<Texture>		  textures;	  ///< Owns every uploaded texture including shared defaults.
	std::vector<GltfMaterialDesc> materials;  ///< One entry per glTF material.
	std::vector<GltfLodPrimitive> primitives; ///< One LOD chain per renderable triangle primitive.

	glm::vec3 bounds_min = glm::vec3(0.0F); ///< World-space AABB minimum of all geometry.
	glm::vec3 bounds_max = glm::vec3(0.0F); ///< @copydoc bounds_min

	/** @brief Geometric centre of the world-space bounding box. */
	[[nodiscard]] glm::vec3 center() const noexcept { return (bounds_min + bounds_max) * 0.5F; }
	/** @brief Half the diagonal length of the world-space bounding box. */
	[[nodiscard]] float radius() const noexcept { return glm::length(bounds_max - bounds_min) * 0.5F; }
};

/**
 * @brief Load a glTF and decimate every primitive into a GPU-resident LOD chain.
 *
 * Like @ref load_gltf but each primitive becomes @p levels eagerly-uploaded meshes. Decimation is
 * attribute-aware (authored normals *and* UVs feed the metric), and surviving vertices keep their
 * exact attributes — so the texturing is identical across levels. With no levels given, each
 * primitive gets a single full-detail mesh.
 *
 * @param ctx    The engine context providing the Vulkan device and allocator.
 * @param graph  The render graph mesh nodes and source edges are added to.
 * @param path   Filesystem path to the `.gltf`/`.glb` file.
 * @param levels LOD levels to generate per primitive (finest first); empty ⇒ one full-detail level.
 * @return A @ref GltfLodModel, or a @ref GltfError on failure.
 */
[[nodiscard]] std::expected<GltfLodModel, GltfError> load_gltf_lods(const Context& ctx, graph::Graph& graph,
																	const std::string&		  path,
																	std::span<const LodLevel> levels);
} // namespace veng::assets

#endif // VENG_GLTFLOADER_HPP
