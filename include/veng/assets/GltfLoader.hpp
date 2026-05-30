//
// Created by chris on 5/30/26.
//
// L2 asset — a glTF 2.0 scene loader built on fastgltf (review.md items 6 + 7). It turns a
// .gltf/.glb file into graph-ready data: each mesh primitive becomes a MeshNode (gpu::PbrVertex
// geometry) added to the graph, each material's textures are decoded + uploaded as `Texture`s and
// exposed as graph source edges, and the node hierarchy is flattened to a per-primitive world
// matrix (a model-matrix source edge). The result wires directly into a `passes::PbrPass`:
//   for (m : model.materials) pass.add_material({...m...});
//   for (p : model.primitives) pass.add_object(p.mesh, p.model, p.material);
//
// Scope (review.md "static PBR mesh" acceptance): static geometry + materials + node transforms.
// No skinning, animation, morph targets, cameras or lights. Tangents are read from the asset when
// present and computed from UVs otherwise; normals are required. See findings.md for the cut edges.
//

#ifndef VENG_GLTFLOADER_HPP
#define VENG_GLTFLOADER_HPP

#include <cstdint>
#include <expected>
#include <glm/glm.hpp>
#include <string>
#include <string_view>
#include <vector>
#include <veng/assets/Texture.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>

namespace veng
{
class Context;
}

namespace veng::assets
{
enum class GltfError : std::uint8_t
{
	FileUnreadable,	 // the file could not be opened / read
	ParseFailed,	 // fastgltf rejected the document
	NoGeometry,		 // the asset has no renderable triangle primitives
	MissingPosition, // a primitive lacked the required POSITION attribute
	MissingNormal,	 // a primitive lacked NORMAL (we do not synthesize normals)
	TextureLoad,	 // an image failed to decode / upload
	GpuUpload,		 // a mesh/texture GPU upload failed
};

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

/// glTF alpha mode (3.9.3). Mirrors `passes::AlphaMode` value-for-value; kept in the assets layer so
/// the loader stays decoupled from the passes library (the caller maps one to the other).
enum class AlphaMode : std::uint8_t
{
	Opaque,
	Mask,
	Blend,
};

/// A material resolved to graph texture-source edges + scalar factors. Maps one-to-one onto
/// `passes::PbrMaterial`; missing glTF channels point at the model's shared default textures.
struct GltfMaterialDesc
{
	graph::DataHandle base_color;
	graph::DataHandle normal;
	graph::DataHandle metal_rough;
	graph::DataHandle emissive;
	graph::DataHandle occlusion;

	glm::vec4 base_color_factor	 = glm::vec4(1.0F);
	float	  metallic_factor	 = 1.0F;
	float	  roughness_factor	 = 1.0F;
	float	  normal_scale		 = 1.0F;
	float	  occlusion_strength = 1.0F;
	glm::vec3 emissive_factor	 = glm::vec3(0.0F);

	AlphaMode alpha_mode   = AlphaMode::Opaque;
	float	  alpha_cutoff = 0.5F;
};

/// One drawable primitive: a mesh edge (a MeshNode's output), a per-primitive world-matrix source
/// edge, and an index into `GltfModel::materials`.
struct GltfPrimitive
{
	graph::DataHandle mesh;
	graph::DataHandle model;
	std::uint32_t	  material = 0;
};

/// The loaded model. Owns the GPU `Texture`s (keep it alive while anything renders the model — the
/// material edges reference these textures' image handles). The meshes and model matrices live as
/// nodes/sources the loader added to the graph.
struct GltfModel
{
	std::vector<Texture>		  textures; // owns every uploaded texture incl. the shared defaults
	std::vector<GltfMaterialDesc> materials;
	std::vector<GltfPrimitive>	  primitives;

	// World-space axis-aligned bounds of all geometry, for framing a camera on the model.
	glm::vec3 bounds_min = glm::vec3(0.0F);
	glm::vec3 bounds_max = glm::vec3(0.0F);

	[[nodiscard]] glm::vec3 center() const noexcept { return (bounds_min + bounds_max) * 0.5F; }
	[[nodiscard]] float		radius() const noexcept { return glm::length(bounds_max - bounds_min) * 0.5F; }
};

/// Load `path` (.gltf or .glb), uploading geometry + textures into `ctx` and adding the resulting
/// MeshNodes / source edges to `graph`. The returned model is wired into a PbrPass by the caller.
[[nodiscard]] std::expected<GltfModel, GltfError> load_gltf(const Context& ctx, graph::Graph& graph,
															const std::string& path);
} // namespace veng::assets

#endif // VENG_GLTFLOADER_HPP
