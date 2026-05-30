//
// L5 pass — metallic-roughness PBR (review.md item 5). The lit, textured counterpart to PhongPass:
// it renders registered objects with a Cook-Torrance BRDF over a single directional light, reading
// the standard glTF material inputs (baseColor / normal / metallic-roughness / emissive / occlusion
// textures + scalar factors). It is the pass the glTF loader targets to display real assets.
//
// Materials are registered up front (`add_material` -> index); objects reference a material by index
// plus a mesh edge and a per-object model-matrix edge. Each material binds its own descriptor set of
// five textures, so a multi-material model renders in one pass. The pass does NOT own textures — the
// caller (glTF loader / example / test) keeps the `assets::Texture`s alive; a material slot with no
// texture must still be given a default (white / flat-normal) texture so the shader stays uniform.
//
// Scope: opaque, single directional light, no IBL (all deferred follow-ons — see findings.md).
//

#ifndef VENG_PBRPASS_HPP
#define VENG_PBRPASS_HPP

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <veng/culling/Clusters.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::passes
{
struct PbrConfig
{
	glm::vec3 light_direction		 = glm::normalize(glm::vec3(-0.4F, 1.0F, 0.6F)); // direction from surface TO light
	glm::vec3 light_color			 = glm::vec3(1.0F, 1.0F, 1.0F);
	float	  light_intensity		 = 3.0F;
	glm::vec3 ambient				 = glm::vec3(0.03F, 0.03F, 0.03F);
	std::array<float, 4> clear_color = {0.02F, 0.03F, 0.05F, 1.0F};
	vk::CullModeFlags	 cull_mode	 = vk::CullModeFlagBits::eBack; // glTF is back-face culled; tests use eNone
};

/// glTF alpha handling for a material (3.9.3). OPAQUE ignores alpha; MASK discards fragments below
/// `alpha_cutoff` (rides the opaque, depth-writing pipeline); BLEND alpha-blends in a separate,
/// depth-tested-but-not-written batch drawn after all opaque/mask geometry.
enum class AlphaMode : std::uint8_t
{
	Opaque,
	Mask,
	Blend,
};

/// A material's five sampled-image edges plus its scalar factors. A texture edge must resolve to a
/// `ValueData<gpu::ImageRef>` (typically an `assets::Texture::ref()` fed through a graph source);
/// for a channel a model doesn't provide, bind a 1x1 default so the shader stays uniform — white
/// for baseColor/MR/emissive/occlusion, a flat (128,128,255) normal for the normal map.
struct PbrMaterial
{
	graph::DataHandle base_color;  // RGBA, sRGB-encoded colour
	graph::DataHandle normal;	   // linear, tangent-space normal map
	graph::DataHandle metal_rough; // linear, glTF packing G=roughness B=metallic
	graph::DataHandle emissive;	   // RGB, sRGB-encoded
	graph::DataHandle occlusion;   // linear, R=ambient occlusion

	glm::vec4 base_color_factor	 = glm::vec4(1.0F, 1.0F, 1.0F, 1.0F);
	float	  metallic_factor	 = 1.0F;
	float	  roughness_factor	 = 1.0F;
	float	  normal_scale		 = 1.0F;
	float	  occlusion_strength = 1.0F;
	glm::vec3 emissive_factor	 = glm::vec3(0.0F, 0.0F, 0.0F);

	AlphaMode alpha_mode   = AlphaMode::Opaque;
	float	  alpha_cutoff = 0.5F; // only consulted for AlphaMode::Mask
};

class PbrRenderNode; // rendering half, defined in the .cpp

class PbrPass
{
	 public:
	/// `screen` sizes the owned colour + depth targets; `output` is the lit-scene image edge the
	/// caller demands; `view_proj` and `eye` (xyz = camera position) drive the camera/lighting.
	PbrPass(graph::Graph& graph, vk::Format color_format, vk::Format depth_format,
			graph::TypedHandle<vk::Extent2D> screen, graph::DataHandle output, graph::TypedHandle<glm::mat4> view_proj,
			graph::TypedHandle<glm::vec4> eye, const PbrConfig& config = {});

	PbrPass(const PbrPass&)			   = delete;
	PbrPass& operator=(const PbrPass&) = delete;
	PbrPass(PbrPass&&)				   = delete;
	PbrPass& operator=(PbrPass&&)	   = delete;
	~PbrPass()						   = default;

	/// Register a material; the returned index is passed to `add_object`.
	[[nodiscard]] std::uint32_t add_material(const PbrMaterial& material);

	/// Register an object: its mesh edge, per-object model-matrix edge, and material index.
	void add_object(graph::DataHandle mesh, graph::DataHandle model, std::uint32_t material);

	/// Enable clustered point lighting. `view` is the world->view matrix edge; `lights` is a
	/// `gpu::BufferRef` edge of world-space `culling::GpuLight`s; `light_grid` and `light_index` are
	/// the cull's per-cluster (offset,count) grid and flat index-list `gpu::BufferRef` edges (the
	/// LightCullCpu outputs). `grid` must match the one the cull built. Optional — without it the pass
	/// shades the directional key light + ambient only. See `culling::wire_clustered_lights`.
	void set_clustered_lights(graph::DataHandle view, graph::DataHandle lights, graph::DataHandle light_grid,
							  graph::DataHandle light_index, const culling::ClusterGrid& grid);

	[[nodiscard]] graph::DataHandle output() const noexcept { return m_output; }
	[[nodiscard]] graph::NodeHandle node() const noexcept { return m_node; }

	 private:
	graph::Graph*	  m_graph  = nullptr;
	PbrRenderNode*	  m_render = nullptr; // graph-owned
	graph::DataHandle m_output;
	graph::NodeHandle m_node;
};
} // namespace veng::passes

#endif // VENG_PBRPASS_HPP
