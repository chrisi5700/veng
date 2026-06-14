/**
 * @file
 * @author chris
 * @brief Metallic-roughness PBR pass — the lit, textured counterpart to @ref veng::passes::PhongPass.
 *
 * @ref veng::passes::PbrPass renders registered objects with a Cook-Torrance BRDF over a single directional
 * light, reading the standard glTF material inputs (baseColor / normal / metallic-roughness /
 * emissive / occlusion textures plus scalar factors). It is the pass the glTF loader targets to
 * display real assets.
 *
 * Materials are registered up front (@ref veng::passes::PbrPass::add_material → index); objects reference a
 * material by index plus a mesh edge and a per-object model-matrix edge. Each material binds its
 * own descriptor set of five textures, so a multi-material model renders in one pass. The pass
 * does NOT own textures — the caller (glTF loader / example / test) keeps the textures alive; a
 * material slot with no texture must still be given a default (white / flat-normal) texture so the
 * shader stays uniform.
 *
 * Current scope: opaque geometry, single directional key light, optional clustered point lights
 * via @ref veng::passes::PbrPass::set_clustered_lights, no image-based lighting.
 *
 * @ingroup render_passes
 * @see PhongPass
 * @see OutlinePass
 * @see ClusteredLightEdges
 */

#ifndef VENG_PBRPASS_HPP
#define VENG_PBRPASS_HPP

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <veng/culling/Clusters.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <veng/rhi/Enums.hpp>

namespace veng::passes
{
/**
 * @brief Tunables for the directional key light and render state of @ref veng::passes::PbrPass.
 * @ingroup render_passes
 */
struct PbrConfig
{
	glm::vec3 light_direction = glm::normalize(glm::vec3(-0.4F, 1.0F, 0.6F)); ///< Direction from surface TO light.
	glm::vec3 light_color	  = glm::vec3(1.0F, 1.0F, 1.0F);				  ///< Linear-RGB light color.
	float	  light_intensity = 3.0F;										  ///< Multiplier applied to `light_color`.
	glm::vec3 ambient		  = glm::vec3(0.03F, 0.03F, 0.03F);				  ///< Constant ambient term (linear RGB).
	std::array<float, 4> clear_color = {0.02F, 0.03F, 0.05F, 1.0F};			  ///< RGBA background clear value.
	rhi::CullMode		 cull_mode	 = rhi::CullMode::BACK; ///< glTF assets use `eBack`; tests may use `eNone`.
	/// MSAA sample count, clamped to the device on first record (`e1` = off). The pass renders into
	/// a multisampled color + depth target and resolves to the single-sample image the output edge
	/// carries, so anti-aliasing is transparent to downstream consumers.
	rhi::SampleCount samples = rhi::SampleCount::X1;
};

/**
 * @brief glTF alpha handling for a material (glTF spec §3.9.3).
 *
 * `OPAQUE` ignores alpha. `MASK` discards fragments whose alpha falls below `alpha_cutoff` and
 * rides the opaque, depth-writing pipeline. `BLEND` alpha-blends in a separate,
 * depth-tested-but-not-written batch drawn after all opaque/mask geometry.
 *
 * @ingroup render_passes
 */
enum class AlphaMode : std::uint8_t
{
	Opaque, ///< Alpha ignored; fully opaque depth-writing draw.
	Mask,	///< Fragments below `PbrMaterial::alpha_cutoff` are discarded; depth-written.
	Blend,	///< Alpha-blended in a post-opaque batch; depth-tested but not written.
};

/**
 * @brief A material's five sampled-image edges plus its scalar factors.
 *
 * Each texture edge must resolve to a `ValueData<gpu::ImageRef>` (typically an
 * `assets::Texture` fed through a graph source). For any channel a model does not provide,
 * bind a 1×1 default so the shader stays uniform — white for baseColor / metallic-roughness /
 * emissive / occlusion, and a flat (128, 128, 255) normal for the normal map.
 *
 * @ingroup render_passes
 */
struct PbrMaterial
{
	graph::DataHandle base_color;  ///< RGBA, sRGB-encoded base color texture.
	graph::DataHandle normal;	   ///< Linear, tangent-space normal map.
	graph::DataHandle metal_rough; ///< Linear; glTF packing: G = roughness, B = metallic.
	graph::DataHandle emissive;	   ///< RGB, sRGB-encoded emissive texture.
	graph::DataHandle occlusion;   ///< Linear; R = ambient occlusion factor.

	glm::vec4 base_color_factor = glm::vec4(1.0F, 1.0F, 1.0F, 1.0F); ///< Multiplied with the base color texture sample.
	float	  metallic_factor	= 1.0F;								 ///< Multiplied with the metallic channel.
	float	  roughness_factor	= 1.0F;								 ///< Multiplied with the roughness channel.
	float	  normal_scale		= 1.0F;								 ///< Normal map intensity scale.
	float	  occlusion_strength = 1.0F;							 ///< Blend weight for the occlusion term.
	glm::vec3 emissive_factor	 = glm::vec3(0.0F, 0.0F, 0.0F);		 ///< Multiplied with the emissive texture sample.

	AlphaMode alpha_mode   = AlphaMode::Opaque;
	float	  alpha_cutoff = 0.5F; ///< Fragment discard threshold; consulted only for @ref AlphaMode::Mask.
};

/// @cond INTERNAL
class PbrRenderNode; // rendering half, defined in the .cpp
/// @endcond

/**
 * @brief Metallic-roughness PBR render pass wired as a reactive sub-graph.
 *
 * Non-owning: the @ref veng::graph::Graph owns the @ref veng::passes::PbrRenderNode; this object holds handles into
 * it. Neither copyable nor movable (it wires graph producers in its constructor and must outlive
 * the graph it was built on).
 *
 * @ingroup render_passes
 * @see PhongPass
 * @see OutlinePass
 * @see wire_clustered_lights
 */
class PbrPass
{
	 public:
	/**
	 * @brief Wire the PBR render node into @p graph.
	 *
	 * @p screen sizes the owned colour and depth targets. @p output is the lit-scene
	 * `ValueData<gpu::ImageRef>` edge the caller demands. @p view_proj and @p eye (xyz = camera
	 * position) drive the per-frame camera and lighting uniform.
	 *
	 * @param graph        The render graph to wire nodes into.
	 * @param color_format Format of the owned HDR colour target.
	 * @param depth_format Format of the owned depth target.
	 * @param screen       Reactive extent edge; sizes both targets each frame.
	 * @param output       `ValueData<gpu::ImageRef>` edge this pass writes the lit scene into.
	 * @param view_proj    Reactive combined view-projection matrix edge.
	 * @param eye          Reactive camera position edge (xyz = world-space eye; w unused).
	 * @param config       Directional light and pipeline tunables.
	 */
	PbrPass(graph::Graph& graph, rhi::Format color_format, rhi::Format depth_format,
			graph::TypedHandle<rhi::Extent2D> screen, graph::DataHandle output, graph::TypedHandle<glm::mat4> view_proj,
			graph::TypedHandle<glm::vec4> eye, const PbrConfig& config = {});

	PbrPass(const PbrPass&)			   = delete;
	PbrPass& operator=(const PbrPass&) = delete;
	PbrPass(PbrPass&&)				   = delete;
	PbrPass& operator=(PbrPass&&)	   = delete;
	~PbrPass()						   = default;

	/**
	 * @brief Register a material; the returned index is passed to @ref add_object.
	 * @param material The five texture edges and scalar factors for this material slot.
	 * @return Opaque material index for use with @ref add_object.
	 */
	[[nodiscard]] std::uint32_t add_material(const PbrMaterial& material);

	/**
	 * @brief Register an object for rendering.
	 * @param mesh     `ValueData<gpu::MeshRef>` edge of the object's geometry.
	 * @param model    `ValueData<glm::mat4>` edge — the per-object model matrix.
	 * @param material Material index returned by @ref add_material.
	 */
	void add_object(graph::DataHandle mesh, graph::DataHandle model, std::uint32_t material);

	/**
	 * @brief Enable clustered point lighting.
	 *
	 * @p view is the world-to-view matrix edge. @p lights is a `gpu::BufferRef` edge of
	 * world-space `culling::GpuLight` values. @p light_grid and @p light_index are the
	 * LightCullCpu outputs: per-cluster (offset, count) pairs and a flat light-index list,
	 * both `gpu::BufferRef` edges. @p grid must match the one the cull was built with.
	 *
	 * Optional — without this call the pass shades the directional key light and ambient only.
	 * Use @ref veng::passes::wire_clustered_lights to build the three buffer edges.
	 *
	 * @param view         World-to-view matrix edge (used for the fragment's depth-slice lookup).
	 * @param lights       `gpu::BufferRef` edge — world-space `culling::GpuLight[]`.
	 * @param light_grid   `gpu::BufferRef` edge — per-cluster (offset, count) pairs.
	 * @param light_index  `gpu::BufferRef` edge — flat light index list.
	 * @param grid         Froxel grid configuration; must match the one used by the cull node.
	 */
	void set_clustered_lights(graph::DataHandle view, graph::DataHandle lights, graph::DataHandle light_grid,
							  graph::DataHandle light_index, const culling::ClusterGrid& grid);

	/// @return The lit-scene `ValueData<gpu::ImageRef>` edge this pass produces.
	[[nodiscard]] graph::DataHandle output() const noexcept { return m_output; }

	/// @return Handle to the underlying @ref veng::passes::PbrRenderNode in the graph.
	[[nodiscard]] graph::NodeHandle node() const noexcept { return m_node; }

	 private:
	PbrRenderNode*	  m_render = nullptr; ///< Graph-owned; the pass holds a non-owning pointer.
	graph::DataHandle m_output;
	graph::NodeHandle m_node;
};
} // namespace veng::passes

#endif // VENG_PBRPASS_HPP
