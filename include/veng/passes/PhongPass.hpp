/**
 * @file
 * @author chris
 * @brief Per-pixel Blinn-Phong shading pass with order-sort-free transparency.
 *
 * @ref veng::passes::PhongPass renders registered objects with Blinn-Phong shading. Each object carries its own
 * mesh, model transform, and RGBA color; an alpha < 1 marks it translucent.
 *
 * A custom `GpuNode` is used instead of a @ref veng::nodes::GraphicsNode because a
 * @ref veng::nodes::GraphicsNode always clears and owns its target, preventing two nodes from sharing
 * one depth buffer — and depth-tested transparency over opaque geometry needs exactly that: the
 * transparent draws must test against the opaque depth without writing it. @ref veng::passes::PhongPass
 * therefore owns a single color + depth target and records the whole ordered draw itself, in one
 * render pass, with three pipelines built from the same shader pair (`passes/phong.{vert,frag}`)
 * differing only in cull / depth-write / blend state:
 *
 *   1. **opaque**            — back-cull, depth-write, no blend
 *   2. **transparent BACK**  — front-cull (far faces), depth-test only, alpha blend
 *   3. **transparent FRONT** — back-cull (near faces), depth-test only, alpha blend
 *
 * Drawing each translucent object's far faces then its near faces (the classic two-pass
 * back-then-front technique) gives correct see-through shading for convex objects with no sorting
 * or OIT machinery. It does NOT sort *between* overlapping transparent objects — for an object
 * viewer this is fine; many overlapping translucents would want a depth sort or WBOIT.
 *
 * Front-face winding is `eClockwise`, not `eCounterClockwise`: the geometry is CCW-front in world
 * space, but the camera projection negates Y for Vulkan clip space (`proj[1][1] *= -1`), which
 * flips triangle winding in framebuffer space — where Vulkan decides facing. (See PhongPass.cpp
 * for the full derivation; contrast with @ref veng::passes::PbrPass which uses `eCounterClockwise` for CCW-front
 * glTF geometry.)
 *
 * Shading is push-constant-only (no descriptors): a 160-byte vertex block
 * `{model, view_proj, color, eye+shininess}`; the fragment stage reads only varyings. The eye
 * position comes from the camera's reactive source so specular highlights track the viewer. The
 * mesh's vertex color is multiplied by the per-object push color, so feeding white meshes makes
 * the per-object color authoritative.
 *
 * @ingroup render_passes
 * @see PbrPass
 * @see OutlinePass
 */

#ifndef VENG_PASSES_PHONGPASS_HPP
#define VENG_PASSES_PHONGPASS_HPP

#include <array>
#include <glm/glm.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/rhi/Enums.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::passes
{
/**
 * @brief Tunables for the look of @ref veng::passes::PhongPass.
 * @ingroup render_passes
 */
struct PhongConfig
{
	float shininess = 48.0F; ///< Default Blinn-Phong specular exponent (gloss). Overridden per-object when >= 0.
	std::array<float, 4> clear_color = {0.02F, 0.03F, 0.05F,
										1.0F}; ///< RGBA background the colour target is cleared to.
	/// MSAA sample count, clamped to the device on first record (`e1` = off). The pass renders into
	/// a multisampled target and resolves to the single-sample image the output edge carries.
	rhi::SampleCount samples = rhi::SampleCount::X1;
};

/// @cond INTERNAL
class PhongRenderNode; // Rendering node, defined in the .cpp; the pass holds a pointer to it.
/// @endcond

/**
 * @brief Blinn-Phong render pass with convex-object transparency, wired as a reactive sub-graph.
 *
 * Builds the Phong render node on a @ref veng::graph::Graph and exposes @ref add_object. Non-owning:
 * the graph owns the node; this object holds handles into it. Neither copyable nor movable (it
 * wires graph producers in its constructor and must outlive the graph it was built on).
 *
 * @ingroup render_passes
 * @see PbrPass
 * @see OutlinePass
 */
class PhongPass
{
	 public:
	/**
	 * @brief Wire the Phong render node into @p graph.
	 *
	 * @p color_format and @p depth_format are the formats of the owned targets. @p screen sizes
	 * them. @p output is the lit-scene `ValueData<gpu::ImageRef>` edge this pass produces, left
	 * in `eColorAttachmentOptimal` for a downstream sampler or composite — the same contract as
	 * @ref veng::nodes::GraphicsNode. @p view_proj and @p eye are the camera's reactive sources; eye is
	 * a `vec4` with xyz = world-space eye position.
	 *
	 * @param graph        The render graph to wire the node into.
	 * @param color_format Format of the owned colour target.
	 * @param depth_format Format of the owned depth target.
	 * @param screen       Reactive extent edge; sizes both targets each frame.
	 * @param output       `ValueData<gpu::ImageRef>` edge this pass writes the lit scene into.
	 * @param view_proj    Reactive combined view-projection matrix edge.
	 * @param eye          Reactive camera position edge (xyz = world-space eye; w unused).
	 * @param config       Shininess and clear-color tunables.
	 */
	PhongPass(graph::Graph& graph, rhi::Format color_format, rhi::Format depth_format,
			  graph::TypedHandle<vk::Extent2D> screen, graph::DataHandle output,
			  graph::TypedHandle<glm::mat4> view_proj, graph::TypedHandle<glm::vec4> eye,
			  const PhongConfig& config = {});

	PhongPass(const PhongPass&)			   = delete;
	PhongPass& operator=(const PhongPass&) = delete;
	PhongPass(PhongPass&&)				   = delete;
	PhongPass& operator=(PhongPass&&)	   = delete;
	~PhongPass()						   = default;

	/**
	 * @brief Register an object for rendering.
	 *
	 * @p mesh is a `ValueData<gpu::MeshRef>` edge (typically from a @ref veng::nodes::GraphicsNode).
	 * @p model is a `ValueData<glm::mat4>` per-object transform edge. `color.a < 1` marks the
	 * object transparent — it is drawn back-faces-then-front-faces, alpha-blended over the opaque
	 * pass. @p shininess is the per-object Blinn-Phong specular exponent; a negative value (the
	 * default) falls back to @ref veng::passes::PhongConfig::shininess. Safe to call before or after the loop
	 * starts: it appends a draw and marks the node dirty so the planner re-runs it next frame.
	 *
	 * @param mesh      `ValueData<gpu::MeshRef>` edge of the object's geometry.
	 * @param model     `ValueData<glm::mat4>` edge — the per-object model matrix.
	 * @param color     RGBA tint (linear). `color.a < 1` enables transparent rendering.
	 * @param shininess Per-object specular exponent; negative → use @ref veng::passes::PhongConfig::shininess.
	 */
	void add_object(graph::DataHandle mesh, graph::DataHandle model, glm::vec4 color, float shininess = -1.0F);

	/// @return The lit-scene `ValueData<gpu::ImageRef>` edge this pass produces.
	[[nodiscard]] graph::DataHandle output() const noexcept { return m_output; }

	/// @return Handle to the underlying @ref veng::passes::PhongRenderNode in the graph.
	[[nodiscard]] graph::NodeHandle node() const noexcept { return m_node; }

	 private:
	PhongRenderNode*  m_render = nullptr; ///< Graph-owned; @ref add_object extends it.
	graph::DataHandle m_output;
	graph::NodeHandle m_node;
};
} // namespace veng::passes

#endif // VENG_PASSES_PHONGPASS_HPP
