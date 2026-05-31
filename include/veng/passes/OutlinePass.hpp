/**
 * @file
 * @author chris
 * @brief Screen-space "glow outline" effect assembled as a reactive sub-graph.
 *
 * @ref veng::passes::OutlinePass renders the silhouette of registered meshes, blurs that mask with a separable
 * Gaussian, and extracts the halo that spills outside the silhouette — a soft, colored outline.
 * Both knobs the effect exposes are reactive: change the glow's size or color at runtime via
 * @ref veng::passes::OutlinePass::set_width / @ref veng::passes::OutlinePass::set_color and only the affected nodes re-render
 * next frame (the silhouette stays cached).
 *
 * A "pass" here is a small sub-graph built onto the reactive @ref veng::graph::Graph — the constructor
 * declares three @ref veng::nodes::GraphicsNode instances and wires them together, then the engine
 * schedules and N-buffers them like any other node — the pass object itself records no GPU work.
 *
 * @code
 *   silhouette (passes/outline_silhouette.{vert,frag}) — solid-white mask, one draw per mesh
 *       └─ silhouette_image
 *   blur       (passes/outline_fullscreen.vert + passes/outline_blur.frag) — horizontal Gaussian
 *       └─ blurred_image
 *   ring       (passes/outline_fullscreen.vert + passes/outline_ring.frag) — vertical Gaussian
 *                 + (blurred − sharp) extract + tint → writes the caller's `output` edge
 * @endcode
 *
 * The pass produces the glow image only — premultiplied, ready to be added over a scene. The
 * caller composites it downstream (typically a fullscreen `scene + outline` pass), so several
 * effects can accumulate into one composite. Feed the meshes whose silhouettes glow with
 * @ref veng::passes::OutlinePass::add_mesh; the `mvp` edge is normally the same per-object transform the scene
 * pass already uses for that mesh (handles are reactive, so a moving object updates the outline).
 *
 * Mesh layout: @ref veng::passes::OutlinePass::add_mesh draws through `outline_silhouette.vert`, whose reflected
 * vertex input is the engine's standard `{position, normal, color}` (matching @ref veng::nodes::GraphicsNode
 * geometry). Meshes with a different layout would be misread — only position is used, but the full
 * layout must be present so the reflected vertex stride matches the buffer.
 *
 * @ingroup render_passes
 * @see PbrPass
 * @see PhongPass
 */

#ifndef VENG_PASSES_OUTLINEPASS_HPP
#define VENG_PASSES_OUTLINEPASS_HPP

#include <glm/glm.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::passes
{
/**
 * @brief Initial configuration for an @ref veng::passes::OutlinePass.
 *
 * Both fields stay live after construction — @ref veng::passes::OutlinePass::set_width and
 * @ref veng::passes::OutlinePass::set_color drive the same graph sources. These are just the initial values.
 *
 * @ingroup render_passes
 */
struct OutlineConfig
{
	glm::vec3 color = {0.15F, 0.85F, 1.0F}; ///< Glow tint in linear RGB; default cyan.
	float	  width = 1.0F;					///< Glow spread multiplier; 1.0 = default reach.
};

/**
 * @brief Silhouette-blur-ring outline effect wired as a reactive sub-graph.
 *
 * Builds the silhouette → blur → ring chain on a @ref veng::graph::Graph and exposes size and color
 * knobs. Non-owning: the graph owns every node and source it creates; an @ref veng::passes::OutlinePass instance
 * only holds handles into them, so it must not outlive the graph it was built on.
 *
 * @ingroup render_passes
 * @see PbrPass
 * @see PhongPass
 */
class OutlinePass
{
	 public:
	/**
	 * @brief Wire the outline chain into @p graph, writing the glow into @p output.
	 *
	 * @p output must be a `ValueData<gpu::ImageRef>` edge that the caller composites downstream.
	 * @p color_format is the format of the internal targets and the output. @p screen sizes them
	 * and drives the resolution-independent blur taps. Add the outlined meshes with
	 * @ref add_mesh afterwards.
	 *
	 * @param graph        The render graph to wire nodes into.
	 * @param color_format Format of the internal silhouette/blur images and the output.
	 * @param screen       Reactive extent edge; resizing the window updates the blur tap spacing.
	 * @param output       `ValueData<gpu::ImageRef>` edge this pass writes its glow result into.
	 * @param config       Initial color and width; both are live and adjustable after construction.
	 */
	OutlinePass(graph::Graph& graph, vk::Format color_format, graph::TypedHandle<vk::Extent2D> screen,
				graph::DataHandle output, const OutlineConfig& config = {});

	/**
	 * @brief Add a mesh (with its per-object MVP edge) to the silhouette.
	 *
	 * Safe both during setup and at runtime: it appends a draw and marks the silhouette node
	 * dirty, so the planner re-runs it on the next frame. @p mvp is a `ValueData<glm::mat4>`
	 * edge — typically the very handle the scene pass uses for the same object.
	 *
	 * @param mesh  `ValueData<gpu::MeshRef>` edge of the mesh to outline.
	 * @param mvp   `ValueData<glm::mat4>` edge — the per-object model-view-projection matrix.
	 */
	void add_mesh(graph::DataHandle mesh, graph::DataHandle mvp);

	/**
	 * @brief Reactively retint the glow.
	 *
	 * Queued to the next frame boundary; re-renders the ring node only.
	 *
	 * @param color New glow tint in linear RGB.
	 */
	void set_color(glm::vec3 color);

	/**
	 * @brief Reactively rescale the glow's spread.
	 *
	 * Queued to the next frame boundary; re-renders both the horizontal and vertical blur nodes.
	 *
	 * @param width Glow spread multiplier (1.0 = default reach).
	 */
	void set_width(float width);

	/// @brief Node handle for plan introspection and telemetry (e.g. counting frames the branch ran).
	/// @return Handle to the silhouette @ref veng::nodes::GraphicsNode in the graph.
	[[nodiscard]] graph::NodeHandle silhouette_node() const noexcept { return m_silhouette_node; }

	/// @brief Node handle for plan introspection and telemetry.
	/// @return Handle to the horizontal-blur @ref veng::nodes::GraphicsNode in the graph.
	[[nodiscard]] graph::NodeHandle blur_node() const noexcept { return m_blur_node; }

	/// @brief Node handle for plan introspection and telemetry.
	/// @return Handle to the ring-extraction @ref veng::nodes::GraphicsNode in the graph.
	[[nodiscard]] graph::NodeHandle ring_node() const noexcept { return m_ring_node; }

	 private:
	graph::Graph*				  m_graph	   = nullptr;
	nodes::GraphicsNode*		  m_silhouette = nullptr; ///< Lives in the graph; @ref add_mesh extends it.
	graph::TypedHandle<float>	  m_width_src;			  ///< Glow spread source, fed to both blur passes.
	graph::TypedHandle<glm::vec4> m_color_src;			  ///< Glow tint source (rgb; a padded for std430).
	graph::NodeHandle			  m_silhouette_node;
	graph::NodeHandle			  m_blur_node;
	graph::NodeHandle			  m_ring_node;
};
} // namespace veng::passes

#endif // VENG_PASSES_OUTLINEPASS_HPP
