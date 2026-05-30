//
// OutlinePass — a reusable screen-space "glow outline" effect, the first entry in veng's pass
// library. It renders the silhouette of the meshes you give it, blurs that mask with a
// separable Gaussian, and extracts the halo that spills *outside* the silhouette: a soft,
// colored outline. Both knobs the effect exposes are reactive — change the glow's size or
// color at runtime via set_width / set_color and only the affected nodes re-render next frame
// (the silhouette stays cached).
//
// A "pass" here is just a small sub-graph built onto the reactive Graph: the constructor
// declares three GraphicsNodes and wires them together, then the engine schedules and
// N-buffers them like any other node — the pass runs no GPU work itself.
//
//   silhouette (passes/outline_silhouette.{vert,frag}) — solid-white mask, one draw per mesh
//       └─ silhouette_image
//   blur       (passes/outline_fullscreen.vert + passes/outline_blur.frag) — horizontal Gaussian
//       └─ blurred_image
//   ring       (passes/outline_fullscreen.vert + passes/outline_ring.frag) — vertical Gaussian
//                 + (blurred − sharp) extract + tint → writes the caller's `output` edge
//
// The pass produces the glow image only — premultiplied, ready to be *added* over a scene.
// The caller composites it downstream (typically a fullscreen `scene + outline` pass), so
// several effects can accumulate into one composite. Feed the meshes whose silhouettes glow
// with add_mesh(mesh, mvp); the `mvp` edge is normally the same per-object transform the
// scene pass already uses for that mesh (handles are reactive, so a moving object updates the
// outline too).
//
// Mesh layout: add_mesh draws through outline_silhouette.vert, whose reflected vertex input is
// the engine's standard {position, normal, color} (matching MeshNode geometry / the example
// Geometry library). Meshes with a different layout would be misread — only position is used,
// but the full layout must be present so the reflected vertex stride matches the buffer.
//

#ifndef VENG_PASSES_OUTLINEPASS_HPP
#define VENG_PASSES_OUTLINEPASS_HPP

#include <glm/glm.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::passes
{
/// Starting look of the outline. Both fields stay live after construction (set_width /
/// set_color drive the same graph sources) — these are just the initial values.
struct OutlineConfig
{
	glm::vec3 color = {0.15F, 0.85F, 1.0F}; ///< glow tint (linear RGB); default cyan
	float	  width = 1.0F;					///< glow spread multiplier; 1.0 = default reach
};

/// Builds the silhouette → blur → ring chain on a Graph and exposes the size/color knobs.
/// Non-owning: the Graph owns every node and source it creates; an OutlinePass instance just
/// holds handles into them, so it must not outlive the Graph it was built on.
class OutlinePass
{
	 public:
	/// Wire the outline chain into `graph`, writing the glow into `output` (a
	/// `ValueData<gpu::ImageRef>` edge the caller composites downstream). `color_format` is the
	/// format of the internal targets and the output; `screen` sizes them and drives the
	/// resolution-independent blur taps. Add the outlined meshes with add_mesh afterwards.
	OutlinePass(graph::Graph& graph, vk::Format color_format, graph::TypedHandle<vk::Extent2D> screen,
				graph::DataHandle output, const OutlineConfig& config = {});

	/// Add a mesh (with its per-object MVP edge) to the silhouette. Safe both during setup and
	/// at runtime: it appends a draw and marks the silhouette node dirty, so the planner re-runs
	/// it on the next frame. `mvp` is a `ValueData<glm::mat4>` edge — typically the very handle
	/// the scene pass uses for the same object.
	void add_mesh(graph::DataHandle mesh, graph::DataHandle mvp);

	/// Reactively retint the glow. Queued to the next frame boundary; re-renders the ring node.
	void set_color(glm::vec3 color);

	/// Reactively rescale the glow's size (1.0 = default reach). Queued to the next frame
	/// boundary; re-renders both blur passes.
	void set_width(float width);

	// Node handles for plan introspection / telemetry (e.g. counting frames the branch ran).
	// Not needed for wiring — the chain is internal.
	[[nodiscard]] graph::NodeHandle silhouette_node() const noexcept { return m_silhouette_node; }
	[[nodiscard]] graph::NodeHandle blur_node() const noexcept { return m_blur_node; }
	[[nodiscard]] graph::NodeHandle ring_node() const noexcept { return m_ring_node; }

	 private:
	graph::Graph*				  m_graph	   = nullptr;
	nodes::GraphicsNode*		  m_silhouette = nullptr; // lives in the graph; add_mesh extends it
	graph::TypedHandle<float>	  m_width_src;			  // glow spread, fed to both blur passes
	graph::TypedHandle<glm::vec4> m_color_src;			  // glow tint (rgb; a padded for std430)
	graph::NodeHandle			  m_silhouette_node;
	graph::NodeHandle			  m_blur_node;
	graph::NodeHandle			  m_ring_node;
};
} // namespace veng::passes

#endif // VENG_PASSES_OUTLINEPASS_HPP
