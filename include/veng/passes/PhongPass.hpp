//
// PhongPass — per-pixel Blinn-Phong shading with order-independent-free transparency, packaged
// as a sub-graph on the reactive Graph (a pass in veng's pass library). Each registered object
// carries its own mesh, model transform and RGBA color; an alpha < 1 marks it translucent.
//
// Why a custom GpuNode rather than a GraphicsNode: a GraphicsNode always clears and owns its
// target, so two of them cannot share one depth buffer — and depth-tested transparency over
// opaque geometry needs exactly that (the transparent draws must test against the opaque depth
// without writing it). PhongPass therefore owns a single color + depth target and records the
// whole ordered draw itself, in one render pass, with three pipelines built from the SAME shader
// pair (passes/phong.{vert,frag}) differing only in cull / depth-write / blend state:
//
//   1. opaque            — back-cull, depth-write,    no blend
//   2. transparent BACK  — front-cull (far faces),    depth-test only, alpha blend
//   3. transparent FRONT — back-cull  (near faces),   depth-test only, alpha blend
//
// Drawing each translucent object's far faces then its near faces (the classic two-pass
// back-then-front technique) gives correct see-through shading for convex objects with no
// sorting or OIT machinery. It does NOT sort *between* overlapping transparent objects — for an
// object viewer that is fine; many overlapping translucents would want a depth sort or WBOIT.
//
// Front-face winding is eClockwise, not eCounterClockwise: the geometry is CCW-front in world
// space, but the camera projection negates Y for Vulkan clip space (proj[1][1] *= -1), which
// flips triangle winding in framebuffer space — where Vulkan decides facing. (See PhongPass.cpp.)
//
// Shading is push-constant-only (no descriptors): a 160-byte vertex block {model, view_proj,
// color, eye+shininess}; the fragment stage reads only varyings. The eye position comes from the
// camera's reactive eye_pos() source so specular highlights track the viewer. The mesh's vertex
// color is multiplied by the per-object push color, so feeding white meshes makes the per-object
// color authoritative — the caller "sets the color and mesh themselves" via add_object().
//

#ifndef VENG_PASSES_PHONGPASS_HPP
#define VENG_PASSES_PHONGPASS_HPP

#include <array>
#include <glm/glm.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::passes
{
/// Tunables for the look of the pass.
struct PhongConfig
{
	float				 shininess	 = 48.0F;						///< default Blinn-Phong specular exponent (gloss)
	std::array<float, 4> clear_color = {0.02F, 0.03F, 0.05F, 1.0F}; ///< background the target is cleared to
};

// The rendering node, defined in the .cpp (the pass only holds a pointer to it).
class PhongRenderNode;

/// Builds the Phong render node on a Graph and exposes add_object. Non-owning: the Graph owns the
/// node; this holds handles into it. Neither copyable nor movable (it wires graph producers in
/// its ctor and must outlive the Graph it was built on).
class PhongPass
{
	 public:
	/// Wire the pass into `graph`. `color_format`/`depth_format` are the owned target's formats;
	/// `screen` sizes it; `output` is the lit-scene `ValueData<gpu::ImageRef>` edge this pass
	/// produces (left in eColorAttachmentOptimal for a downstream sampler/composite, same contract
	/// as GraphicsNode). `view_proj` and `eye` are the camera's reactive sources (eye is a vec4,
	/// xyz = world-space eye). Add objects with add_object before/while running.
	PhongPass(graph::Graph& graph, vk::Format color_format, vk::Format depth_format,
			  graph::TypedHandle<vk::Extent2D> screen, graph::DataHandle output,
			  graph::TypedHandle<glm::mat4> view_proj, graph::TypedHandle<glm::vec4> eye,
			  const PhongConfig& config = {});

	PhongPass(const PhongPass&)			   = delete;
	PhongPass& operator=(const PhongPass&) = delete;
	PhongPass(PhongPass&&)				   = delete;
	PhongPass& operator=(PhongPass&&)	   = delete;
	~PhongPass()						   = default;

	/// Register an object: its `mesh` edge (a `ValueData<gpu::MeshRef>` from a MeshNode), its
	/// `model` transform edge (a `ValueData<glm::mat4>`), and an RGBA `color`. `color.a < 1` marks
	/// the object transparent (drawn back-faces-then-front-faces, alpha-blended over the opaque
	/// pass). `shininess` is the per-object Blinn-Phong specular exponent (gloss tightness); a
	/// negative value (the default) uses `PhongConfig::shininess`. Call before adding more frames;
	/// appends a draw and marks the node dirty.
	void add_object(graph::DataHandle mesh, graph::DataHandle model, glm::vec4 color, float shininess = -1.0F);

	/// The lit-scene image edge this pass produces.
	[[nodiscard]] graph::DataHandle output() const noexcept { return m_output; }

	[[nodiscard]] graph::NodeHandle node() const noexcept { return m_node; }

	 private:
	graph::Graph*	  m_graph  = nullptr;
	PhongRenderNode*  m_render = nullptr; // graph-owned; add_object extends it
	graph::DataHandle m_output;
	graph::NodeHandle m_node;
};
} // namespace veng::passes

#endif // VENG_PASSES_PHONGPASS_HPP
