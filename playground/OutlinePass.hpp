//
// OutlinePass — the silhouette → horizontal blur → vertical blur + ring extraction chain that
// produces the demo's "outline glow" image, packaged as a reusable building block on the
// reactive graph (a composability test of the engine: build a sub-graph in a function, hand
// the caller back something they only have to feed *meshes* to). The function does not run
// any GPU work itself — it just declares three `GraphicsNode`s on the supplied `graph` and
// wires them together; the engine schedules and N-buffers them like any other node.
//
// What it builds:
//   silhouette (mesh.vert + silhouette.frag — solid white mask, one draw per mesh added)
//       └─ silhouette_image (sampled below)
//   blur_h    (fullscreen.vert + blur_h.frag — horizontal Gaussian)
//       └─ blurred_h_image (sampled below)
//   ring      (fullscreen.vert + ring.frag — vertical Gaussian + (blurred − sharp) extract)
//       └─ writes the caller-supplied `output` ImageRef edge
//
// What the caller still owns: the `output` edge (the ring image), so the result can be wired
// into a composite/blit downstream. `pool` is taken for forward-compat (the engine owns
// transient resources and the pool is the conduit); construction itself does not touch it.
//

#ifndef VENG_OUTLINEPASS_HPP
#define VENG_OUTLINEPASS_HPP

#include <glm/glm.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/ResourcePool.hpp>
#include <vulkan/vulkan.hpp>

namespace demo
{
struct OutlinePass
{
	/// Add a mesh (with its per-object MVP) to the silhouette draw list. The outline glow will
	/// follow the silhouette of every mesh added here. Call before the first frame is resolved —
	/// extends the silhouette node's input set so the planner sees the new draws on next resolve.
	void add_mesh(veng::graph::DataHandle mesh, veng::graph::DataHandle mvp);

	/// Node handles for stats / plan introspection (e.g. counting how many frames the outline
	/// branch ran). The caller does NOT use these to feed graph wiring — the chain is internal.
	[[nodiscard]] veng::graph::NodeHandle silhouette_node() const noexcept { return m_silhouette_node; }
	[[nodiscard]] veng::graph::NodeHandle blur_h_node() const noexcept { return m_blur_h_node; }
	[[nodiscard]] veng::graph::NodeHandle ring_node() const noexcept { return m_ring_node; }

	 private:
	friend OutlinePass create_outline_pass(veng::graph::Graph&, veng::ResourcePool&, vk::Format,
										   veng::graph::TypedHandle<vk::Extent2D>, veng::graph::DataHandle);
	OutlinePass(veng::nodes::GraphicsNode* silhouette, veng::graph::NodeHandle silhouette_node,
				veng::graph::NodeHandle blur_h_node, veng::graph::NodeHandle ring_node) noexcept
		: m_silhouette(silhouette)
		, m_silhouette_node(silhouette_node)
		, m_blur_h_node(blur_h_node)
		, m_ring_node(ring_node)
	{
	}

	veng::nodes::GraphicsNode* m_silhouette = nullptr; // live in the graph; add_mesh extends it
	veng::graph::NodeHandle	   m_silhouette_node;
	veng::graph::NodeHandle	   m_blur_h_node;
	veng::graph::NodeHandle	   m_ring_node;
};

/// Build the outline chain into `graph`: silhouette → horizontal blur → vertical blur + ring
/// extract, writing the ring (glow) into `output`. The caller then adds the meshes whose
/// silhouettes contribute via `pass.add_mesh(mesh, mvp)` and wires `output` downstream
/// (typically as a sampled input to a compositor). `format` is the color format for the
/// internal targets; `screen` drives both the target sizes and the blur taps (1 / extent in
/// UV space). `pool` is passed through unchanged for forward compatibility.
OutlinePass create_outline_pass(veng::graph::Graph& graph, veng::ResourcePool& pool, vk::Format format,
								veng::graph::TypedHandle<vk::Extent2D> screen, veng::graph::DataHandle output);
} // namespace demo

#endif // VENG_OUTLINEPASS_HPP
