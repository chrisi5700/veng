//
// PickingPass — GPU color-picking, packaged as a sub-graph on the reactive Graph (the second
// entry in veng's pass library). It renders every pickable object into an offscreen "id buffer"
// — each object as a flat color encoding its integer id, depth-tested so the frontmost wins —
// then reads back the pixel under a queried point and reports which object is there.
//
// The readback is asynchronous: pick(x, y, cb) records a request; the id buffer is copied into a
// host-visible buffer and, once that frame's fence has signalled (the engine's on_retired hook),
// the requested pixel is decoded and `cb(PickResult)` fires — a few frames after the call, never
// blocking the caller. This is the "draw -> readback -> cpu -> decode -> callback" path.
//
//   render   (passes/picking.{vert,frag}) — one draw per object, id encoded into an RGBA8 target
//       └─ id_image
//   readback (PickingReadbackNode) — copies id_image to host memory, decodes on retirement, and
//                produces done_token (the edge the driver demands to keep the branch in the plan)
//
// Reactivity: picking costs nothing while idle. pick() marks the readback node dirty, which pulls
// it (and, if an object moved since, the id render) into the next frame's plan; with no pending
// pick the branch is clean and drops out. The caller must demand done_token() each frame (e.g.
// AppLoop::add_sink) so the branch is allowed into the plan when it does have work.
//
// Mesh layout: add_object draws through picking.vert, whose reflected vertex input is the
// engine's standard {position, normal, color}; only position is used. The `mvp` edge is normally
// the same per-object transform the scene pass uses, so picks track the on-screen geometry.
//
// Ids: each object gets a caller-chosen id that must be >= 1 — id 0 is reserved for "no hit"
// (the background clears to it). pick coordinates are in framebuffer pixels (the id_image is
// sized by the same `screen` extent as the scene).
//

#ifndef VENG_PASSES_PICKINGPASS_HPP
#define VENG_PASSES_PICKINGPASS_HPP

#include <cstdint>
#include <functional>
#include <veng/rendergraph/Graph.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::nodes
{
class GraphicsNode;
}

namespace veng::passes
{
/// Outcome of a pick, delivered to the pick() callback once the readback retires.
struct PickResult
{
	bool		  hit = false; ///< false = background or out-of-bounds (no object at the pixel)
	std::uint32_t id  = 0;	   ///< picked object id (valid when `hit`)
	std::uint32_t x	  = 0;	   ///< the queried pixel, echoed back
	std::uint32_t y	  = 0;
};

// The readback half of the pass: a GpuNode that copies the id image to host memory and decodes
// queued picks on retirement. Defined in the .cpp (the pass only holds a pointer to it).
class PickingReadbackNode;

/// Builds the id-render + readback chain on a Graph and exposes add_object / pick. Non-owning:
/// the Graph owns the nodes; this object holds handles into them, and the readback node's
/// retirement callback captures it — so it must outlive the Graph it was built on, and it is
/// neither copyable nor movable.
class PickingPass
{
	 public:
	/// Wire the picking chain into `graph`. `screen` sizes the offscreen id target (so pick
	/// coordinates are in those framebuffer pixels); `depth_format` is the id pass's own depth
	/// buffer so the nearest object wins per pixel. Add objects with add_object, then have the
	/// driver demand done_token() each frame.
	PickingPass(graph::Graph& graph, graph::TypedHandle<vk::Extent2D> screen,
				vk::Format depth_format = vk::Format::eD32Sfloat);

	PickingPass(const PickingPass&)			   = delete;
	PickingPass& operator=(const PickingPass&) = delete;
	PickingPass(PickingPass&&)				   = delete;
	PickingPass& operator=(PickingPass&&)	   = delete;
	~PickingPass()							   = default;

	/// Register a pickable object: its `mesh` + per-object `mvp` edge (typically the same handle
	/// the scene pass uses) + a unique id (>= 1; 0 is reserved for "no hit"). Safe at setup and
	/// at runtime — appends a draw and marks the id render dirty.
	void add_object(graph::DataHandle mesh, graph::DataHandle mvp, std::uint32_t id);

	/// Request the id at framebuffer pixel (x, y). `on_result` fires asynchronously once the
	/// readback for the capturing frame retires (a few frames later) — not on this call. Best
	/// called from the thread that drives the frame loop (the same thread on_result runs on).
	void pick(std::uint32_t x, std::uint32_t y, std::function<void(PickResult)> on_result);

	/// True while a pick is in flight (requested but its callback has not yet fired). The async
	/// readback completes only on a frame's retirement, which needs `frames_in_flight` more frames
	/// to actually render — so a purely on-demand loop that idles after the capture frame would
	/// never retire it. Register this with `AppLoop::add_keep_alive` (or otherwise keep rendering
	/// while it is true) so the pipeline keeps pumping presenting frames until the pick resolves.
	[[nodiscard]] bool pending() const noexcept;

	/// The edge the driver must demand each frame (add it to the executor's sinks) so the
	/// picking branch can enter the plan when a pick is pending.
	[[nodiscard]] graph::DataHandle done_token() const noexcept { return m_done_token; }

	/// The offscreen id buffer (introspection / debug; e.g. blit it to inspect the id colors).
	[[nodiscard]] graph::DataHandle id_image() const noexcept { return m_id_image; }

	[[nodiscard]] graph::NodeHandle render_node() const noexcept { return m_render_node; }
	[[nodiscard]] graph::NodeHandle readback_node() const noexcept { return m_readback_node; }

	 private:
	graph::Graph*		 m_graph	= nullptr;
	nodes::GraphicsNode* m_render	= nullptr; // id render (graph-owned); add_object extends it
	PickingReadbackNode* m_readback = nullptr; // readback + decode (graph-owned)
	graph::DataHandle	 m_id_image;
	graph::DataHandle	 m_done_token;
	graph::NodeHandle	 m_render_node;
	graph::NodeHandle	 m_readback_node;
};
} // namespace veng::passes

#endif // VENG_PASSES_PICKINGPASS_HPP
