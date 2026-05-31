/**
 * @file
 * @author chris
 * @brief GPU color-picking pass assembled as a reactive sub-graph.
 *
 * @ref veng::passes::PickingPass renders every pickable object into an offscreen "id buffer" — each as a flat
 * color encoding its integer id, depth-tested so the frontmost wins — then reads back the pixel
 * under a queried point and reports which object is there.
 *
 * The readback is asynchronous: @ref veng::passes::PickingPass::pick records a request; the id buffer is copied
 * into a host-visible staging buffer and, once that frame's fence has signalled (the engine's
 * `on_retired` hook), the requested pixel is decoded and the callback fires — a few frames after
 * the call, never blocking the caller. This is the draw → readback → CPU → decode → callback
 * path.
 *
 * @code
 *   render   (passes/picking.{vert,frag}) — one draw per object, id encoded into an RGBA8 target
 *       └─ id_image
 *   readback (PickingReadbackNode) — copies id_image to host memory, decodes on retirement, and
 *                produces done_token (the edge the driver demands to keep the branch in the plan)
 * @endcode
 *
 * Reactivity: picking costs nothing while idle. @ref veng::passes::PickingPass::pick marks the readback node
 * dirty, which pulls it (and, if an object moved since, the id render) into the next frame's
 * plan; with no pending pick the branch is clean and drops out. The caller must demand
 * @ref veng::passes::PickingPass::done_token each frame (e.g. via `AppLoop::add_sink`) so the branch is allowed
 * into the plan when it does have work.
 *
 * Mesh layout: @ref veng::passes::PickingPass::add_object draws through `picking.vert`, whose reflected vertex
 * input is the engine's standard `{position, normal, color}`; only position is used. The `mvp`
 * edge is normally the same per-object transform the scene pass uses, so picks track the
 * on-screen geometry.
 *
 * Object ids: each object gets a caller-chosen id that must be >= 1 — id 0 is reserved for
 * "no hit" (the background clears to it). Pick coordinates are in framebuffer pixels; the id
 * image is sized by the same `screen` extent as the scene.
 *
 * @ingroup render_passes
 * @see PhongPass
 * @see PbrPass
 */

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
/**
 * @brief Outcome of a pick, delivered to the @ref veng::passes::PickingPass::pick callback once the readback retires.
 * @ingroup render_passes
 */
struct PickResult
{
	bool		  hit = false; ///< `false` when the pixel hit the background or was out-of-bounds.
	std::uint32_t id  = 0;	   ///< Picked object id (valid only when `hit` is `true`).
	std::uint32_t x =
		0; ///< The queried pixel column, echoed back from the original @ref veng::passes::PickingPass::pick call.
	std::uint32_t y =
		0; ///< The queried pixel row, echoed back from the original @ref veng::passes::PickingPass::pick call.
};

/// @cond INTERNAL
// The readback half of the pass: a GpuNode that copies the id image to host memory and decodes
// queued picks on retirement. Defined in the .cpp (the pass only holds a pointer to it).
class PickingReadbackNode;
/// @endcond

/**
 * @brief Id-render and async readback picking pass, wired as a reactive sub-graph.
 *
 * Builds the id-render + readback chain on a @ref veng::graph::Graph and exposes @ref add_object and
 * @ref pick. Non-owning: the graph owns the nodes; this object holds handles into them and the
 * readback node's retirement callback holds a pointer back to it — so this object must outlive
 * the graph it was built on, and it is neither copyable nor movable.
 *
 * @ingroup render_passes
 * @see PhongPass
 * @see PbrPass
 */
class PickingPass
{
	 public:
	/**
	 * @brief Wire the picking chain into @p graph.
	 *
	 * @p screen sizes the offscreen id target, so pick coordinates are in those framebuffer
	 * pixels. @p depth_format is the id pass's own depth buffer — the frontmost object wins per
	 * pixel. Add objects with @ref add_object, then demand @ref done_token each frame.
	 *
	 * @param graph        The render graph to wire nodes into.
	 * @param screen       Reactive extent edge; sizes the id image and depth buffer.
	 * @param depth_format Format of the owned depth buffer used for id-render depth testing.
	 */
	PickingPass(graph::Graph& graph, graph::TypedHandle<vk::Extent2D> screen,
				vk::Format depth_format = vk::Format::eD32Sfloat);

	PickingPass(const PickingPass&)			   = delete;
	PickingPass& operator=(const PickingPass&) = delete;
	PickingPass(PickingPass&&)				   = delete;
	PickingPass& operator=(PickingPass&&)	   = delete;
	~PickingPass()							   = default;

	/**
	 * @brief Register a pickable object.
	 *
	 * @p mesh and @p mvp are typically the same handles the scene pass uses for the same object,
	 * so picks track the on-screen geometry. @p id must be >= 1; 0 is reserved for "no hit".
	 * Safe at setup and at runtime — appends a draw and marks the id render dirty.
	 *
	 * @param mesh  `ValueData<gpu::MeshRef>` edge of the object's geometry.
	 * @param mvp   `ValueData<glm::mat4>` edge — the combined model-view-projection matrix.
	 * @param id    Caller-assigned object id (must be >= 1).
	 */
	void add_object(graph::DataHandle mesh, graph::DataHandle mvp, std::uint32_t id);

	/**
	 * @brief Request the object id at framebuffer pixel (@p x, @p y).
	 *
	 * @p on_result fires asynchronously once the readback for the capturing frame retires (a few
	 * frames later) — not on this call. Best called from the thread that drives the frame loop,
	 * which is the same thread @p on_result runs on.
	 *
	 * @param x         Framebuffer pixel column to query.
	 * @param y         Framebuffer pixel row to query.
	 * @param on_result Callback invoked with a @ref veng::passes::PickResult once the retirement completes.
	 */
	void pick(std::uint32_t x, std::uint32_t y, std::function<void(PickResult)> on_result);

	/**
	 * @brief `true` while a pick is in flight (requested but its callback has not yet fired).
	 *
	 * The async readback completes only on a frame's retirement, which needs `frames_in_flight`
	 * more frames to render — so a purely on-demand loop that idles after the capture frame would
	 * never retire it. Register this with `AppLoop::add_keep_alive` (or otherwise keep rendering
	 * while it is `true`) so the pipeline keeps pumping presenting frames until the pick resolves.
	 *
	 * @return `true` if at least one pick callback has not yet fired.
	 */
	[[nodiscard]] bool pending() const noexcept;

	/**
	 * @brief The edge the driver must demand each frame so the picking branch enters the plan.
	 *
	 * Add this to the executor's sink set (e.g. `AppLoop::add_sink`) so the readback node is
	 * allowed into the frame plan when a pick is pending.
	 *
	 * @return The `done_token` `ValueData<int>` edge produced by the readback node.
	 */
	[[nodiscard]] graph::DataHandle done_token() const noexcept { return m_done_token; }

	/**
	 * @brief The offscreen id buffer edge.
	 *
	 * Useful for introspection and debugging — e.g. blit it to screen to inspect the encoded id
	 * colors.
	 *
	 * @return `ValueData<gpu::ImageRef>` edge produced by the id-render node.
	 */
	[[nodiscard]] graph::DataHandle id_image() const noexcept { return m_id_image; }

	/// @return Handle to the id-render @ref veng::nodes::GraphicsNode in the graph.
	[[nodiscard]] graph::NodeHandle render_node() const noexcept { return m_render_node; }

	/// @return Handle to the readback @ref veng::passes::PickingReadbackNode in the graph.
	[[nodiscard]] graph::NodeHandle readback_node() const noexcept { return m_readback_node; }

	 private:
	graph::Graph*		 m_graph	= nullptr;
	nodes::GraphicsNode* m_render	= nullptr; ///< Id-render node (graph-owned); @ref add_object extends it.
	PickingReadbackNode* m_readback = nullptr; ///< Readback and decode node (graph-owned).
	graph::DataHandle	 m_id_image;
	graph::DataHandle	 m_done_token;
	graph::NodeHandle	 m_render_node;
	graph::NodeHandle	 m_readback_node;
};
} // namespace veng::passes

#endif // VENG_PASSES_PICKINGPASS_HPP
