/**
 * @file
 * @author chris
 * @brief First-class sink mixin for nodes with work that must happen around submit.
 *
 * A sink is a node with work that must happen around submit rather than inside the recorded
 * command buffer: `vkQueuePresentKHR` after submit (`PresentNode`), a fence-gated host
 * readback after the frame retires (`ScreenshotNode`, `PickingReadbackNode`). Those hooks used
 * to live as no-op virtuals on every `GpuNode`, reached by a downcast in the driver — which
 * meant "being a sink" was implicit and a node could silently fail to participate. Hoisting
 * them onto a dedicated mixin makes sink-ness an explicit, opt-in capability: a node that wants
 * post-submit or post-retire work inherits @ref veng::gpu::Sink and the hooks are right there; a node that
 * does not never carries them.
 *
 * A sink is almost always also a @ref veng::gpu::GpuNode (it records, has edges, sits in the plan); it
 * inherits both. The driver discovers sinks with a single `dynamic_cast<gpu::Sink*>` over the
 * executed plan — a cross-cast that resolves only for nodes that actually opted in.
 *
 * @ingroup gpu_handles
 */

#ifndef VENG_SINK_HPP
#define VENG_SINK_HPP

#include <veng/gpu/SubmitContext.hpp>

namespace veng::gpu
{
/**
 * @brief Mixin base for GPU nodes that need post-submit or post-retire callbacks.
 *
 * Inherit alongside @ref veng::gpu::GpuNode to opt into `on_submitted` and/or `on_retired`. The driver
 * discovers participating nodes via `dynamic_cast<gpu::Sink*>` over the executed plan.
 *
 * @ingroup gpu_handles
 * @see GpuNode
 * @see SubmitContext
 */
class Sink
{
	 public:
	Sink()						 = default;
	Sink(const Sink&)			 = default;
	Sink& operator=(const Sink&) = default;
	Sink(Sink&&)				 = default;
	Sink& operator=(Sink&&)		 = default;
	virtual ~Sink()				 = default;

	/**
	 * @brief Post-submit hook called once per `Sink` after the frame's command buffer is submitted.
	 *
	 * After the frame's command buffer is ended and submitted on the graphics queue, the driver
	 * invokes this once per @ref veng::gpu::Sink in the executed plan. `PresentNode` overrides it to issue
	 * `vkQueuePresentKHR` (a queue op that must follow submit); a future video-encode sink would
	 * enqueue its encode here. Multiple sinks in one frame are peers.
	 *
	 * @param ctx The post-submit context providing graph data access and present-frame info.
	 */
	virtual void on_submitted(SubmitContext& ctx) noexcept { (void)ctx; }

	/**
	 * @brief Post-retirement hook called once per `Sink` after the GPU work for a slot completes.
	 *
	 * For sinks that need the GPU work fully complete (not just submitted) — e.g. a CPU readback
	 * that must wait on the slot's in-flight fence before mapping its staging buffer. The driver
	 * invokes this once per @ref veng::gpu::Sink of the frame that just retired in a given slot, at the next
	 * `acquire(slot)` call (which waited the fence). `ScreenshotNode` and the picking readback
	 * override it to decode the captured pixels.
	 *
	 * @param ctx The post-submit context for the retiring frame slot.
	 */
	virtual void on_retired(SubmitContext& ctx) noexcept { (void)ctx; }
};
} // namespace veng::gpu

#endif // VENG_SINK_HPP
