//
// Created by chris on 5/30/26.
//
// First-class *sink* interface ([[pass-draw-redesign]]). A sink is a node with work that must
// happen around submit rather than inside the recorded command buffer: `vkQueuePresentKHR`
// after submit (PresentNode), a fence-gated host readback after the frame retires
// (ScreenshotNode, PickingReadbackNode). Those hooks used to live as no-op virtuals on *every*
// `GpuNode`, reached by a downcast in the driver — which meant "being a sink" was implicit and a
// node could silently fail to participate. Hoisting them onto a dedicated mixin makes sink-ness
// an explicit, opt-in capability: a node that wants post-submit / post-retire work inherits
// `gpu::Sink` and the hooks are right there; a node that does not never carries them.
//
// A sink is almost always also a `gpu::GpuNode` (it records, has edges, sits in the plan); it
// inherits both. The driver discovers sinks with a single `dynamic_cast<gpu::Sink*>` over the
// executed plan — a cross-cast that resolves only for nodes that actually opted in.
//

#ifndef VENG_SINK_HPP
#define VENG_SINK_HPP

#include <veng/gpu/SubmitContext.hpp>

namespace veng::gpu
{
class Sink
{
	 public:
	Sink()						 = default;
	Sink(const Sink&)			 = default;
	Sink& operator=(const Sink&) = default;
	Sink(Sink&&)				 = default;
	Sink& operator=(Sink&&)		 = default;
	virtual ~Sink()				 = default;

	/// Post-submit hook. After the frame's command buffer is ended + submitted on the graphics
	/// queue, the driver invokes this once per `Sink` in the executed plan. `PresentNode`
	/// overrides it to issue `vkQueuePresentKHR` (a queue op that must follow submit); a future
	/// video-encode sink would enqueue its encode here. Multiple sinks in one frame are peers.
	virtual void on_submitted(SubmitContext& ctx) noexcept { (void)ctx; }

	/// Post-retirement hook, for a sink needing the GPU work *complete* (not just submitted) —
	/// e.g. a CPU readback that must wait the slot's in-flight fence before mapping its staging
	/// buffer. The driver invokes it once per `Sink` of the frame that just retired in that slot,
	/// at the next `acquire(slot)` (which waited the fence). `ScreenshotNode` / the picking
	/// readback override it to decode the captured pixels.
	virtual void on_retired(SubmitContext& ctx) noexcept { (void)ctx; }
};
} // namespace veng::gpu

#endif // VENG_SINK_HPP
