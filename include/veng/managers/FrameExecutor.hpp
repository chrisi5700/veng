//
// L4 driver — the per-frame GPU loop body, lifted out of `playground/main.cpp` ([[pass-draw-
// redesign]]). The reactive graph still describes *what* to render and when; this class is
// the *engine* that drives it each frame: slot computation, the pool's retirement window,
// the swap acquire/in-flight fence wait, the on_retired dispatch at slot reuse, the command
// buffer lifecycle, the graph execute, the queue submit (with sync straight from the
// `SwapchainManager::Frame` — no semaphore plumbing on graph edges anymore), and the
// on_submitted dispatch with a `PresentFrame` for `PresentNode`.
//
// What stays out: window event handling, the pacing toggle, the writer-thread CV (those are
// driver-side concerns — the executor handles a *frame*, not the loop). The pacing flag
// decides whether the swapchain image is fed as a per-frame dirty pulse (Continuous, set +
// resolve) or value-only (OnDemand, set_now after a pre-acquire resolve that idles on an
// empty plan). The optional `graph_mutex` is locked around graph mutations + resolves so the
// driver can mutate sources from another thread (the writer) without racing.
//

#ifndef VENG_FRAMEEXECUTOR_HPP
#define VENG_FRAMEEXECUTOR_HPP

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <utility>
#include <vector>
#include <veng/context/Context.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/gpu/Sink.hpp>
#include <veng/managers/CommandManager.hpp>
#include <veng/managers/SwapchainManager.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/ResourcePool.hpp>

namespace veng
{
class FrameExecutor
{
	 public:
	/// How the swapchain image source is fed each frame.
	///   Continuous: `set` it (dirties the source), then resolve — blit + present always run.
	///   OnDemand:   resolve first (the swapchain source is *not* on the demanded cone via dirt);
	///               an empty plan means nothing has changed and the frame is idled (no acquire/
	///               submit/present). On a non-empty plan, `set_now` writes the value without
	///               dirtying — the plan was decided before acquire.
	enum class Pacing : std::uint8_t
	{
		Continuous,
		OnDemand
	};

	enum class Status : std::uint8_t
	{
		Rendered,	   // frame submitted + present queued
		Idled,		   // OnDemand and the plan was empty — caller should sleep on its own signal
		OutOfDate,	   // swapchain is out of date — caller should rebuild
		AcquireFailed, // command-pool or CB-begin failure (catastrophic)
		NodeFailed	   // graph.execute returned false; submit skipped, failed node stays dirty
	};

	struct Frame
	{
		Status			 status;
		graph::FramePlan plan; // populated when status == Rendered; for stats / introspection
	};

	FrameExecutor(Context& context, SwapchainManager& swap, ResourcePool& pool, CommandManager& commands,
				  graph::Scheduler& scheduler, graph::TypedHandle<gpu::ImageRef> swapchain_handle,
				  std::size_t frames_in_flight) noexcept;

	/// Drive one frame. Returns the status + (for `Rendered`) the plan executed. `graph_mutex`,
	/// if non-null, is locked around the graph mutations + resolves the executor performs.
	[[nodiscard]] Frame run_frame(graph::Graph& graph, std::span<const graph::DataHandle> sinks, Pacing pacing,
								  std::mutex* graph_mutex = nullptr);

	/// Current monotonic frame index (number of frames started). Useful for stats / pacing.
	[[nodiscard]] std::uint64_t frame_index() const noexcept { return m_frame_index; }

	 private:
	Context*						  m_context;
	SwapchainManager*				  m_swap;
	ResourcePool*					  m_pool;
	CommandManager*					  m_commands;
	graph::Scheduler*				  m_scheduler;
	graph::TypedHandle<gpu::ImageRef> m_swapchain_handle;
	std::size_t						  m_frames_in_flight;
	std::uint64_t					  m_frame_index = 0;

	// Per-slot list of Sinks that ran in the frame that previously held that slot; fired in
	// on_retired at the next `acquire(slot)` (which waited the fence, so they have retired).
	std::vector<std::vector<gpu::Sink*>> m_pending_retire;
};
} // namespace veng

#endif // VENG_FRAMEEXECUTOR_HPP
