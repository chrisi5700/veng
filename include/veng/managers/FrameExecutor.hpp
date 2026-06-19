/**
 * @file
 * @author chris
 * @brief L4 driver — the per-frame GPU loop body executed by the engine each frame.
 *
 * The reactive graph still describes *what* to render and when; @ref veng::FrameExecutor is the
 * *engine* that drives it each frame: slot computation, the @ref veng::ResourcePool retirement
 * window, the swap acquire/in-flight fence wait, the `on_retired` dispatch at slot reuse,
 * the command buffer lifecycle, the graph execute, the queue submit (with sync straight from
 * the @ref veng::SwapchainManager::Frame — no semaphore plumbing on graph edges), and the
 * `on_submitted` dispatch with a `PresentFrame` for `PresentNode`.
 *
 * What stays out: window event handling, the pacing toggle, the writer-thread CV (those are
 * driver-side concerns — the executor handles a *frame*, not the loop). The pacing flag
 * decides whether the swapchain image is fed as a per-frame dirty pulse (`Continuous`: `set`
 * + resolve) or value-only (`OnDemand`: `set_now` after a pre-acquire resolve that idles on
 * an empty plan). The optional `graph_mutex` is locked around graph mutations + resolves so
 * the driver can mutate sources from another thread (the writer) without racing.
 *
 * @ingroup managers
 */

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
/**
 * @brief Drives the reactive render graph for one frame, from slot selection through
 *        present.
 *
 * @ingroup managers
 * @see SwapchainManager
 * @see CommandManager
 * @see ResourcePool
 */
class FrameExecutor
{
	 public:
	/**
	 * @brief How the swapchain image source is fed into the graph each frame.
	 *
	 * - `Continuous`: `set` the swapchain source (dirties it), then resolve — the blit and
	 *   present nodes always run regardless of scene changes.
	 * - `OnDemand`: resolve first (the swapchain source is not on the demanded cone via
	 *   dirt); an empty plan means nothing has changed and the frame is idled (no
	 *   acquire/submit/present). On a non-empty plan, `set_now` writes the freshly-acquired
	 *   image value without dirtying — the plan was decided before acquire.
	 *
	 * @ingroup managers
	 */
	enum class Pacing : std::uint8_t
	{
		Continuous, ///< Always acquire + submit + present; swapchain source dirties the plan.
		OnDemand	///< Idle the frame when nothing in the graph has changed.
	};

	/**
	 * @brief Outcome of a single @ref run_frame call.
	 * @ingroup managers
	 */
	enum class Status : std::uint8_t
	{
		Rendered,	   ///< Frame submitted and present queued.
		Idled,		   ///< `OnDemand` pacing and the plan was empty; caller should sleep on its own signal.
		OutOfDate,	   ///< Swapchain is out of date; caller should rebuild.
		AcquireFailed, ///< Command-pool or CB-begin failure (catastrophic).
		NodeFailed	   ///< `graph.execute` returned false; submit skipped, failed node stays dirty.
	};

	/**
	 * @brief Result returned by @ref run_frame.
	 * @ingroup managers
	 */
	struct Frame
	{
		Status			 status;
		graph::FramePlan plan; ///< Populated when status == `Rendered`; for stats/introspection.
	};

	/**
	 * @brief Construct the executor, borrowing all collaborators by reference.
	 * @param context          The engine @ref veng::Context.
	 * @param swap             The @ref veng::SwapchainManager that supplies per-frame images.
	 * @param pool             The @ref veng::ResourcePool whose retirement window is advanced each frame.
	 * @param commands         The @ref veng::CommandManager that records per-slot command buffers.
	 * @param scheduler        The graph scheduler used to execute the frame plan.
	 * @param swapchain_handle Graph handle for the typed `gpu::ImageRef` source that receives
	 *                         the acquired swapchain image each frame.
	 * @param frames_in_flight Number of in-flight frame slots (clamped to at least 1).
	 */
	FrameExecutor(Context& context, SwapchainManager& swap, ResourcePool& pool, CommandManager& commands,
				  graph::Scheduler& scheduler, graph::TypedHandle<gpu::ImageRef> swapchain_handle,
				  std::size_t frames_in_flight) noexcept;

	/**
	 * @brief Drive one frame through the full acquire → record → submit → present pipeline.
	 *
	 * @param graph       The reactive graph to resolve and execute.
	 * @param sinks       The set of sink handles whose demanded cone determines the frame plan.
	 * @param pacing      Whether to always submit (`Continuous`) or idle on an unchanged plan
	 *                    (`OnDemand`).
	 * @param graph_mutex If non-null, locked around graph mutations and resolves so a writer
	 *                    thread can safely mutate sources concurrently.
	 * @return The frame @ref Status plus the executed plan when status is `Rendered`.
	 */
	[[nodiscard]] Frame run_frame(graph::Graph& graph, std::span<const graph::DataHandle> sinks, Pacing pacing,
								  std::mutex* graph_mutex = nullptr);

	/**
	 * @brief Current monotonic frame index (number of frames started).
	 * @return The frame count, useful for stats and pacing decisions.
	 */
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

	/// Per-slot list of Sinks that ran in the frame that previously held that slot; fired in
	/// `on_retired` at the next `acquire(slot)` (which waited the fence, so they have retired).
	std::vector<std::vector<gpu::Sink*>> m_pending_retire;
};
} // namespace veng

#endif // VENG_FRAMEEXECUTOR_HPP
