/**
 * @file
 * @author chris
 * @brief Headless one-shot graph executor — drive a render graph to completion with no swapchain.
 *
 * A peer of @ref veng::FrameExecutor for the offscreen case (no window, no present). @ref run_once
 * advances the resource pool, records the demanded plan into a one-shot command buffer, submits it,
 * and fires the @ref veng::gpu::Sink `on_submitted` / `on_retired` hooks around a CPU-side fence
 * wait — so a @ref veng::nodes::ScreenshotNode has written its file by the time it returns. This is
 * exactly the harness a caller would otherwise hand-write to capture an offscreen render.
 *
 * @ingroup managers
 */

#ifndef VENG_HEADLESSEXECUTOR_HPP
#define VENG_HEADLESSEXECUTOR_HPP

#include <expected>
#include <span>
#include <veng/context/Context.hpp>
#include <veng/managers/CommandManager.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/ResourcePool.hpp>
#include <vulkan/vulkan.hpp>

namespace veng
{
/**
 * @brief Drives a render graph for a single synchronous, swapchain-free frame.
 *
 * Borrows all collaborators by reference. Build a graph whose sink is e.g. a
 * @ref veng::nodes::ScreenshotNode, then @ref run_once it: offscreen capture without the windowed
 * per-frame loop. Single-buffered (always frame slot 0); it waits the GPU idle each call.
 *
 * @ingroup managers
 * @see FrameExecutor
 */
class HeadlessExecutor
{
	 public:
	/**
	 * @brief Construct the executor, borrowing all collaborators.
	 * @param context   The engine @ref veng::Context (device + graphics queue).
	 * @param pool      The @ref veng::ResourcePool, advanced once per @ref run_once.
	 * @param commands  The @ref veng::CommandManager that records the one-shot command buffer.
	 * @param scheduler The graph scheduler used to execute the plan.
	 */
	HeadlessExecutor(Context& context, ResourcePool& pool, CommandManager& commands,
					 graph::Scheduler& scheduler) noexcept;

	/**
	 * @brief Resolve @p sinks, record + submit their plan, and block until the GPU finishes.
	 *
	 * Fires every plan node's @ref veng::gpu::Sink `on_submitted` hook after submit and `on_retired`
	 * after the fence signals (so a screenshot sink writes its file before this returns).
	 *
	 * @param graph The reactive graph to resolve and execute.
	 * @param sinks The sink handles whose demanded cone forms the frame plan.
	 * @return Nothing on success, or a `vk::Result` error if any GPU step failed.
	 */
	[[nodiscard]] std::expected<void, vk::Result> run_once(graph::Graph&					  graph,
														   std::span<const graph::DataHandle> sinks);

	 private:
	Context*		  m_context;
	ResourcePool*	  m_pool;
	CommandManager*	  m_commands;
	graph::Scheduler* m_scheduler;
};
} // namespace veng

#endif // VENG_HEADLESSEXECUTOR_HPP
