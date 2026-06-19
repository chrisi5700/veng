/**
 * @file
 * @author chris
 * @brief Implementation of @ref veng::FrameExecutor.
 * @ingroup managers
 */

#include <utility>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/SubmitContext.hpp>
#include <veng/managers/FrameExecutor.hpp>
#include <veng/managers/QueueKind.hpp>
#include <veng/rhi/Convert.hpp>

namespace veng
{
FrameExecutor::FrameExecutor(Context& context, SwapchainManager& swap, ResourcePool& pool, CommandManager& commands,
							 graph::Scheduler& scheduler, graph::TypedHandle<gpu::ImageRef> swapchain_handle,
							 std::size_t frames_in_flight) noexcept
	: m_context(&context)
	, m_swap(&swap)
	, m_pool(&pool)
	, m_commands(&commands)
	, m_scheduler(&scheduler)
	, m_swapchain_handle(swapchain_handle)
	, m_frames_in_flight(frames_in_flight == 0 ? 1 : frames_in_flight)
	, m_pending_retire(m_frames_in_flight)
{
}

FrameExecutor::Frame FrameExecutor::run_frame(graph::Graph& graph, std::span<const graph::DataHandle> sinks,
											  Pacing pacing, std::mutex* graph_mutex)
{
	const std::size_t slot = m_frame_index % m_frames_in_flight;

	// OnDemand: resolve BEFORE acquire so an unchanged graph idles the thread (no GPU work,
	// no semaphore wait, no submit). The swapchain source is set_now'd later as a value-only
	// hand-off so it doesn't itself drive the plan.
	graph::FramePlan plan;
	if (pacing == Pacing::OnDemand)
	{
		// Only bind the mutex when one was supplied: `*graph_mutex` would dereference null otherwise
		// (UB even with defer_lock — the reference binds before the null-check could skip it).
		std::unique_lock<std::mutex> lock;
		if (graph_mutex != nullptr)
		{
			lock = std::unique_lock<std::mutex>(*graph_mutex);
		}
		auto resolved = graph.resolve(sinks);
		if (!resolved.has_value())
		{
			return Frame{.status = Status::NodeFailed, .plan = {}};
		}
		if (resolved->empty())
		{
			return Frame{.status = Status::Idled, .plan = {}};
		}
		plan = std::move(resolved.value());
	}

	m_pool->begin_frame(m_frame_index);

	auto acquired = m_swap->acquire(slot);
	if (!acquired.has_value())
	{
		return Frame{.status = Status::AcquireFailed, .plan = {}};
	}
	if (!acquired.value().has_value())
	{
		return Frame{.status = Status::OutOfDate, .plan = {}};
	}
	const SwapchainManager::Frame frame = acquired.value().value();

	// The slot's in-flight fence has just signalled (acquire waited it), so the frame that
	// previously occupied this slot has retired — fire its on_retired hooks before recycling.
	gpu::SubmitContext retire_ctx(graph, *m_context, slot);
	for (gpu::Sink* sink : m_pending_retire[slot])
	{
		sink->on_retired(retire_ctx);
	}
	m_pending_retire[slot].clear();

	m_commands->reset_frame(slot);

	// Stamp the monotonic frame index into the ref's version so consecutive sets compare
	// unequal even when the swapchain hands us the same vk::Image handle. Without this,
	// Continuous mode's `graph.set(swapchain_handle, ref)` is a no-op whenever the swap
	// recycles an image, BlitNode + PresentNode drop out of the plan, no present is issued,
	// and we leak acquired images (vkAcquireNextImageKHR eventually deadlocks at UINT64_MAX
	// because every swap image is already acquired).
	const gpu::ImageRef ref{.texture = m_swap->texture_handle(frame.image_index),
							.extent	 = m_swap->extent(),
							.format	 = m_swap->format(),
							.version = m_frame_index};

	if (pacing == Pacing::OnDemand)
	{
		// Value only: the plan was decided pre-acquire from real data changes; hand the blit the
		// freshly-acquired image without marking it as a dirty pulse.
		graph.set_now(m_swapchain_handle, ref);
	}
	else
	{
		std::unique_lock<std::mutex> lock;
		if (graph_mutex != nullptr)
		{
			lock = std::unique_lock<std::mutex>(*graph_mutex);
		}
		graph.set(m_swapchain_handle, ref);
		auto resolved = graph.resolve(sinks);
		if (!resolved.has_value())
		{
			return Frame{.status = Status::NodeFailed, .plan = {}};
		}
		plan = std::move(resolved.value());
	}

	auto cmd = m_commands->begin(QueueKind::Graphics, slot);
	if (!cmd.has_value())
	{
		return Frame{.status = Status::AcquireFailed, .plan = std::move(plan)};
	}

	gpu::GpuExecContext gpu_ctx(graph, *m_context, *m_pool, rhi::CommandEncoder(cmd.value(), m_context->rhi()), slot);
	if (!graph.execute(plan, *m_scheduler, gpu_ctx))
	{
		return Frame{.status = Status::NodeFailed, .plan = std::move(plan)};
	}

	(void)cmd.value().end();

	// Submit sync comes straight from the acquired Frame — no semaphore plumbing on edges.
	const vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eTransfer;
	(void)m_context->graphics_queue().submit(vk::SubmitInfo()
												 .setWaitSemaphores(frame.image_available)
												 .setWaitDstStageMask(wait_stage)
												 .setCommandBuffers(cmd.value())
												 .setSignalSemaphores(frame.render_finished),
											 frame.in_flight);

	// Dispatch on_submitted with the present info attached. Only nodes that opted into the
	// gpu::Sink interface (PresentNode, screenshot, picking readback, future video) participate;
	// the cross-cast resolves to nullptr for every non-sink node in the plan.
	gpu::SubmitContext sub_ctx(graph, *m_context, slot,
							   gpu::PresentFrame{.image_index	 = frame.image_index,
												 .present_signal = m_swap->render_finished_handle(frame.image_index)});
	for (const graph::NodeHandle handle : plan.nodes())
	{
		if (auto* sink = dynamic_cast<gpu::Sink*>(graph.get_node(handle)); sink != nullptr)
		{
			sink->on_submitted(sub_ctx);
			m_pending_retire[slot].push_back(sink);
		}
	}

	++m_frame_index;
	return Frame{.status = Status::Rendered, .plan = std::move(plan)};
}
} // namespace veng
