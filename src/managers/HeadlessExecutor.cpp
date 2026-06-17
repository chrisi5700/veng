/**
 * @file
 * @author chris
 * @brief Implementation of @ref veng::HeadlessExecutor.
 * @ingroup managers
 */

#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/Sink.hpp>
#include <veng/gpu/SubmitContext.hpp>
#include <veng/managers/HeadlessExecutor.hpp>
#include <veng/managers/QueueKind.hpp>
#include <veng/rhi/CommandEncoder.hpp>

namespace veng
{
HeadlessExecutor::HeadlessExecutor(Context& context, ResourcePool& pool, CommandManager& commands,
								   graph::Scheduler& scheduler) noexcept
	: m_context(&context)
	, m_pool(&pool)
	, m_commands(&commands)
	, m_scheduler(&scheduler)
{
}

std::expected<void, vk::Result> HeadlessExecutor::run_once(graph::Graph&					  graph,
														   std::span<const graph::DataHandle> sinks)
{
	m_pool->begin_frame(0);

	auto cmd = m_commands->begin(QueueKind::Graphics, 0);
	if (!cmd.has_value())
	{
		return std::unexpected(cmd.error());
	}

	gpu::GpuExecContext gpu_ctx(graph, *m_context, *m_pool, rhi::CommandEncoder(cmd.value(), m_context->rhi()), 0);
	auto				plan = graph.resolve(sinks);
	if (!plan.has_value() || !graph.execute(plan.value(), *m_scheduler, gpu_ctx) ||
		cmd.value().end() != vk::Result::eSuccess)
	{
		return std::unexpected(vk::Result::eErrorUnknown);
	}

	const auto fence = m_context->device().createFence({});
	if (fence.result != vk::Result::eSuccess)
	{
		return std::unexpected(fence.result);
	}

	// Guard the submit before waiting: a failed submit leaves the fence unsignalled, so an
	// unconditional waitForFences(UINT64_MAX) would deadlock.
	const vk::Result submitted =
		m_context->graphics_queue().submit(vk::SubmitInfo().setCommandBuffers(cmd.value()), fence.value);
	if (submitted != vk::Result::eSuccess)
	{
		m_context->device().destroyFence(fence.value);
		return std::unexpected(submitted);
	}

	// Sinks (e.g. ScreenshotNode) opt into the post-submit / post-retire hooks; the cross-cast is
	// nullptr for every non-sink node. No PresentFrame: this path never presents.
	gpu::SubmitContext post(graph, *m_context, 0);
	for (const graph::NodeHandle handle : plan.value().nodes())
	{
		if (auto* sink = dynamic_cast<gpu::Sink*>(graph.get_node(handle)); sink != nullptr)
		{
			sink->on_submitted(post);
		}
	}

	(void)m_context->device().waitForFences(fence.value, vk::True, UINT64_MAX);
	for (const graph::NodeHandle handle : plan.value().nodes())
	{
		if (auto* sink = dynamic_cast<gpu::Sink*>(graph.get_node(handle)); sink != nullptr)
		{
			sink->on_retired(post);
		}
	}
	m_context->device().destroyFence(fence.value);

	return {};
}
} // namespace veng
