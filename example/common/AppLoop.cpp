//
// See AppLoop.hpp.
//

#include "AppLoop.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <GLFW/glfw3.h>
#include <print>
#include <thread>
#include <utility>
#include <veng/logging/Logger.hpp>
#include <veng/nodes/BlitNode.hpp>
#include <veng/rendergraph/data/Data.hpp>

namespace example
{
namespace
{
bool plan_contains_present(const veng::graph::FramePlan& plan, veng::graph::NodeHandle present)
{
	for (const veng::graph::NodeHandle handle : plan.nodes())
	{
		if (handle == present)
		{
			return true;
		}
	}
	return false;
}
} // namespace

AppLoop::AppLoop(const AppConfig& config)
	: m_config(config)
	, m_window(config.title, config.width, config.height)
	, m_depth_format(config.depth_format)
{
	veng::Logger::instance().set_level(spdlog::level::warn);

	auto ctx_result =
		veng::Context::create("veng reactive renderer", m_window.required_extensions(),
							  [this](VkInstance instance) { return m_window.create_surface(instance); });
	if (!ctx_result.has_value())
	{
		throw std::runtime_error("AppLoop: failed to create Vulkan context");
	}
	m_ctx = std::make_unique<veng::Context>(std::move(ctx_result.value()));

	auto swap_result = veng::SwapchainManager::create(*m_ctx, m_window.framebuffer_extent(), m_config.frames_in_flight);
	if (!swap_result.has_value())
	{
		throw std::runtime_error("AppLoop: failed to create swapchain");
	}
	m_swap = std::make_unique<veng::SwapchainManager>(std::move(swap_result.value()));

	m_pool	   = std::make_unique<veng::ResourcePool>(m_ctx->device(), m_ctx->allocator(), m_config.frames_in_flight);
	m_commands = std::make_unique<veng::CommandManager>(*m_ctx);

	// --- Sources + frame closer (BlitNode + PresentNode) -----------------------------
	using namespace veng::graph;

	m_screen		  = m_graph.add_source<vk::Extent2D>(m_swap->extent());
	m_swapchain_image = m_graph.add_source<veng::gpu::ImageRef>(veng::gpu::ImageRef{});
	m_scene_image	  = m_graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));
	m_presented_image = m_graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));
	m_frame_done	  = m_graph.add(std::make_unique<ValueData<int>>(0));

	auto blit = std::make_unique<veng::nodes::BlitNode>(m_scene_image, m_swapchain_image, m_presented_image,
														vk::ImageLayout::ePresentSrcKHR);
	m_graph.set_producer(m_presented_image, m_graph.add(std::move(blit)));

	auto present  = std::make_unique<veng::nodes::PresentNode>(*m_swap, m_presented_image, m_frame_done);
	m_present_ptr = present.get();
	m_graph.set_producer(m_frame_done, m_graph.add(std::move(present)));

	m_executor =
		std::make_unique<veng::FrameExecutor>(*m_ctx, *m_swap, *m_pool, *m_commands, m_scheduler, m_swapchain_image,
											  m_config.frames_in_flight);

	m_camera = std::make_unique<OrbitCamera>(m_graph, m_swap->extent(), m_config.camera_target,
											 m_config.camera_distance, m_config.camera_yaw, m_config.camera_pitch);
}

AppLoop::~AppLoop()
{
	if (m_ctx)
	{
		(void)m_ctx->device().waitIdle();
	}
}

void AppLoop::rebuild_swapchain(vk::Extent2D extent)
{
	(void)m_ctx->device().waitIdle();
	if (!m_swap->rebuild(extent).has_value())
	{
		return;
	}
	const std::scoped_lock lock(m_graph_mutex);
	m_graph.set(m_screen, m_swap->extent());
	m_camera->publish(m_swap->extent());
}

void AppLoop::run(std::function<void(const veng::graph::FramePlan&)> on_frame)
{
	using namespace std::chrono;

	veng::FrameExecutor::Pacing pacing		   = m_config.pacing;
	bool						space_was_down = false;
	std::uint64_t				presents	   = 0;
	std::uint64_t				renders		   = 0;
	auto						stats_at	   = steady_clock::now();

	while (!m_window.should_close())
	{
		m_window.poll();

		const bool space = m_window.key_down(GLFW_KEY_SPACE);
		if (space && !space_was_down)
		{
			pacing = (pacing == veng::FrameExecutor::Pacing::OnDemand) ? veng::FrameExecutor::Pacing::Continuous
																	   : veng::FrameExecutor::Pacing::OnDemand;
			std::println("\n>> pacing: {}", pacing == veng::FrameExecutor::Pacing::Continuous
												? "Continuous (render every frame)"
												: "OnDemand (render only on data change)");
			std::fflush(stdout);
		}
		space_was_down = space;

		{
			const std::scoped_lock lock(m_graph_mutex);
			if (m_camera->tick(m_window, m_swap->extent()))
			{
				m_wake.notify_one();
			}
		}

		vk::Extent2D fb = m_window.framebuffer_extent();
		if (fb.width == 0 || fb.height == 0)
		{
			std::this_thread::sleep_for(milliseconds(10));
			continue;
		}
		if (fb.width != m_swap->extent().width || fb.height != m_swap->extent().height)
		{
			rebuild_swapchain(fb);
		}

		auto outcome = m_executor->run_frame(m_graph, std::array{m_frame_done}, pacing, &m_graph_mutex);

		if (outcome.status == veng::FrameExecutor::Status::Idled)
		{
			std::unique_lock lock(m_graph_mutex);
			m_wake.wait_for(lock, milliseconds(16));
			continue;
		}
		if (outcome.status == veng::FrameExecutor::Status::OutOfDate)
		{
			rebuild_swapchain(m_window.framebuffer_extent());
			continue;
		}
		if (outcome.status == veng::FrameExecutor::Status::NodeFailed)
		{
			std::println(">> frame {} dropped (node failed)", m_executor->frame_index());
			continue;
		}
		if (outcome.status == veng::FrameExecutor::Status::AcquireFailed)
		{
			std::println("acquire/begin failed");
			break;
		}

		if (m_present_ptr->out_of_date())
		{
			rebuild_swapchain(m_window.framebuffer_extent());
		}

		++presents;
		// "Rendered" = the present sink was driven; a "real" scene re-render only counts if a
		// non-trivial plan ran. The plan always contains the present (it's the sink); use
		// `nodes().size() > 1` as a cheap proxy for "more than just the present sink ran".
		if (outcome.plan.nodes().size() > 1)
		{
			++renders;
		}

		if (on_frame)
		{
			on_frame(outcome.plan);
		}

		const auto now = steady_clock::now();
		if (now - stats_at >= seconds(1))
		{
			const double seconds_elapsed = duration<double>(now - stats_at).count();
			std::println("[{}] present {:7.0f} fps | re-render {:6.0f} fps | cached {:5.1f}%",
						 pacing == veng::FrameExecutor::Pacing::Continuous ? "continuous" : "on-demand ",
						 static_cast<double>(presents) / seconds_elapsed,
						 static_cast<double>(renders) / seconds_elapsed,
						 presents > 0
							 ? 100.0 * static_cast<double>(presents - renders) / static_cast<double>(presents)
							 : 0.0);
			std::fflush(stdout);
			presents = 0;
			renders	 = 0;
			stats_at = now;
		}
	}
}
} // namespace example
