//
// veng reactive renderer demo — a spinning cube on a real swapchain (design.md §1, §L4,
// §L5). Two threads exercise the demand-driven caching thesis on live GPU work:
//
//   * a writer thread spins the cube's rotation angle at a steady 60 Hz (a fresh angle
//     every tick), so the cube re-renders 60 times a second;
//   * the render thread (this, the main thread — GLFW requires it) is paced one of two
//     ways, toggled live with SPACE:
//       - Continuous: render every loop iteration (the swapchain image is fed as a
//         per-frame dirty pulse) — uncapped; present fps far exceeds the 60 Hz cube
//         re-render rate, the rest being cached re-blits of the last scene. Max power.
//       - OnDemand: render only when a data source actually changed — an empty FramePlan
//         means there is nothing to do, so the thread idles (no acquire/submit/present).
//         Here present fps tracks the 60 Hz data rate exactly: same picture, no waste.
//
// The graph tail is declarative: the cube renders into a scene ImageData; a BlitNode
// copies it into the acquired swapchain image; a PresentNode closes the frame (submit +
// present). The pacing flag only decides whether the swapchain image is fed as a dirty
// pulse (Continuous) or merely as an execute-time value via set_now (OnDemand).
//

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <GLFW/glfw3.h> // for the GLFW_KEY_* constant used by the live pacing toggle
#include <memory>
#include <mutex>
#include <print>
#include <thread>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/managers/CommandManager.hpp>
#include <veng/managers/QueueKind.hpp>
#include <veng/managers/SwapchainManager.hpp>
#include <veng/nodes/BlitNode.hpp>
#include <veng/nodes/PresentNode.hpp>
#include <veng/nodes/RasterCubeNode.hpp>
#include <veng/pipelines/GraphicsPipeline.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/shader/Shader.hpp>

#include "Window.hpp"

using namespace veng;
using namespace veng::graph;

namespace
{
// How the render thread is paced — the "electricity" knob (toggle live with SPACE):
//   Continuous: render every loop iteration (the swapchain image is fed as a per-frame
//               dirty pulse) — uncapped, maximum throughput, maximum power.
//   OnDemand:   render only when a CPU-side data source actually changed — a static scene
//               costs nothing (empty FramePlan -> the thread idles, no acquire/submit).
enum class Pacing : std::uint8_t
{
	Continuous,
	OnDemand
};

bool plan_contains(const FramePlan& plan, NodeHandle node)
{
	for (const NodeHandle handle : plan.nodes())
	{
		if (handle == node)
		{
			return true;
		}
	}
	return false;
}
} // namespace

int main()
{
	Logger::instance().set_level(spdlog::level::warn);

	demo::Window window("veng — reactive spinning cube", 1280, 720);

	auto ctx_result = Context::create("veng reactive renderer", window.required_extensions(),
									  [&window](VkInstance instance) { return window.create_surface(instance); });
	if (!ctx_result.has_value())
	{
		std::println("Failed to create Vulkan context");
		return 1;
	}
	Context ctx = std::move(ctx_result.value());

	auto swap_result = SwapchainManager::create(ctx, window.framebuffer_extent(), 1);
	if (!swap_result.has_value())
	{
		std::println("Failed to create swapchain: {}", vk::to_string(swap_result.error()));
		return 1;
	}
	SwapchainManager swap = std::move(swap_result.value());

	// The cube renders into a scene target of the swapchain's format, so the present blit
	// is a straight copy; the depth target is D32.
	const vk::Format	 scene_color  = swap.format();
	constexpr vk::Format depth_format = vk::Format::eD32Sfloat;

	auto vertex	  = Shader::create_shader(ctx.device(), "demo/cube.vert");
	auto fragment = Shader::create_shader(ctx.device(), "demo/cube.frag");
	if (!vertex.has_value() || !fragment.has_value())
	{
		std::println("Failed to load cube shaders");
		return 1;
	}
	const std::array color_formats{scene_color};
	auto			 pipeline =
		GraphicsPipelineBuilder(vertex.value(), fragment.value())
			.color_formats(color_formats)
			.depth_format(depth_format)
			.rasterization(vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise)
			.build(ctx);
	if (!pipeline.has_value())
	{
		std::println("Failed to build cube pipeline: {}", to_string(pipeline.error()));
		return 1;
	}

	const auto fence_result =
		ctx.device().createFence(vk::FenceCreateInfo().setFlags(vk::FenceCreateFlagBits::eSignaled));
	if (fence_result.result != vk::Result::eSuccess)
	{
		std::println("Failed to create fence");
		return 1;
	}
	const vk::Fence fence = fence_result.value;

	// --- The reactive graph, built once -------------------------------------------------
	//   screen + angle -> cube -> scene_image
	//   scene_image + swapchain_image (dirty each frame) -> blit -> presented_image
	//   presented_image -> present (submit + present) -> frame_done  (the demanded sink)
	Graph graph;
	auto  screen		  = graph.add_source<vk::Extent2D>(swap.extent());
	auto  angle			  = graph.add_source<float>(0.0F);
	auto  swapchain_image = graph.add_source<gpu::ImageRef>(gpu::ImageRef{});

	const DataHandle scene_image	 = graph.add(std::make_unique<ValueData<gpu::ImageRef>>(gpu::ImageRef{}));
	const DataHandle presented_image = graph.add(std::make_unique<ValueData<gpu::ImageRef>>(gpu::ImageRef{}));
	const DataHandle frame_done		 = graph.add(std::make_unique<ValueData<int>>(0));

	auto cube = std::make_unique<nodes::RasterCubeNode>(std::move(pipeline.value()), scene_color, depth_format, screen,
														angle, scene_image);
	const NodeHandle cube_node = graph.add(std::move(cube));
	graph.set_producer(scene_image, cube_node);

	auto blit = std::make_unique<nodes::BlitNode>(scene_image, swapchain_image, presented_image,
												  vk::ImageLayout::ePresentSrcKHR);
	graph.set_producer(presented_image, graph.add(std::move(blit)));

	auto  present	  = std::make_unique<nodes::PresentNode>(swap, fence, presented_image, frame_done);
	auto* present_ptr = present.get();
	graph.set_producer(frame_done, graph.add(std::move(present)));

	CommandManager	commands(ctx);
	InlineScheduler scheduler;

	std::mutex				graph_mutex;
	std::condition_variable wake; // OnDemand: the writer pokes this when it mutates a source
	std::atomic<bool>		running{true};
	Pacing					pacing = Pacing::OnDemand; // start power-frugal; SPACE toggles

	// Writer thread: a continuous 60 Hz spin — a fresh angle every tick. set() queues under
	// the lock; the render thread applies it in resolve() (also under the lock) — a clean
	// frame-boundary snapshot, with the expensive execute running outside the lock. The
	// notify wakes the render thread in OnDemand mode (it sleeps when nothing has changed).
	std::thread writer(
		[&]
		{
			using namespace std::chrono;
			const auto period	   = microseconds(16'667); // ~60 Hz
			auto	   next		   = steady_clock::now();
			float	   angle_value = 0.0F;
			while (running.load(std::memory_order_relaxed))
			{
				angle_value += 0.04F;
				{
					const std::scoped_lock lock(graph_mutex);
					graph.set(angle, angle_value);
				}
				wake.notify_one();
				next += period;
				std::this_thread::sleep_until(next);
			}
		});

	const vk::Device device		  = ctx.device();
	std::uint64_t	 presents	  = 0;
	std::uint64_t	 cube_renders = 0;
	auto			 stats_at	  = std::chrono::steady_clock::now();

	const auto rebuild_for_resize = [&](vk::Extent2D size)
	{
		(void)device.waitIdle();
		if (!swap.rebuild(size).has_value())
		{
			return false;
		}
		const std::scoped_lock lock(graph_mutex);
		graph.set(screen, swap.extent());
		return true;
	};

	bool space_was_down = false;
	while (!window.should_close())
	{
		window.poll();

		// SPACE toggles pacing live, so the fps/power gap is visible on a single run.
		const bool space = window.key_down(GLFW_KEY_SPACE);
		if (space && !space_was_down)
		{
			pacing = (pacing == Pacing::OnDemand) ? Pacing::Continuous : Pacing::OnDemand;
			std::println("\n>> pacing: {}", pacing == Pacing::Continuous ? "Continuous (render every frame)"
																		 : "OnDemand (render only on data change)");
			std::fflush(stdout);
		}
		space_was_down = space;

		vk::Extent2D fb = window.framebuffer_extent();
		if (fb.width == 0 || fb.height == 0)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(10)); // minimized
			continue;
		}

		// A resize dirties the screen source (a real data change), driving a render in either mode.
		if (fb.width != swap.extent().width || fb.height != swap.extent().height)
		{
			if (!rebuild_for_resize(fb))
			{
				break;
			}
		}

		// OnDemand: decide before spending anything. Resolve the demanded plan WITHOUT
		// touching the swapchain image — if nothing upstream changed it is empty, so we idle
		// (no acquire / submit / present) until the writer pokes us. This is the whole point:
		// a static scene costs a sleeping thread and nothing else.
		FramePlan plan;
		if (pacing == Pacing::OnDemand)
		{
			std::unique_lock lock(graph_mutex);
			auto			 resolved = graph.resolve(std::array{frame_done});
			if (!resolved.has_value())
			{
				std::println("resolve failed");
				break;
			}
			if (resolved->empty())
			{
				wake.wait_for(lock, std::chrono::milliseconds(16)); // sleep; timeout keeps events responsive
				continue;
			}
			plan = std::move(resolved.value());
		}

		// Retire the previous frame, recycle its command pool, acquire the next image.
		(void)device.waitForFences(fence, vk::True, UINT64_MAX);
		(void)device.resetFences(fence);
		commands.reset_frame(0);

		auto acquired = swap.acquire(0);
		if (!acquired.has_value())
		{
			std::println("acquire failed: {}", vk::to_string(acquired.error()));
			break;
		}
		if (!acquired.value().has_value())
		{
			if (!rebuild_for_resize(window.framebuffer_extent()))
			{
				break;
			}
			continue;
		}
		const SwapchainManager::Frame frame = acquired.value().value();
		const gpu::ImageRef			  ref{.image		  = swap.image(frame.image_index),
										  .extent		  = swap.extent(),
										  .format		  = swap.format(),
										  .index		  = frame.image_index,
										  .acquire_wait	  = frame.image_available,
										  .present_signal = frame.render_finished};

		if (pacing == Pacing::OnDemand)
		{
			// Value only: the plan above already committed to render off a real data change;
			// just hand the blit the freshly-acquired image. set_now never marks it dirty.
			graph.set_now(swapchain_image, ref);
		}
		else
		{
			// Continuous: feed the image as a per-frame dirty pulse so blit + present always
			// run, then resolve.
			const std::scoped_lock lock(graph_mutex);
			graph.set(swapchain_image, ref);
			auto resolved = graph.resolve(std::array{frame_done});
			if (!resolved.has_value())
			{
				std::println("resolve failed");
				break;
			}
			plan = std::move(resolved.value());
		}

		auto cmd = commands.begin(QueueKind::Graphics, 0);
		if (!cmd.has_value())
		{
			std::println("command buffer begin failed");
			break;
		}

		// Execute records the cube + blit and, as the sink, the PresentNode ends the
		// command buffer, submits it, and presents — all GPU/queue work lives in the graph.
		gpu::GpuExecContext gpu(graph, ctx, cmd.value(), 0);
		graph.execute(plan, scheduler, gpu);

		if (present_ptr->out_of_date() && !rebuild_for_resize(window.framebuffer_extent()))
		{
			break;
		}

		++presents;
		if (plan_contains(plan, cube_node))
		{
			++cube_renders;
		}

		const auto now = std::chrono::steady_clock::now();
		if (now - stats_at >= std::chrono::seconds(1))
		{
			const double seconds = std::chrono::duration<double>(now - stats_at).count();
			std::println("[{}] present {:7.0f} fps | cube re-render {:6.0f} fps | cached {:5.1f}%",
						 pacing == Pacing::Continuous ? "continuous" : "on-demand ",
						 static_cast<double>(presents) / seconds, static_cast<double>(cube_renders) / seconds,
						 presents > 0
							 ? 100.0 * static_cast<double>(presents - cube_renders) / static_cast<double>(presents)
							 : 0.0);
			std::fflush(stdout); // a live monitor: flush each tick rather than at exit
			presents	 = 0;
			cube_renders = 0;
			stats_at	 = now;
		}
	}

	running.store(false);
	writer.join();
	(void)device.waitIdle();
	device.destroyFence(fence);
	return 0;
}
