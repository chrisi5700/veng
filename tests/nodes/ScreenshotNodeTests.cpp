//
// L4 test ([[pass-draw-redesign]]): a sink that needs *completed* GPU work runs through
// `on_retired`, the post-fence hook on `GpuNode`. A solid fullscreen pass renders red; a
// `ScreenshotNode` peers as a sink and records a copy of the rendered image into a host-visible
// staging buffer; the test ends + submits + waits the fence + invokes `on_retired` (the
// driver/executor's job in the playground) — which writes a PPM file. The test reads the file
// back and verifies the center pixel is red, proving the multi-sink mechanism + post-fence
// readback work end-to-end without `PresentNode` (this graph is headless, no swapchain).
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <veng/context/Context.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/gpu/SubmitContext.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/managers/CommandManager.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/nodes/ScreenshotNode.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/ResourcePool.hpp>

using namespace veng::graph;

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("Screenshot Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

constexpr veng::rhi::Format COLOR = veng::rhi::Format::RGBA8_UNORM;
constexpr std::uint32_t		SIDE  = 32;
} // namespace

TEST_CASE("ScreenshotNode captures a rendered image via on_retired (peer sink, no swapchain)",
		  "[nodes][screenshot][slice]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			 ctx	= make_context();
	const vk::Device device = ctx.device();

	// Graph: a solid red fullscreen pass leaves an image in TRANSFER_SRC; a ScreenshotNode sink
	// reads it and writes a PPM file on retire. No PresentNode — this graph is headless.
	Graph			 graph;
	auto			 screen		 = graph.add_source<veng::rhi::Extent2D>(veng::rhi::Extent2D{SIDE, SIDE});
	auto			 color		 = graph.add_source<glm::vec4>(glm::vec4{1.0F, 0.0F, 0.0F, 1.0F});
	const DataHandle scene_image = graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));
	const DataHandle shot_done	 = graph.add(std::make_unique<ValueData<int>>(0));

	auto solid = std::make_unique<veng::nodes::GraphicsNode>("demo/fullscreen.vert", "tests/slice/solid.frag", COLOR,
															 veng::rhi::Format::UNDEFINED, 3, screen, scene_image);
	solid->push_constant<glm::vec4>(color, veng::rhi::ShaderStage::FRAGMENT);
	graph.set_producer(scene_image, graph.add(std::move(solid)));

	const std::string path = "/tmp/veng_screenshot_test.ppm";
	(void)std::remove(path.c_str()); // start clean
	auto  shot	   = std::make_unique<veng::nodes::ScreenshotNode>(scene_image, shot_done, path);
	auto* shot_ptr = shot.get();
	graph.set_producer(shot_done, graph.add(std::move(shot)));

	veng::ResourcePool res_pool(ctx.device(), ctx.rhi(), ctx.allocator(), 1);
	res_pool.begin_frame(0);
	veng::CommandManager commands(ctx);
	InlineScheduler		 scheduler;

	auto cmd = commands.begin(veng::QueueKind::Graphics, 0);
	REQUIRE(cmd.has_value());
	veng::gpu::GpuExecContext gpu_ctx(graph, ctx, res_pool, veng::rhi::CommandEncoder(*cmd, ctx.rhi()), 0);
	auto					  plan = graph.resolve(std::array{shot_done});
	REQUIRE(plan.has_value());
	REQUIRE(graph.execute(*plan, scheduler, gpu_ctx));
	REQUIRE(cmd->end() == vk::Result::eSuccess);

	// Submit (no swapchain semaphores — purely a fence-gated submission).
	const auto fence = device.createFence({});
	REQUIRE(fence.result == vk::Result::eSuccess);
	REQUIRE(ctx.graphics_queue().submit(vk::SubmitInfo().setCommandBuffers(*cmd), fence.value) == vk::Result::eSuccess);

	// Dispatch on_submitted (no-op for these sinks — present isn't in the plan).
	veng::gpu::SubmitContext post_ctx(graph, ctx, 0);
	for (const NodeHandle h : plan->nodes())
	{
		if (auto* sink = dynamic_cast<veng::gpu::Sink*>(graph.get_node(h)))
		{
			sink->on_submitted(post_ctx);
		}
	}
	REQUIRE(device.waitForFences(fence.value, vk::True, UINT64_MAX) == vk::Result::eSuccess);

	// on_retired: the screenshot reads the staging buffer + writes the PPM file. This is what
	// the driver does at the next acquire(slot) when the slot's fence signals.
	for (const NodeHandle h : plan->nodes())
	{
		if (auto* sink = dynamic_cast<veng::gpu::Sink*>(graph.get_node(h)))
		{
			sink->on_retired(post_ctx);
		}
	}
	REQUIRE(shot_ptr->capture_count() == 1);

	// Verify the file: PPM (P6) header, then RGB bytes. Center pixel should be red.
	std::ifstream file(path, std::ios::binary);
	REQUIRE(file.is_open());
	std::string	  magic;
	std::uint32_t width = 0, height = 0, maxv = 0;
	file >> magic >> width >> height >> maxv;
	REQUIRE(magic == "P6");
	REQUIRE(width == SIDE);
	REQUIRE(height == SIDE);
	REQUIRE(maxv == 255);
	file.get(); // consume the single whitespace separating header from binary pixels
	std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * 3);
	file.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
	const std::size_t center = (static_cast<std::size_t>(SIDE / 2) * SIDE + (SIDE / 2)) * 3;
	REQUIRE(pixels[center + 0] > 200); // red
	REQUIRE(pixels[center + 1] < 50);
	REQUIRE(pixels[center + 2] < 50);

	device.destroyFence(fence.value);
	(void)std::remove(path.c_str());
}
