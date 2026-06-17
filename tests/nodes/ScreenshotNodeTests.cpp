//
// L4 test ([[pass-draw-redesign]]): a sink that needs *completed* GPU work runs through
// `on_retired`, the post-fence hook on `GpuNode`. A solid fullscreen pass renders red; a
// `ScreenshotNode` peers as a sink and copies the rendered image into a host-visible staging
// buffer; `HeadlessExecutor::run_once` records + submits + waits the fence + fires the sink hooks —
// writing the file. This proves the multi-sink mechanism + post-fence readback end-to-end with no
// swapchain (and exercises HeadlessExecutor), for both the PPM and PNG encoders.
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>
#include <veng/context/Context.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/managers/CommandManager.hpp>
#include <veng/managers/HeadlessExecutor.hpp>
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

// Render a solid red fullscreen pass and screenshot it to `path` in `fmt`, driven headlessly by
// HeadlessExecutor — the same harness an offscreen capture uses. Returns the node's capture_count
// (1 on a successful write).
std::size_t capture_red(veng::Context& ctx, const std::string& path, veng::nodes::ImageFormat fmt)
{
	Graph			 graph;
	auto			 screen		 = graph.add_source<veng::rhi::Extent2D>(veng::rhi::Extent2D{SIDE, SIDE});
	auto			 color		 = graph.add_source<glm::vec4>(glm::vec4{1.0F, 0.0F, 0.0F, 1.0F});
	const DataHandle scene_image = graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));
	const DataHandle shot_done	 = graph.add(std::make_unique<ValueData<int>>(0));

	auto solid = std::make_unique<veng::nodes::GraphicsNode>("demo/fullscreen.vert", "tests/slice/solid.frag", COLOR,
															 veng::rhi::Format::UNDEFINED, 3, screen, scene_image);
	solid->push_constant<glm::vec4>(color, veng::rhi::ShaderStage::FRAGMENT);
	graph.set_producer(scene_image, graph.add(std::move(solid)));

	auto  shot	   = std::make_unique<veng::nodes::ScreenshotNode>(scene_image, shot_done, path, fmt);
	auto* shot_ptr = shot.get();
	graph.set_producer(shot_done, graph.add(std::move(shot)));

	veng::ResourcePool	   pool(ctx.device(), ctx.rhi(), ctx.allocator(), 1);
	veng::CommandManager   commands(ctx);
	InlineScheduler		   scheduler;
	veng::HeadlessExecutor executor(ctx, pool, commands, scheduler);
	if (!executor.run_once(graph, std::array{shot_done}).has_value())
	{
		return 0;
	}
	return shot_ptr->capture_count();
}
} // namespace

TEST_CASE("ScreenshotNode writes a PPM via HeadlessExecutor (peer sink, no swapchain)", "[nodes][screenshot][slice]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			  ctx  = make_context();
	const std::string path = "/tmp/veng_screenshot_test.ppm";
	(void)std::remove(path.c_str());

	REQUIRE(capture_red(ctx, path, veng::nodes::ImageFormat::Ppm) == 1);

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

	(void)std::remove(path.c_str());
}

TEST_CASE("ScreenshotNode writes a PNG when asked", "[nodes][screenshot][slice]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			  ctx  = make_context();
	const std::string path = "/tmp/veng_screenshot_test.png";
	(void)std::remove(path.c_str());

	REQUIRE(capture_red(ctx, path, veng::nodes::ImageFormat::Png) == 1);

	// Verify the 8-byte PNG signature — proves stb_image_write produced a real PNG, no decoder needed.
	std::ifstream file(path, std::ios::binary);
	REQUIRE(file.is_open());
	std::array<unsigned char, 8> sig{};
	file.read(reinterpret_cast<char*>(sig.data()), static_cast<std::streamsize>(sig.size()));
	const std::array<unsigned char, 8> png_magic{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
	REQUIRE(sig == png_magic);

	(void)std::remove(path.c_str());
}
