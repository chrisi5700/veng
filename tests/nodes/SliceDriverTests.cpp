//
// L4/L5 slice proof (design.md §1, §L4, §L5): the reactive rendering thesis, offscreen.
// A ScreenSize source -> RasterTriangleNode -> scene token; a PresentNode depends on
// BOTH the scene token AND a swapchain *source* dirtied every frame. A minimal driver
// loop bumps the swapchain source and runs frame(present_sink) each iteration.
//
// The assertion is the whole point: with a static scene, the present/blit runs every
// frame but the raster node runs 0 extra times (cached); a ScreenSize change re-runs
// the raster node. The wiring rule (scene must not depend on the swapchain source) is
// what makes this hold.
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/managers/CommandManager.hpp>
#include <veng/nodes/PresentNode.hpp>
#include <veng/nodes/RasterTriangleNode.hpp>
#include <veng/pipelines/GraphicsPipeline.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/resources/Image.hpp>
#include <veng/shader/Shader.hpp>

using namespace veng::graph;

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("Slice Driver Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

veng::Shader load(const veng::Context& ctx, std::string_view name)
{
	auto result = veng::Shader::create_shader(ctx.device(), name);
	REQUIRE(result.has_value());
	return std::move(result.value());
}

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

constexpr vk::Format	COLOR = vk::Format::eR8G8B8A8Unorm;
constexpr std::uint32_t SIDE  = 64;
} // namespace

TEST_CASE("a static scene caches the raster node while present/blit runs every frame", "[nodes][slice][driver]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			 ctx	= make_context();
	const vk::Device device = ctx.device();

	const auto		 vert = load(ctx, "tests/slice/triangle.vert");
	const auto		 frag = load(ctx, "tests/slice/triangle.frag");
	const std::array formats{COLOR};
	auto			 pipeline =
		veng::GraphicsPipelineBuilder(vert, frag)
			.color_formats(formats)
			.rasterization(vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise)
			.build(ctx);
	REQUIRE(pipeline.has_value());

	// The acquired present target (offscreen stand-in for a swapchain image).
	auto target = veng::Image::create(ctx.allocator(), device, vk::Extent2D{SIDE, SIDE}, COLOR,
									  vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc);
	REQUIRE(target.has_value());

	// Graph: ScreenSize -> raster -> scene token; present depends on scene token + swapchain source.
	Graph			 graph;
	auto			 screen		   = graph.add_source<vk::Extent2D>(vk::Extent2D{SIDE, SIDE});
	auto			 swapchain	   = graph.add_source<std::uint64_t>(0); // dirtied every frame (the "acquire" clock)
	const DataHandle scene_token   = graph.add(std::make_unique<ValueData<int>>(0));
	const DataHandle present_token = graph.add(std::make_unique<ValueData<int>>(0));

	auto  raster	 = std::make_unique<veng::nodes::RasterTriangleNode>(std::move(pipeline.value()), COLOR,
																		 static_cast<DataHandle>(screen), scene_token);
	auto* raster_ptr = raster.get();
	const NodeHandle raster_node = graph.add(std::move(raster));
	graph.set_producer(scene_token, raster_node);

	auto  present	  = std::make_unique<veng::nodes::PresentNode>(*raster_ptr, target.value(), scene_token,
																   static_cast<DataHandle>(swapchain), present_token);
	auto* present_ptr = present.get();
	const NodeHandle present_node = graph.add(std::move(present));
	graph.set_producer(present_token, present_node);

	auto* swapchain_data = dynamic_cast<ValueData<std::uint64_t>*>(graph.get_data(static_cast<DataHandle>(swapchain)));
	REQUIRE(swapchain_data != nullptr);

	veng::CommandManager commands(ctx);
	const auto			 fence = device.createFence({});
	REQUIRE(fence.result == vk::Result::eSuccess);
	InlineScheduler scheduler;

	std::uint64_t acquire = 0;
	// One driver frame: bump the swapchain source, record the demanded plan into a
	// command buffer via a GpuExecContext, submit, wait, recycle. Returns the plan.
	const auto render_frame = [&]() -> FramePlan
	{
		swapchain_data->set(++acquire);
		auto cmd = commands.begin(veng::QueueKind::Graphics, 0);
		REQUIRE(cmd.has_value());

		veng::gpu::GpuExecContext gpu_ctx(graph, ctx, *cmd, 0);
		const std::array		  sinks{present_token};
		auto					  plan = graph.resolve(sinks);
		REQUIRE(plan.has_value());
		graph.execute(*plan, scheduler, gpu_ctx);

		REQUIRE(cmd->end() == vk::Result::eSuccess);
		REQUIRE(ctx.graphics_queue().submit(vk::SubmitInfo().setCommandBuffers(*cmd), fence.value) ==
				vk::Result::eSuccess);
		REQUIRE(device.waitForFences(fence.value, vk::True, UINT64_MAX) == vk::Result::eSuccess);
		REQUIRE(device.resetFences(fence.value) == vk::Result::eSuccess);
		commands.reset_frame(0);
		return std::move(plan.value());
	};

	// Frame 0 is cold: the raster node and the present node both run.
	const FramePlan cold = render_frame();
	REQUIRE(plan_contains(cold, raster_node));
	REQUIRE(plan_contains(cold, present_node));
	REQUIRE(raster_ptr->scene() != nullptr);

	// Frames 1..5 with a static scene: ONLY the present node runs each frame; the
	// expensive raster render is cached (this is the design's headline claim).
	constexpr int STATIC_FRAMES = 5;
	for (int i = 0; i < STATIC_FRAMES; ++i)
	{
		const FramePlan plan = render_frame();
		REQUIRE(plan.size() == 1);
		REQUIRE(plan_contains(plan, present_node));
		REQUIRE_FALSE(plan_contains(plan, raster_node)); // raster cached — runs 0 times
	}
	REQUIRE(present_ptr->record_count() == 1 + STATIC_FRAMES); // present ran every frame

	// A scene-source change (resize) re-invalidates the raster node: it returns to the
	// plan. (Resolve-only — executing would need a matching-size target.)
	swapchain_data->set(++acquire);
	dynamic_cast<ValueData<vk::Extent2D>*>(graph.get_data(static_cast<DataHandle>(screen)))
		->set(vk::Extent2D{SIDE * 2, SIDE * 2});
	const auto after_resize = graph.resolve(std::array{present_token});
	REQUIRE(after_resize.has_value());
	REQUIRE(plan_contains(*after_resize, raster_node)); // resize re-runs the raster node

	device.destroyFence(fence.value);
}

TEST_CASE("the presented target receives the rendered triangle", "[nodes][slice][present]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			 ctx	= make_context();
	const vk::Device device = ctx.device();

	const auto		 vert = load(ctx, "tests/slice/triangle.vert");
	const auto		 frag = load(ctx, "tests/slice/triangle.frag");
	const std::array formats{COLOR};
	auto			 pipeline =
		veng::GraphicsPipelineBuilder(vert, frag)
			.color_formats(formats)
			.rasterization(vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise)
			.build(ctx);
	REQUIRE(pipeline.has_value());

	auto target = veng::Image::create(ctx.allocator(), device, vk::Extent2D{SIDE, SIDE}, COLOR,
									  vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc);
	REQUIRE(target.has_value());
	auto staging =
		veng::Buffer::create(ctx.allocator(), static_cast<vk::DeviceSize>(SIDE) * SIDE * 4,
							 vk::BufferUsageFlagBits::eTransferDst, vma::MemoryUsage::eAuto,
							 vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessRandom);
	REQUIRE(staging.has_value());

	Graph			 graph;
	auto			 screen		 = graph.add_source<vk::Extent2D>(vk::Extent2D{SIDE, SIDE});
	auto			 swapchain	 = graph.add_source<std::uint64_t>(0);
	const DataHandle scene_token = graph.add(std::make_unique<ValueData<int>>(0));
	const DataHandle present_tok = graph.add(std::make_unique<ValueData<int>>(0));
	auto  raster	 = std::make_unique<veng::nodes::RasterTriangleNode>(std::move(pipeline.value()), COLOR,
																		 static_cast<DataHandle>(screen), scene_token);
	auto* raster_ptr = raster.get();
	graph.set_producer(scene_token, graph.add(std::move(raster)));
	auto present = std::make_unique<veng::nodes::PresentNode>(*raster_ptr, target.value(), scene_token,
															  static_cast<DataHandle>(swapchain), present_tok);
	graph.set_producer(present_tok, graph.add(std::move(present)));

	const auto pool =
		device.createCommandPool(vk::CommandPoolCreateInfo().setQueueFamilyIndex(ctx.queue_indices().graphics));
	REQUIRE(pool.result == vk::Result::eSuccess);
	const auto cmds = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo()
														.setCommandPool(pool.value)
														.setLevel(vk::CommandBufferLevel::ePrimary)
														.setCommandBufferCount(1));
	REQUIRE(cmds.result == vk::Result::eSuccess);
	const vk::CommandBuffer cmd = cmds.value.front();
	REQUIRE(cmd.begin(vk::CommandBufferBeginInfo().setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)) ==
			vk::Result::eSuccess);

	veng::gpu::GpuExecContext gpu_ctx(graph, ctx, cmd, 0);
	InlineScheduler			  scheduler;
	const std::array		  sinks{present_tok};
	const auto				  plan = graph.resolve(sinks);
	REQUIRE(plan.has_value());
	graph.execute(*plan, scheduler, gpu_ctx); // raster -> present (copy) into target, left TRANSFER_SRC

	const auto layers = vk::ImageSubresourceLayers().setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1);
	cmd.copyImageToBuffer(
		target->image(), vk::ImageLayout::eTransferSrcOptimal, staging->buffer(),
		vk::BufferImageCopy().setImageSubresource(layers).setImageExtent(vk::Extent3D{SIDE, SIDE, 1}));
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {},
						vk::MemoryBarrier()
							.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
							.setDstAccessMask(vk::AccessFlagBits::eHostRead),
						{}, {});
	REQUIRE(cmd.end() == vk::Result::eSuccess);

	const auto fence = device.createFence({});
	REQUIRE(fence.result == vk::Result::eSuccess);
	REQUIRE(ctx.graphics_queue().submit(vk::SubmitInfo().setCommandBuffers(cmd), fence.value) == vk::Result::eSuccess);
	REQUIRE(device.waitForFences(fence.value, vk::True, UINT64_MAX) == vk::Result::eSuccess);

	const auto* pixels = static_cast<const std::uint8_t*>(staging->mapped());
	const auto	at	   = [&](std::uint32_t x, std::uint32_t y) { return (static_cast<std::size_t>(y) * SIDE + x) * 4; };
	REQUIRE(pixels[at(SIDE / 2, SIDE / 2) + 0] == 255); // center: triangle reached the presented target
	REQUIRE(pixels[at(0, 0) + 0] == 0);					// corner: clear

	device.destroyFence(fence.value);
	device.destroyCommandPool(pool.value);
}
