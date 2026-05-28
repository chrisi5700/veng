//
// L4/L5 slice proof (design.md §1, §L4, §L5): the reactive rendering thesis, offscreen.
// A ScreenSize source -> RasterTriangleNode -> scene ImageData; a BlitNode blits that
// into a destination ImageData (the offscreen stand-in for a swapchain image) that is
// dirtied every frame. A minimal driver loop re-feeds the destination and runs
// frame(blit_sink) each iteration.
//
// The assertion is the whole point: with a static scene, the blit runs every frame but
// the raster node runs 0 extra times (cached); a ScreenSize change re-runs the raster
// node. The wiring rule (scene must not depend on the per-frame destination) is what
// makes this hold.
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/managers/CommandManager.hpp>
#include <veng/nodes/BlitNode.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/resources/Image.hpp>
#include <veng/resources/ResourcePool.hpp>

using namespace veng::graph;

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("Slice Driver Test");
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

TEST_CASE("a static scene caches the raster node while the blit runs every frame", "[nodes][slice][driver]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			 ctx	= make_context();
	const vk::Device device = ctx.device();

	// The blit destination (offscreen stand-in for a swapchain image).
	auto target = veng::Image::create(ctx.allocator(), device, vk::Extent2D{SIDE, SIDE}, COLOR,
									  vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc);
	REQUIRE(target.has_value());
	const veng::gpu::ImageRef target_ref{.image = target->image(), .extent = {SIDE, SIDE}, .format = COLOR};

	// Graph: ScreenSize -> raster -> scene image; blit depends on the scene image + a
	// destination image fed in (and re-dirtied) every frame.
	Graph			 graph;
	auto			 screen		 = graph.add_source<vk::Extent2D>(vk::Extent2D{SIDE, SIDE});
	auto			 dst		 = graph.add_source<veng::gpu::ImageRef>(target_ref); // re-fed each frame
	const DataHandle scene_image = graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));
	const DataHandle presented_image =
		graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));

	auto  raster = std::make_unique<veng::nodes::GraphicsNode>("tests/slice/triangle.vert", "tests/slice/triangle.frag",
															   COLOR, vk::Format::eUndefined, 3, screen, scene_image);
	auto* raster_ptr			 = raster.get();
	const NodeHandle raster_node = graph.add(std::move(raster));
	graph.set_producer(scene_image, raster_node);

	auto			 blit	   = std::make_unique<veng::nodes::BlitNode>(scene_image, dst, presented_image,
																		 vk::ImageLayout::eTransferSrcOptimal);
	auto*			 blit_ptr  = blit.get();
	const NodeHandle blit_node = graph.add(std::move(blit));
	graph.set_producer(presented_image, blit_node);

	veng::CommandManager commands(ctx);
	veng::ResourcePool	 res_pool(ctx.device(), ctx.allocator(), 1);
	std::uint64_t		 frame_index = 0;
	const auto			 fence		 = device.createFence({});
	REQUIRE(fence.result == vk::Result::eSuccess);
	InlineScheduler scheduler;

	// One driver frame: re-feed the destination (dirty every frame), record the demanded
	// plan into a command buffer via a GpuExecContext, submit, wait, recycle.
	const auto render_frame = [&]() -> FramePlan
	{
		res_pool.begin_frame(frame_index++);
		// Stand in for the swapchain handing out a fresh ref each frame (which is what makes
		// the blit re-run): bump a per-frame version on the dst so consecutive sets compare
		// unequal — gpu::ImageRef is value-comparable now, equality-on-same-target wouldn't
		// dirty the source otherwise.
		veng::gpu::ImageRef per_frame_ref = target_ref;
		per_frame_ref.version			  = frame_index;
		graph.set(dst, per_frame_ref);
		auto cmd = commands.begin(veng::QueueKind::Graphics, 0);
		REQUIRE(cmd.has_value());

		veng::gpu::GpuExecContext gpu_ctx(graph, ctx, res_pool, *cmd, 0);
		const std::array		  sinks{presented_image};
		auto					  plan = graph.resolve(sinks);
		REQUIRE(plan.has_value());
		REQUIRE(graph.execute(*plan, scheduler, gpu_ctx));

		REQUIRE(cmd->end() == vk::Result::eSuccess);
		REQUIRE(ctx.graphics_queue().submit(vk::SubmitInfo().setCommandBuffers(*cmd), fence.value) ==
				vk::Result::eSuccess);
		REQUIRE(device.waitForFences(fence.value, vk::True, UINT64_MAX) == vk::Result::eSuccess);
		REQUIRE(device.resetFences(fence.value) == vk::Result::eSuccess);
		commands.reset_frame(0);
		return std::move(plan.value());
	};

	// Frame 0 is cold: the raster node and the blit node both run.
	const FramePlan cold = render_frame();
	REQUIRE(plan_contains(cold, raster_node));
	REQUIRE(plan_contains(cold, blit_node));
	REQUIRE(raster_ptr->scene() != nullptr);

	// Frames 1..5 with a static scene: ONLY the blit node runs each frame; the expensive
	// raster render is cached (this is the design's headline claim).
	constexpr int STATIC_FRAMES = 5;
	for (int i = 0; i < STATIC_FRAMES; ++i)
	{
		const FramePlan plan = render_frame();
		REQUIRE(plan.size() == 1);
		REQUIRE(plan_contains(plan, blit_node));
		REQUIRE_FALSE(plan_contains(plan, raster_node)); // raster cached — runs 0 times
	}
	REQUIRE(blit_ptr->record_count() == 1 + STATIC_FRAMES); // blit ran every frame

	// A scene-source change (resize) re-invalidates the raster node: it returns to the
	// plan. (Resolve-only — executing would need a matching-size target.)
	graph.set(dst, target_ref);
	graph.set(screen, vk::Extent2D{SIDE * 2, SIDE * 2});
	const auto after_resize = graph.resolve(std::array{presented_image});
	REQUIRE(after_resize.has_value());
	REQUIRE(plan_contains(*after_resize, raster_node)); // resize re-runs the raster node

	device.destroyFence(fence.value);
}

TEST_CASE("the blit destination receives the rendered triangle", "[nodes][slice][present]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			 ctx	= make_context();
	const vk::Device device = ctx.device();

	auto target = veng::Image::create(ctx.allocator(), device, vk::Extent2D{SIDE, SIDE}, COLOR,
									  vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc);
	REQUIRE(target.has_value());
	const veng::gpu::ImageRef target_ref{.image = target->image(), .extent = {SIDE, SIDE}, .format = COLOR};
	auto					  staging =
		veng::Buffer::create(ctx.allocator(), static_cast<vk::DeviceSize>(SIDE) * SIDE * 4,
							 vk::BufferUsageFlagBits::eTransferDst, vma::MemoryUsage::eAuto,
							 vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessRandom);
	REQUIRE(staging.has_value());

	Graph			 graph;
	auto			 screen		 = graph.add_source<vk::Extent2D>(vk::Extent2D{SIDE, SIDE});
	auto			 dst		 = graph.add_source<veng::gpu::ImageRef>(target_ref);
	const DataHandle scene_image = graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));
	const DataHandle presented_image =
		graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));
	auto raster = std::make_unique<veng::nodes::GraphicsNode>("tests/slice/triangle.vert", "tests/slice/triangle.frag",
															  COLOR, vk::Format::eUndefined, 3, screen, scene_image);
	graph.set_producer(scene_image, graph.add(std::move(raster)));
	auto blit = std::make_unique<veng::nodes::BlitNode>(scene_image, dst, presented_image,
														vk::ImageLayout::eTransferSrcOptimal);
	graph.set_producer(presented_image, graph.add(std::move(blit)));

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

	veng::ResourcePool res_pool(ctx.device(), ctx.allocator(), 1);
	res_pool.begin_frame(0);
	veng::gpu::GpuExecContext gpu_ctx(graph, ctx, res_pool, cmd, 0);
	InlineScheduler			  scheduler;
	const std::array		  sinks{presented_image};
	const auto				  plan = graph.resolve(sinks);
	REQUIRE(plan.has_value());
	REQUIRE(graph.execute(*plan, scheduler, gpu_ctx)); // raster -> blit into target, left TRANSFER_SRC

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
	REQUIRE(pixels[at(SIDE / 2, SIDE / 2) + 0] == 255); // center: triangle reached the blit destination
	REQUIRE(pixels[at(0, 0) + 0] == 0);					// corner: clear

	device.destroyFence(fence.value);
	device.destroyCommandPool(pool.value);
}
