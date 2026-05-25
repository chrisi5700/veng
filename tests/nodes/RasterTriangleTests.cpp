//
// L4 slice test (design.md §L4): the engine's first rendered frame. A ScreenSize source
// feeds a RasterTriangleNode that draws a triangle into a persistent scene-color image
// via dynamic rendering; the graph dispatches it through a GpuExecContext, and a
// readback proves a red triangle landed in the center over a black-cleared corner.
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
#include <veng/nodes/RasterTriangleNode.hpp>
#include <veng/pipelines/GraphicsPipeline.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/shader/Shader.hpp>

using namespace veng::graph;

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("RasterTriangle Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

veng::Shader load(const veng::Context& ctx, std::string_view name)
{
	auto result = veng::Shader::create_shader(ctx.device(), name);
	REQUIRE(result.has_value());
	return std::move(result.value());
}

constexpr vk::Format	COLOR = vk::Format::eR8G8B8A8Unorm;
constexpr std::uint32_t SIDE  = 64;
} // namespace

TEST_CASE("RasterTriangleNode renders a triangle into a scene target", "[nodes][raster][slice]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			 ctx	= make_context();
	const vk::Device device = ctx.device();

	// Pipeline from the slice triangle shaders, targeting one RGBA8 color attachment.
	const auto		 vert = load(ctx, "tests/slice/triangle.vert");
	const auto		 frag = load(ctx, "tests/slice/triangle.frag");
	const std::array formats{COLOR};
	// No culling: a single demo triangle, winding-agnostic.
	auto pipeline =
		veng::GraphicsPipelineBuilder(vert, frag)
			.color_formats(formats)
			.rasterization(vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise)
			.build(ctx);
	REQUIRE(pipeline.has_value());

	// Graph: ScreenSize source -> RasterTriangleNode -> scene token.
	Graph			 graph;
	auto			 screen = graph.add_source<vk::Extent2D>(vk::Extent2D{SIDE, SIDE});
	const DataHandle token	= graph.add(std::make_unique<ValueData<int>>(0));
	auto			 node	= std::make_unique<veng::nodes::RasterTriangleNode>(std::move(pipeline.value()), COLOR,
																				static_cast<DataHandle>(screen), token);
	auto*			 node_ptr	 = node.get();
	const NodeHandle node_handle = graph.add(std::move(node));
	graph.set_producer(token, node_handle);

	// Readback staging buffer (host-visible).
	auto staging =
		veng::Buffer::create(ctx.allocator(), static_cast<vk::DeviceSize>(SIDE) * SIDE * 4,
							 vk::BufferUsageFlagBits::eTransferDst, vma::MemoryUsage::eAuto,
							 vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessRandom);
	REQUIRE(staging.has_value());

	// One-shot command buffer + a GPU context to drive the frame.
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
	const std::array		  sinks{token};
	const auto				  plan = graph.resolve(sinks);
	REQUIRE(plan.has_value());
	REQUIRE(plan->size() == 1);
	graph.execute(*plan, scheduler, gpu_ctx); // records the triangle into the scene target

	REQUIRE(node_ptr->scene() != nullptr); // target was created

	// The node left the scene in TRANSFER_SRC; copy it out for inspection.
	const auto region =
		vk::BufferImageCopy()
			.setImageSubresource(
				vk::ImageSubresourceLayers().setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1))
			.setImageExtent(vk::Extent3D{SIDE, SIDE, 1});
	cmd.copyImageToBuffer(node_ptr->scene()->image(), vk::ImageLayout::eTransferSrcOptimal, staging->buffer(), region);
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

	// Read pixels (RGBA8, tightly packed). Center is inside the triangle (red); the
	// top-left corner is outside (cleared black).
	const auto* pixels		 = static_cast<const std::uint8_t*>(staging->mapped());
	const auto	pixel_offset = [](std::uint32_t x, std::uint32_t y)
	{ return (static_cast<std::size_t>(y) * SIDE + x) * 4; };

	const std::size_t center = pixel_offset(SIDE / 2, SIDE / 2);
	const std::size_t corner = pixel_offset(0, 0);
	REQUIRE(pixels[center + 0] == 255); // center R: triangle
	REQUIRE(pixels[center + 1] == 0);
	REQUIRE(pixels[corner + 0] == 0);	// corner R: clear color
	REQUIRE(pixels[corner + 3] == 255); // corner A: clear alpha

	device.destroyFence(fence.value);
	device.destroyCommandPool(pool.value);
}
