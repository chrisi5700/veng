//
// L4 test (design.md §L4): RasterCubeNode draws a depth-tested cube from a reactive
// rotation angle into a persistent scene-color target. A readback proves the cube lands
// in the center of the frame (over the dark clear corner) — which also guards the
// glm<->Slang push-constant matrix convention: a transposed MVP would throw the cube off
// the center, leaving the clear color there. A held angle then proves the node caches.
//

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/nodes/RasterCubeNode.hpp>
#include <veng/pipelines/GraphicsPipeline.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/shader/Shader.hpp>

using namespace veng::graph;

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("RasterCube Test");
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
constexpr vk::Format	DEPTH = vk::Format::eD32Sfloat;
constexpr std::uint32_t SIDE  = 128;
} // namespace

TEST_CASE("RasterCubeNode renders a centered cube and caches a held angle", "[nodes][cube][slice]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			 ctx	= make_context();
	const vk::Device device = ctx.device();

	const auto		 vert = load(ctx, "demo/cube.vert");
	const auto		 frag = load(ctx, "demo/cube.frag");
	const std::array formats{COLOR};
	auto			 pipeline =
		veng::GraphicsPipelineBuilder(vert, frag)
			.color_formats(formats)
			.depth_format(DEPTH)
			.rasterization(vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise)
			.build(ctx);
	REQUIRE(pipeline.has_value());

	// Graph: ScreenSize + angle -> RasterCubeNode -> scene token.
	Graph			 graph;
	auto			 screen	  = graph.add_source<vk::Extent2D>(vk::Extent2D{SIDE, SIDE});
	auto			 angle	  = graph.add_source<float>(0.0F);
	const DataHandle token	  = graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));
	auto			 node	  = std::make_unique<veng::nodes::RasterCubeNode>(std::move(pipeline.value()), COLOR, DEPTH,
																			  static_cast<DataHandle>(screen),
																			  static_cast<DataHandle>(angle), token);
	auto*			 node_ptr = node.get();
	const NodeHandle node_handle = graph.add(std::move(node));
	graph.set_producer(token, node_handle);

	auto staging =
		veng::Buffer::create(ctx.allocator(), static_cast<vk::DeviceSize>(SIDE) * SIDE * 4,
							 vk::BufferUsageFlagBits::eTransferDst, vma::MemoryUsage::eAuto,
							 vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessRandom);
	REQUIRE(staging.has_value());

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
	const auto				  plan = graph.resolve(std::array{token});
	REQUIRE(plan.has_value());
	REQUIRE(plan->size() == 1); // cold: the cube runs
	graph.execute(*plan, scheduler, gpu_ctx);
	REQUIRE(node_ptr->scene() != nullptr);

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

	const auto* pixels = static_cast<const std::uint8_t*>(staging->mapped());
	const auto	at	   = [](std::uint32_t x, std::uint32_t y) { return (static_cast<std::size_t>(y) * SIDE + x) * 4; };
	const auto	brightest = [&](std::size_t offset)
	{ return std::max({pixels[offset + 0], pixels[offset + 1], pixels[offset + 2]}); };

	// Center is a (flat-colored) cube face — a face color has a 1.0 channel (255); the
	// clear color is near-black. A transposed/wrong MVP would not cover the center.
	REQUIRE(brightest(at(SIDE / 2, SIDE / 2)) > 150);
	// The cube does not reach the corner: it stays the dark clear color.
	REQUIRE(brightest(at(0, 0)) < 40);

	device.destroyFence(fence.value);
	device.destroyCommandPool(pool.value);

	// Caching: with the angle unchanged the cube is cached (drops out of the plan); a new
	// angle re-invalidates it. This is the reactive thesis at the node level.
	const auto held = graph.resolve(std::array{token});
	REQUIRE(held.has_value());
	REQUIRE(held->empty());

	auto* angle_data = dynamic_cast<ValueData<float>*>(graph.get_data(static_cast<DataHandle>(angle)));
	REQUIRE(angle_data != nullptr);
	angle_data->set(0.5F);
	const auto rotated = graph.resolve(std::array{token});
	REQUIRE(rotated.has_value());
	REQUIRE(plan_contains(*rotated, node_handle));
}
