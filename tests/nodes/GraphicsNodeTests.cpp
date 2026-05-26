//
// L4 test (design.md §L4): the generic GraphicsNode draws the cube shaders from an MVP
// push-constant — there is no cube-specific node. A pure transform turns (angle, screen)
// into the MVP; the GraphicsNode draws 36 vertices with it into a depth-tested target. A
// readback proves the cube lands in the center over the dark clear corner — which also
// guards the glm<->Slang push-constant matrix convention (a transposed MVP would throw
// the cube off-center). A held angle then proves the whole subgraph caches; a new angle
// re-invalidates it.
//

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/nodes/GraphicsNode.hpp>
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
	auto result = veng::Context::create("GraphicsNode Test");
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

glm::mat4 cube_mvp(const float& spin, const vk::Extent2D& size)
{
	const float aspect = static_cast<float>(size.width) / static_cast<float>(size.height);
	glm::mat4	proj   = glm::perspective(glm::radians(55.0F), aspect, 0.1F, 20.0F);
	proj[1][1] *= -1.0F;
	const glm::mat4 view  = glm::lookAt(glm::vec3(2.4F, 1.7F, 2.4F), glm::vec3(0.0F), glm::vec3(0.0F, 1.0F, 0.0F));
	const glm::mat4 model = glm::rotate(glm::mat4(1.0F), spin, glm::normalize(glm::vec3(0.3F, 1.0F, 0.2F)));
	return proj * view * model;
}

constexpr vk::Format	COLOR = vk::Format::eR8G8B8A8Unorm;
constexpr vk::Format	DEPTH = vk::Format::eD32Sfloat;
constexpr std::uint32_t SIDE  = 128;
} // namespace

TEST_CASE("a generic GraphicsNode draws a centered cube from an MVP edge and caches a held angle",
		  "[nodes][graphics][slice]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			 ctx	= make_context();
	const vk::Device device = ctx.device();

	// Graph: (screen, angle) -> mvp transform -> GraphicsNode -> scene image. The cube is
	// just the shaders + this matrix edge — no cube node, no pre-built pipeline.
	Graph			 graph;
	auto			 screen = graph.add_source<vk::Extent2D>(vk::Extent2D{SIDE, SIDE});
	auto			 angle	= graph.add_source<float>(0.0F);
	auto			 mvp	= graph.add_transform(cube_mvp, angle, screen);
	const DataHandle token	= graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));

	auto node = std::make_unique<veng::nodes::GraphicsNode>("demo/cube.vert", "demo/cube.frag", COLOR, DEPTH, 36,
															screen, token);
	node->push_constant<glm::mat4>(mvp, vk::ShaderStageFlagBits::eVertex);
	auto*			 node_ptr	 = node.get();
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
	REQUIRE(plan_contains(*plan, node_handle)); // cold: the cube draw runs (alongside the mvp transform)
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

	// Caching: with the angle unchanged the whole cube subgraph is cached (drops out of the
	// plan); a new angle re-invalidates it. This is the reactive thesis at the node level.
	const auto held = graph.resolve(std::array{token});
	REQUIRE(held.has_value());
	REQUIRE(held->empty());

	graph.set(angle, 0.5F);
	const auto rotated = graph.resolve(std::array{token});
	REQUIRE(rotated.has_value());
	REQUIRE(plan_contains(*rotated, node_handle)); // a changed angle re-runs the draw
}
