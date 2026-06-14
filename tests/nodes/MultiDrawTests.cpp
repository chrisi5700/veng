//
// L4 test: the Pass/Draw split — ONE GraphicsNode hosts several draws. Two
// MeshNodes upload a red and a blue triangle; a single GraphicsNode draws both into one target,
// each with its own per-draw push constant (a model matrix translating it left vs. right). A
// readback proves both draws landed: red in the left half, blue in the right half, the corner
// left at the clear color — so the node looped its draw list, and each draw used its own mesh
// (color) and its own push constant (position). Then the reactive claim at the multi-draw level:
// a held frame caches the whole pass; changing one draw's transform re-runs the pass.
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <span>
#include <utility>
#include <vector>
#include <veng/context/Context.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/nodes/MeshNode.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/resources/ResourcePool.hpp>

using namespace veng::graph;

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("MultiDraw Test");
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

// Must match shaders/demo/mesh.vert.slang (`float3 position; float3 color;`, tightly packed)
// which transforms the position by an MVP push constant — here a plain translation per draw.
struct Vertex
{
	glm::vec3 position;
	glm::vec3 color;
};

// A small triangle around the origin in `color`, so a per-draw translation moves it cleanly
// into one half of the target.
std::vector<Vertex> triangle(glm::vec3 color)
{
	return {
		{.position = {0.0F, -0.4F, 0.0F}, .color = color},
		{.position = {0.45F, 0.4F, 0.0F}, .color = color},
		{.position = {-0.45F, 0.4F, 0.0F}, .color = color},
	};
}

constexpr veng::rhi::Format COLOR = veng::rhi::Format::RGBA8_UNORM;
constexpr std::uint32_t		SIDE  = 64;
} // namespace

TEST_CASE("one GraphicsNode draws two meshes with per-draw push constants", "[nodes][graphics][multidraw][slice]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			 ctx	= make_context();
	const vk::Device device = ctx.device();

	const std::vector<Vertex>		 red   = triangle({1.0F, 0.0F, 0.0F});
	const std::vector<Vertex>		 blue  = triangle({0.0F, 0.0F, 1.0F});
	const std::vector<std::uint32_t> index = {0, 1, 2};

	// Graph: two MeshNodes -> two mesh edges; two model-matrix sources (left / right). One
	// GraphicsNode draws both meshes into one scene image — a draw per mesh, each pushing its
	// own translation.
	Graph			 graph;
	auto			 screen = graph.add_source<vk::Extent2D>(vk::Extent2D{SIDE, SIDE});
	auto			 left  = graph.add_source<glm::mat4>(glm::translate(glm::mat4(1.0F), glm::vec3(-0.5F, 0.0F, 0.0F)));
	auto			 right = graph.add_source<glm::mat4>(glm::translate(glm::mat4(1.0F), glm::vec3(0.5F, 0.0F, 0.0F)));
	const DataHandle mesh_red  = graph.add(std::make_unique<ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));
	const DataHandle mesh_blue = graph.add(std::make_unique<ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));
	const DataHandle token	   = graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));

	graph.set_producer(mesh_red, graph.add(std::make_unique<veng::nodes::MeshNode>(
									 std::span<const Vertex>(red), std::span<const std::uint32_t>(index), mesh_red)));
	graph.set_producer(mesh_blue,
					   graph.add(std::make_unique<veng::nodes::MeshNode>(
						   std::span<const Vertex>(blue), std::span<const std::uint32_t>(index), mesh_blue)));

	auto node = std::make_unique<veng::nodes::GraphicsNode>("demo/mesh.vert", "tests/slice/mesh_triangle.frag", COLOR,
															veng::rhi::Format::UNDEFINED, 0, screen, token);
	node->add_draw(mesh_red).push_constant<glm::mat4>(left, veng::rhi::ShaderStage::VERTEX);
	node->add_draw(mesh_blue).push_constant<glm::mat4>(right, veng::rhi::ShaderStage::VERTEX);
	auto*			 node_ptr	 = node.get();
	const NodeHandle node_handle = graph.add(std::move(node));
	graph.set_producer(token, node_handle);

	auto staging =
		veng::Buffer::create(ctx.allocator(), ctx.rhi(), static_cast<vk::DeviceSize>(SIDE) * SIDE * 4,
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

	veng::ResourcePool res_pool(ctx.device(), ctx.rhi(), ctx.allocator(), 1);
	res_pool.begin_frame(0);
	veng::gpu::GpuExecContext gpu_ctx(graph, ctx, res_pool, cmd, 0);
	InlineScheduler			  scheduler;
	const auto				  plan = graph.resolve(std::array{token});
	REQUIRE(plan.has_value());
	REQUIRE(plan_contains(*plan, node_handle));
	REQUIRE(graph.execute(*plan, scheduler, gpu_ctx));
	REQUIRE(node_ptr->scene() != nullptr);

	const auto region =
		vk::BufferImageCopy()
			.setImageSubresource(
				vk::ImageSubresourceLayers().setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1))
			.setImageExtent(vk::Extent3D{SIDE, SIDE, 1});
	const auto* readback_ref = dynamic_cast<ValueData<veng::gpu::ImageRef>*>(graph.get_data(token));
	res_pool.transition_image(readback_ref->value().pool_id, cmd, vk::ImageLayout::eTransferSrcOptimal,
							  vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
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

	// Left-center is the first draw (red mesh, translated left); right-center is the second draw
	// (blue mesh, translated right). Each proves its own mesh AND its own push constant reached
	// the GPU. The corner is outside both triangles — the clear color.
	const std::size_t left_center  = at(SIDE / 4, SIDE / 2);	 // x = 16
	const std::size_t right_center = at(3 * SIDE / 4, SIDE / 2); // x = 48
	REQUIRE(pixels[left_center + 0] > 150);						 // red R
	REQUIRE(pixels[left_center + 2] < 80);						 // red has little blue
	REQUIRE(pixels[right_center + 2] > 150);					 // blue B
	REQUIRE(pixels[right_center + 0] < 80);						 // blue has little red
	REQUIRE(pixels[at(0, 0) + 0] < 40);							 // corner: clear
	REQUIRE(pixels[at(0, 0) + 2] < 40);

	device.destroyFence(fence.value);
	device.destroyCommandPool(pool.value);

	// Reactive at the multi-draw level: a held frame caches the whole pass (and both static
	// meshes); changing one draw's transform re-invalidates the pass so it re-records every draw.
	const auto held = graph.resolve(std::array{token});
	REQUIRE(held.has_value());
	REQUIRE(held->empty());

	graph.set(left, glm::translate(glm::mat4(1.0F), glm::vec3(-0.4F, 0.0F, 0.0F)));
	const auto moved = graph.resolve(std::array{token});
	REQUIRE(moved.has_value());
	REQUIRE(plan_contains(*moved, node_handle));
}
