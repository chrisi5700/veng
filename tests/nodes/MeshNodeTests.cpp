//
// L4 test (design.md §L4): real buffer-backed geometry. A MeshNode uploads an actual
// vertex buffer (positions + per-vertex color) and an index buffer; a GraphicsNode bound to
// that mesh draws it indexed into a scene target. A readback proves both halves flowed
// through the VkBuffers: the triangle covers the center (positions) in its vertex color
// (per-vertex color), while the corner stays the clear color. Then the reactive claim — the
// mesh uploads exactly once (cold) and caches forever after (a static mesh never re-uploads).
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <glm/glm.hpp>
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
	auto result = veng::Context::create("MeshNode Test");
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

// Must match shaders/tests/slice/mesh_triangle.vert.slang: `float3 position; float3 color;`
// (tightly packed — two vec3, 24-byte stride — which is exactly what Slang reflects).
struct Vertex
{
	glm::vec3 position;
	glm::vec3 color;
};

constexpr vk::Format	COLOR = vk::Format::eR8G8B8A8Unorm;
constexpr std::uint32_t SIDE  = 64;
} // namespace

TEST_CASE("a MeshNode uploads real vertices an indexed GraphicsNode draws", "[nodes][mesh][slice]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			 ctx	= make_context();
	const vk::Device device = ctx.device();

	// A centered green triangle, as real vertex data + indices (not SV_VertexID). Same
	// positions as the SV_VertexID slice triangle; green proves the color came from the buffer.
	const std::vector<Vertex> vertices{
		{.position = {0.0F, -0.6F, 0.0F}, .color = {0.0F, 1.0F, 0.0F}},
		{.position = {0.6F, 0.6F, 0.0F}, .color = {0.0F, 1.0F, 0.0F}},
		{.position = {-0.6F, 0.6F, 0.0F}, .color = {0.0F, 1.0F, 0.0F}},
	};
	const std::vector<std::uint32_t> indices{0, 1, 2};

	// Graph: MeshNode -> mesh edge; ScreenSize source; GraphicsNode draws the mesh -> scene.
	Graph			 graph;
	auto			 screen = graph.add_source<vk::Extent2D>(vk::Extent2D{SIDE, SIDE});
	const DataHandle mesh	= graph.add(std::make_unique<ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));
	const DataHandle token	= graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));

	auto			 mesh_node = std::make_unique<veng::nodes::MeshNode>(std::span<const Vertex>(vertices),
																		 std::span<const std::uint32_t>(indices), mesh);
	const NodeHandle mesh_handle = graph.add(std::move(mesh_node));
	graph.set_producer(mesh, mesh_handle);

	auto node =
		std::make_unique<veng::nodes::GraphicsNode>("tests/slice/mesh_triangle.vert", "tests/slice/mesh_triangle.frag",
													COLOR, vk::Format::eUndefined, 0, screen, token);
	node->set_mesh(mesh);
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

	veng::ResourcePool res_pool(ctx.device(), ctx.allocator(), 1);
	res_pool.begin_frame(0);
	veng::gpu::GpuExecContext gpu_ctx(graph, ctx, res_pool, cmd, 0);
	InlineScheduler			  scheduler;
	const auto				  plan = graph.resolve(std::array{token});
	REQUIRE(plan.has_value());
	REQUIRE(plan_contains(*plan, mesh_handle)); // cold: the mesh uploads
	REQUIRE(plan_contains(*plan, node_handle)); // ...and the draw runs
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

	// Center is inside the triangle, in the vertex color (green): green came from the buffer,
	// position came from the buffer. The corner is outside — the black clear color.
	const std::size_t center = at(SIDE / 2, SIDE / 2);
	REQUIRE(pixels[center + 0] == 0);	// R
	REQUIRE(pixels[center + 1] == 255); // G — the per-vertex color
	REQUIRE(pixels[center + 2] == 0);	// B
	REQUIRE(pixels[at(0, 0) + 1] == 0); // corner G: clear

	device.destroyFence(fence.value);
	device.destroyCommandPool(pool.value);

	// Reactive claim: the mesh is static, so a second resolve caches everything — the upload
	// never re-runs.
	const auto held = graph.resolve(std::array{token});
	REQUIRE(held.has_value());
	REQUIRE(held->empty());
}
