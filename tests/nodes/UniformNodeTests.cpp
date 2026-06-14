//
// L4 test: descriptor-set uniforms. A UniformNode uploads a tint into a
// uniform buffer and publishes it by name; a GraphicsNode bound with add_uniform writes it
// into a descriptor set at the reflected binding and the fragment shader reads it. The mesh
// is white, so the rendered color *is* the tint — a readback proves the uniform reached the
// shader through the descriptor set. Then the reactive claim: with the tint unchanged the
// whole subgraph caches; a new tint re-invalidates the upload and the draw (the uniform is a
// genuine reactive input, not baked in like a mesh).
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
#include <veng/gpu/UniformRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/nodes/MeshNode.hpp>
#include <veng/nodes/UniformNode.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/resources/ResourcePool.hpp>

using namespace veng::graph;

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("UniformNode Test");
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

struct Vertex
{
	glm::vec3 position;
	glm::vec3 color;
};

constexpr veng::rhi::Format COLOR = veng::rhi::Format::RGBA8_UNORM;
constexpr std::uint32_t		SIDE  = 64;
} // namespace

TEST_CASE("a UniformNode feeds a GraphicsNode descriptor by reflected name", "[nodes][uniform][slice]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			 ctx	= make_context();
	const vk::Device device = ctx.device();

	// A white centered triangle: with the tinted shader the output color is exactly the tint.
	const std::vector<Vertex> vertices{
		{.position = {0.0F, -0.6F, 0.0F}, .color = {1.0F, 1.0F, 1.0F}},
		{.position = {0.6F, 0.6F, 0.0F}, .color = {1.0F, 1.0F, 1.0F}},
		{.position = {-0.6F, 0.6F, 0.0F}, .color = {1.0F, 1.0F, 1.0F}},
	};
	const std::vector<std::uint32_t> indices{0, 1, 2};

	// Graph: MeshNode -> mesh; tint source -> UniformNode -> uniform; GraphicsNode draws the
	// mesh with the tint uniform -> scene.
	Graph			 graph;
	auto			 screen	 = graph.add_source<vk::Extent2D>(vk::Extent2D{SIDE, SIDE});
	auto			 tint	 = graph.add_source<glm::vec4>(glm::vec4{0.0F, 1.0F, 0.0F, 1.0F}); // green
	const DataHandle mesh	 = graph.add(std::make_unique<ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));
	const DataHandle uniform = graph.add(std::make_unique<ValueData<veng::gpu::UniformRef>>(veng::gpu::UniformRef{}));
	const DataHandle token	 = graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));

	auto mesh_node = std::make_unique<veng::nodes::MeshNode>(std::span<const Vertex>(vertices),
															 std::span<const std::uint32_t>(indices), mesh);
	graph.set_producer(mesh, graph.add(std::move(mesh_node)));

	auto			 uniform_node	= std::make_unique<veng::nodes::UniformNode>(tint, "tint", uniform);
	const NodeHandle uniform_handle = graph.add(std::move(uniform_node));
	graph.set_producer(uniform, uniform_handle);

	auto node = std::make_unique<veng::nodes::GraphicsNode>("tests/slice/mesh_triangle.vert", "tests/slice/tinted.frag",
															COLOR, veng::rhi::Format::UNDEFINED, 0, screen, token);
	node->set_mesh(mesh).add_uniform(uniform);
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
	REQUIRE(plan_contains(*plan, uniform_handle)); // cold: the uniform uploads
	REQUIRE(plan_contains(*plan, node_handle));	   // ...and the draw runs
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

	// Center is the white triangle tinted green => the uniform reached the fragment shader.
	const std::size_t center = at(SIDE / 2, SIDE / 2);
	REQUIRE(pixels[center + 0] == 0);	// R
	REQUIRE(pixels[center + 1] == 255); // G — the tint
	REQUIRE(pixels[center + 2] == 0);	// B
	REQUIRE(pixels[at(0, 0) + 1] == 0); // corner G: clear

	device.destroyFence(fence.value);
	device.destroyCommandPool(pool.value);

	// Reactive: an unchanged tint caches the whole subgraph; a new tint re-invalidates the
	// upload AND the draw (the uniform is a real input edge, not baked in).
	const auto held = graph.resolve(std::array{token});
	REQUIRE(held.has_value());
	REQUIRE(held->empty());

	graph.set(tint, glm::vec4{0.0F, 0.0F, 1.0F, 1.0F}); // blue
	const auto changed = graph.resolve(std::array{token});
	REQUIRE(changed.has_value());
	REQUIRE(plan_contains(*changed, uniform_handle)); // the upload re-runs
	REQUIRE(plan_contains(*changed, node_handle));	  // the draw re-runs
}
