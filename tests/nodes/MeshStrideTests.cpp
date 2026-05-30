//
// L4 test (review.md item 2): MeshRef vertex-stride validation. A GraphicsNode bound to a mesh
// whose byte stride disagrees with the bound vertex shader's reflected input layout must fail the
// frame with a typed error, rather than stride the buffer wrong and draw garbage. The matching-
// stride control proves the check is not a false positive (a valid mesh still renders).
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
#include <veng/resources/ResourcePool.hpp>

using namespace veng::graph;

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("MeshStride Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

// shaders/tests/slice/mesh_triangle.vert reflects a 24-byte vertex: `float3 position; float3 color;`.
struct CorrectVertex
{
	glm::vec3 position;
	glm::vec3 color;
};
static_assert(sizeof(CorrectVertex) == 24);

// 20 bytes — deliberately the wrong stride for that shader (position + a 2-float attribute).
struct WrongVertex
{
	glm::vec3 position;
	glm::vec2 uv;
};
static_assert(sizeof(WrongVertex) == 20);

constexpr vk::Format	COLOR = vk::Format::eR8G8B8A8Unorm;
constexpr std::uint32_t SIDE  = 32;

// Build a one-frame graph (MeshNode -> GraphicsNode -> scene) for `vertices` and return whether
// the frame executed successfully. A wrong stride makes GraphicsNode::record fail the frame.
template <class Vertex>
bool run_frame_with_vertex(const std::vector<Vertex>& vertices)
{
	auto							 ctx	= make_context();
	const vk::Device				 device = ctx.device();
	const std::vector<std::uint32_t> indices{0, 1, 2};

	Graph			 graph;
	auto			 screen = graph.add_source<vk::Extent2D>(vk::Extent2D{SIDE, SIDE});
	const DataHandle mesh	= graph.add(std::make_unique<ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));
	const DataHandle token	= graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));

	const NodeHandle mesh_handle = graph.add(std::make_unique<veng::nodes::MeshNode>(
		std::span<const Vertex>(vertices), std::span<const std::uint32_t>(indices), mesh));
	graph.set_producer(mesh, mesh_handle);

	auto node =
		std::make_unique<veng::nodes::GraphicsNode>("tests/slice/mesh_triangle.vert", "tests/slice/mesh_triangle.frag",
													COLOR, vk::Format::eUndefined, 0, screen, token);
	node->set_mesh(mesh);
	const NodeHandle node_handle = graph.add(std::move(node));
	graph.set_producer(token, node_handle);

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

	veng::ResourcePool res_pool(device, ctx.allocator(), 1);
	res_pool.begin_frame(0);
	veng::gpu::GpuExecContext gpu_ctx(graph, ctx, res_pool, cmd, 0);
	InlineScheduler			  scheduler;
	const auto				  plan = graph.resolve(std::array{token});
	REQUIRE(plan.has_value());
	const bool ok = graph.execute(*plan, scheduler, gpu_ctx);

	(void)cmd.end();
	device.destroyCommandPool(pool.value);
	return ok;
}
} // namespace

TEST_CASE("a mesh whose stride matches the shader renders", "[nodes][mesh][stride]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	const std::vector<CorrectVertex> verts{
		{.position = {0.0F, -0.6F, 0.0F}, .color = {0.0F, 1.0F, 0.0F}},
		{.position = {0.6F, 0.6F, 0.0F}, .color = {0.0F, 1.0F, 0.0F}},
		{.position = {-0.6F, 0.6F, 0.0F}, .color = {0.0F, 1.0F, 0.0F}},
	};
	REQUIRE(run_frame_with_vertex(verts)); // 24-byte stride == reflected: the frame succeeds
}

TEST_CASE("a mesh whose stride mismatches the shader fails the frame", "[nodes][mesh][stride]")
{
	// The mismatch logs an error by design; silence it so the expected-failure test stays quiet.
	veng::Logger::instance().set_level(spdlog::level::off);
	const std::vector<WrongVertex> verts{
		{.position = {0.0F, -0.6F, 0.0F}, .uv = {0.0F, 0.0F}},
		{.position = {0.6F, 0.6F, 0.0F}, .uv = {1.0F, 0.0F}},
		{.position = {-0.6F, 0.6F, 0.0F}, .uv = {0.0F, 1.0F}},
	};
	REQUIRE_FALSE(run_frame_with_vertex(verts)); // 20-byte stride != reflected 24: typed failure
}
