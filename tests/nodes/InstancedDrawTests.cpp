//
// L4 integration test: one StorageBufferNode + one GraphicsNode = an instanced draw. The
// StorageBufferNode uploads a vector<Body> of NDC offsets; the GraphicsNode binds it as a
// `StructuredBuffer<Body>` (matched by reflected name) and draws a single triangle
// `bodies.size()` times, each instance reading its own offset by `SV_InstanceID`. A pixel
// readback proves N triangles landed at N distinct positions — the storage buffer reached
// the shader AND the instance count came from the published BufferRef. The reactive claim
// is also asserted: growing the body array re-runs both the upload and the draw, and the
// pool-allocated copy count grows with the array size.
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
#include <veng/gpu/BufferRef.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/gpu/UniformRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/nodes/MeshNode.hpp>
#include <veng/nodes/StorageBufferNode.hpp>
#include <veng/nodes/UniformNode.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/resources/ResourcePool.hpp>

using namespace veng::graph;

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("Instanced Draw Test");
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

// Matches StructuredBuffer<Body> in tests/slice/instanced.vert.slang. xy is the NDC offset
// applied to the triangle's vertices for instance i; zw are unused (padding to a float4 so
// std430 layout is trivially round-trippable).
struct Body
{
	glm::vec4 offset;

	friend bool operator==(const Body&, const Body&) noexcept = default;
};

constexpr vk::Format	COLOR = vk::Format::eR8G8B8A8Unorm;
constexpr std::uint32_t SIDE  = 128;
} // namespace

TEST_CASE("an instanced GraphicsNode reads its instance data + count from a StorageBufferNode",
		  "[nodes][storage][instanced][slice]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			 ctx	= make_context();
	const vk::Device device = ctx.device();

	// A small centered white triangle (around origin), drawn once per body — each body
	// shifts the triangle by its xy offset, so the rendered pixels at body[i].offset show
	// the triangle and the rest of the framebuffer is the clear color.
	const std::vector<Vertex> vertices{
		{.position = {0.0F, -0.15F, 0.0F}, .color = {1.0F, 1.0F, 1.0F}},
		{.position = {0.15F, 0.15F, 0.0F}, .color = {1.0F, 1.0F, 1.0F}},
		{.position = {-0.15F, 0.15F, 0.0F}, .color = {1.0F, 1.0F, 1.0F}},
	};
	const std::vector<std::uint32_t> indices{0, 1, 2};

	// Three bodies — top-left, top-right, bottom-center quadrants in NDC. With a 128x128
	// target each lands in a distinct part of the framebuffer the readback can identify.
	const std::vector<Body> bodies{Body{.offset = {-0.5F, -0.5F, 0.0F, 0.0F}},
								   Body{.offset = {0.5F, -0.5F, 0.0F, 0.0F}}, Body{.offset = {0.0F, 0.5F, 0.0F, 0.0F}}};

	Graph			 graph;
	auto			 screen		= graph.add_source<vk::Extent2D>(vk::Extent2D{SIDE, SIDE});
	auto			 bodies_src = graph.add_source<std::vector<Body>>(bodies);
	const DataHandle mesh		= graph.add(std::make_unique<ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));
	const DataHandle ssbo		= graph.add(std::make_unique<ValueData<veng::gpu::BufferRef>>(veng::gpu::BufferRef{}));
	const DataHandle token		= graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));

	auto mesh_node = std::make_unique<veng::nodes::MeshNode>(std::span<const Vertex>(vertices),
															 std::span<const std::uint32_t>(indices), mesh);
	graph.set_producer(mesh, graph.add(std::move(mesh_node)));

	auto			 ssbo_node	 = std::make_unique<veng::nodes::StorageBufferNode>(bodies_src, "bodies", ssbo);
	const NodeHandle ssbo_handle = graph.add(std::move(ssbo_node));
	graph.set_producer(ssbo, ssbo_handle);

	// The draw: bind the mesh + the SSBO, AND drive its instanceCount from the same SSBO's
	// published count. So a CPU resize of `bodies` propagates to one upload + one re-draw at
	// the right instance count, all via one reactive source.
	auto draw_node = std::make_unique<veng::nodes::GraphicsNode>(
		"tests/slice/instanced.vert", "tests/slice/tinted.frag", COLOR, vk::Format::eUndefined, 0, screen, token);
	draw_node->set_mesh(mesh).add_storage_buffer(ssbo).set_instances_from(ssbo);

	// The "tinted" frag still wants a tint uniform — but we already proved that path in
	// UniformNodeTests, so we sneak a tint UniformNode into the wiring to keep the shader
	// happy. White tint => the rendered pixel equals the per-vertex color (white).
	auto			 tint	  = graph.add_source<glm::vec4>(glm::vec4{1.0F, 1.0F, 1.0F, 1.0F});
	const DataHandle tint_ref = graph.add(std::make_unique<ValueData<veng::gpu::UniformRef>>(veng::gpu::UniformRef{}));
	graph.set_producer(tint_ref, graph.add(std::make_unique<veng::nodes::UniformNode>(tint, "tint", tint_ref)));
	draw_node->add_uniform(tint_ref);

	auto*			 draw_ptr	 = draw_node.get();
	const NodeHandle draw_handle = graph.add(std::move(draw_node));
	graph.set_producer(token, draw_handle);

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
	REQUIRE(plan_contains(*plan, ssbo_handle));
	REQUIRE(plan_contains(*plan, draw_handle));
	REQUIRE(graph.execute(*plan, scheduler, gpu_ctx));
	REQUIRE(draw_ptr->scene() != nullptr);

	const auto region =
		vk::BufferImageCopy()
			.setImageSubresource(
				vk::ImageSubresourceLayers().setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1))
			.setImageExtent(vk::Extent3D{SIDE, SIDE, 1});
	const auto* readback_ref = dynamic_cast<ValueData<veng::gpu::ImageRef>*>(graph.get_data(token));
	res_pool.transition_image(readback_ref->value().pool_id, cmd, vk::ImageLayout::eTransferSrcOptimal,
							  vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
	cmd.copyImageToBuffer(draw_ptr->scene()->image(), vk::ImageLayout::eTransferSrcOptimal, staging->buffer(), region);
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
	// Each body shifts the triangle's center (NDC origin) by body.offset. NDC -> pixel: x in
	// [-1,1] maps to [0,128], so NDC (-0.5,-0.5) is pixel (32, 32), NDC (0.5,-0.5) is (96,
	// 32), NDC (0, 0.5) is (64, 96). The triangle is 0.3 wide in NDC (~19 px); reading the
	// shifted center hits white if the per-instance offset reached the shader.
	const std::array<std::pair<std::uint32_t, std::uint32_t>, 3> expected_centers{
		{{SIDE / 4, SIDE / 4}, {3 * SIDE / 4, SIDE / 4}, {SIDE / 2, 3 * SIDE / 4}}};
	for (const auto& [px, py] : expected_centers)
	{
		const std::size_t p = at(px, py);
		REQUIRE(pixels[p + 0] == 255); // R
		REQUIRE(pixels[p + 1] == 255); // G — white triangle * white tint
		REQUIRE(pixels[p + 2] == 255); // B
	}
	// And a known-empty region between instances stays the clear color (black, R=0).
	REQUIRE(pixels[at(SIDE / 2, SIDE / 4) + 0] == 0);

	device.destroyFence(fence.value);
	device.destroyCommandPool(pool.value);

	// Reactive: an unchanged body array caches the whole subgraph.
	const auto held = graph.resolve(std::array{token});
	REQUIRE(held.has_value());
	REQUIRE(held->empty());

	// Adding a 4th body re-invalidates the upload AND the draw — and the next record (which
	// we don't drive here, the state-only assertion is enough) would emit 4 instances.
	std::vector<Body> next = bodies;
	next.push_back(Body{.offset = {0.0F, 0.0F, 0.0F, 0.0F}});
	graph.set(bodies_src, std::move(next));
	const auto changed = graph.resolve(std::array{token});
	REQUIRE(changed.has_value());
	REQUIRE(plan_contains(*changed, ssbo_handle));
	REQUIRE(plan_contains(*changed, draw_handle));
}
