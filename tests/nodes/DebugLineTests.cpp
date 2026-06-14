//
// L4 test: line-topology + reactive vertex upload. A DynamicMeshNode publishes a
// `vector<DebugVertex>` (re-uploaded each frame) and a GraphicsNode in `eLineList` topology
// draws those vertices as line segments. The readback proves that (a) the line topology
// reached the pipeline (pixels along the line position are lit) and (b) the dynamic VB ride
// works (the upload landed in the buffer the draw bound). Asserted on a small offscreen
// target.
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <utility>
#include <vector>
#include <veng/context/Context.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/nodes/DynamicMeshNode.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/resources/ResourcePool.hpp>

using namespace veng::graph;

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("DebugLine Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

struct DebugVertex
{
	glm::vec3 position;
	glm::vec3 color;

	friend bool operator==(const DebugVertex&, const DebugVertex&) noexcept = default;
};

constexpr veng::rhi::Format COLOR = veng::rhi::Format::RGBA8_UNORM;
constexpr std::uint32_t		SIDE  = 64;
} // namespace

TEST_CASE("a DynamicMeshNode + line-topology GraphicsNode renders dynamic line segments",
		  "[nodes][debug_line][dynamic_mesh]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			 ctx	= make_context();
	const vk::Device device = ctx.device();

	// A single horizontal red line at NDC y=0.1 (not exactly y=0, which lands on a Vulkan
	// pixel boundary and may be skipped by the diamond-exit rule). y=0.1 → pixel y≈35 on a
	// 64-pixel target, comfortably inside a single row.
	std::vector<DebugVertex> lines{DebugVertex{.position = {-0.9F, 0.1F, 0.5F}, .color = {1.0F, 0.0F, 0.0F}},
								   DebugVertex{.position = {0.9F, 0.1F, 0.5F}, .color = {1.0F, 0.0F, 0.0F}}};

	Graph			 graph;
	auto			 screen	   = graph.add_source<vk::Extent2D>(vk::Extent2D{SIDE, SIDE});
	auto			 verts_src = graph.add_source<std::vector<DebugVertex>>(lines);
	const DataHandle mesh	   = graph.add(std::make_unique<ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));
	const DataHandle token	   = graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));

	auto mesh_node = std::make_unique<veng::nodes::DynamicMeshNode>(verts_src, mesh);
	graph.set_producer(mesh, graph.add(std::move(mesh_node)));

	// The line-list pass: no instancing, no descriptors. The view_proj push constant is an
	// identity matrix so positions land in NDC verbatim — keeps the readback math trivial.
	auto	   draw = std::make_unique<veng::nodes::GraphicsNode>("demo/debug_line.vert", "demo/debug_line.frag", COLOR,
																  veng::rhi::Format::UNDEFINED, 0, screen, token);
	const auto view_proj = graph.add_source<glm::mat4>(glm::mat4(1.0F));
	draw->set_mesh(mesh).topology(veng::rhi::Topology::LINE_LIST).push_constant<glm::mat4>(view_proj);
	auto*			 draw_ptr	 = draw.get();
	const NodeHandle draw_handle = graph.add(std::move(draw));
	graph.set_producer(token, draw_handle);

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
	// The line is at NDC y=0.1 → framebuffer y ≈ 35 (SIDE/2 * (1 + 0.1) = 35.2). The exact
	// row depends on rasterization rounding; scan a small vertical window around y=35 for
	// any red pixel in the middle column. Strict pixel-precision asserts on lines are
	// brittle; this confirms "a red line landed somewhere here" which is what the test wants.
	bool found_red = false;
	for (std::uint32_t y = 33; y <= 37 && !found_red; ++y)
	{
		const std::size_t mid = at(SIDE / 2, y);
		if (pixels[mid + 0] == 255 && pixels[mid + 1] == 0 && pixels[mid + 2] == 0)
		{
			found_red = true;
		}
	}
	REQUIRE(found_red);
	REQUIRE(pixels[at(0, 0) + 0] == 0); // corner stays clear

	device.destroyFence(fence.value);
	device.destroyCommandPool(pool.value);

	// Reactive: replacing the lines re-invalidates the upload + the draw.
	const auto held = graph.resolve(std::array{token});
	REQUIRE(held.has_value());
	REQUIRE(held->empty());

	lines.front().color = {0.0F, 1.0F, 0.0F}; // change red -> green on one endpoint
	graph.set(verts_src, lines);
	const auto changed = graph.resolve(std::array{token});
	REQUIRE(changed.has_value());
	REQUIRE_FALSE(changed->empty()); // both DynamicMeshNode and the draw re-run
}
