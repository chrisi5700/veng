//
// PhongPassTests — lock in veng::passes::PhongPass's face culling / winding and transparency
// ordering. The pass had a subtle bug: the camera projection negates Y for Vulkan clip space
// (proj[1][1] *= -1), which flips triangle winding in *framebuffer* space (where Vulkan decides
// facing). With the wrong FrontFace, eBack culled the *front* faces and the back-then-front
// transparency batches inverted, so translucent objects showed only their backfaces.
//
// These tests render through the real pass with the SAME Y-flipped projection the app uses, so a
// winding regression is caught: a front-facing opaque quad must survive culling, a back-facing
// one must be culled, and a translucent quad must blend over the opaque geometry behind it.
//
// Driving pattern mirrors RasterTriangleTests: a one-shot command buffer + a GpuExecContext run a
// single resolve/execute, then a manual readback of the pass's published ImageRef pool copy.
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
#include <veng/nodes/MeshNode.hpp>
#include <veng/passes/PhongPass.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/resources/ResourcePool.hpp>

using namespace veng::graph;

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("PhongPass Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

constexpr veng::rhi::Format COLOR = veng::rhi::Format::RGBA8_UNORM;
constexpr veng::rhi::Format DEPTH = veng::rhi::Format::D32_SFLOAT;
constexpr std::uint32_t		SIDE  = 64;

// The engine's standard vertex layout {position, normal, color} that phong.vert reflects.
struct Vtx
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color;
};

constexpr glm::vec3 WHITE{1.0F, 1.0F, 1.0F}; // PhongPass tints by the per-object push color

// A quad at depth z that FACES the camera (eye at +Z): its triangles are wound so that, under the
// camera's Y-flip and PhongPass's FrontFace eClockwise, they read as front-facing and survive the
// opaque pass's eBack cull. (Winding verified against the real pass in the tests below.)
std::vector<Vtx> quad_front(float z)
{
	return {
		{{-1.0F, -1.0F, z}, {0.0F, 0.0F, 1.0F}, WHITE}, {{1.0F, 1.0F, z}, {0.0F, 0.0F, 1.0F}, WHITE},
		{{1.0F, -1.0F, z}, {0.0F, 0.0F, 1.0F}, WHITE},	{{-1.0F, -1.0F, z}, {0.0F, 0.0F, 1.0F}, WHITE},
		{{-1.0F, 1.0F, z}, {0.0F, 0.0F, 1.0F}, WHITE},	{{1.0F, 1.0F, z}, {0.0F, 0.0F, 1.0F}, WHITE},
	};
}

// The same quad with reversed winding: BACK-facing toward the eye, so the opaque eBack pass culls it.
std::vector<Vtx> quad_back(float z)
{
	return {
		{{-1.0F, -1.0F, z}, {0.0F, 0.0F, 1.0F}, WHITE}, {{1.0F, -1.0F, z}, {0.0F, 0.0F, 1.0F}, WHITE},
		{{1.0F, 1.0F, z}, {0.0F, 0.0F, 1.0F}, WHITE},	{{-1.0F, -1.0F, z}, {0.0F, 0.0F, 1.0F}, WHITE},
		{{1.0F, 1.0F, z}, {0.0F, 0.0F, 1.0F}, WHITE},	{{-1.0F, 1.0F, z}, {0.0F, 0.0F, 1.0F}, WHITE},
	};
}

DataHandle upload(Graph& graph, std::span<const Vtx> verts)
{
	static const std::array<std::uint32_t, 6> idx{0, 1, 2, 3, 4, 5};
	const DataHandle slot = graph.add(std::make_unique<ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));
	graph.set_producer(
		slot, graph.add(std::make_unique<veng::nodes::MeshNode>(verts, std::span<const std::uint32_t>(idx), slot)));
	return slot;
}

// The same Vulkan Y-flipped perspective the OrbitCamera publishes — it is what makes winding
// matter, so the test reproduces it to exercise the real face-culling path.
glm::mat4 yflip_view_proj()
{
	const glm::mat4 view = glm::lookAt(glm::vec3{0.0F, 0.0F, 3.0F}, glm::vec3{0.0F}, glm::vec3{0.0F, 1.0F, 0.0F});
	glm::mat4		proj = glm::perspective(glm::radians(45.0F), 1.0F, 0.1F, 10.0F);
	proj[1][1] *= -1.0F;
	return proj * view;
}

// Resolve+execute the graph for `scene`, then read back the center pixel of the pass's output.
std::array<std::uint8_t, 4> render_center(veng::Context& ctx, Graph& graph, DataHandle scene)
{
	const vk::Device device = ctx.device();

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

	veng::ResourcePool res_pool(device, ctx.rhi(), ctx.allocator(), 1);
	res_pool.begin_frame(0);
	veng::gpu::GpuExecContext gpu_ctx(graph, ctx, res_pool, veng::rhi::CommandEncoder(cmd, ctx.rhi()), 0);
	InlineScheduler			  scheduler;
	const std::array		  sinks{scene};
	const auto				  plan = graph.resolve(sinks);
	REQUIRE(plan.has_value());
	REQUIRE(graph.execute(*plan, scheduler, gpu_ctx));

	const auto* ref = dynamic_cast<ValueData<veng::gpu::ImageRef>*>(graph.get_data(scene));
	REQUIRE(ref != nullptr);
	REQUIRE(ctx.rhi().image(ref->value().texture));

	// The pass leaves its output in eColorAttachmentOptimal; transition the pool copy to
	// TRANSFER_SRC, copy it out, then host-sync.
	res_pool.transition_image(ref->value().pool_id, cmd, vk::ImageLayout::eTransferSrcOptimal,
							  vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
	const auto region =
		vk::BufferImageCopy()
			.setImageSubresource(
				vk::ImageSubresourceLayers().setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1))
			.setImageExtent(vk::Extent3D{SIDE, SIDE, 1});
	cmd.copyImageToBuffer(ctx.rhi().image(ref->value().texture), vk::ImageLayout::eTransferSrcOptimal,
						  staging->buffer(), region);
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

	const auto*					pixels = static_cast<const std::uint8_t*>(staging->mapped());
	const std::size_t			off	   = (static_cast<std::size_t>(SIDE / 2) * SIDE + (SIDE / 2)) * 4;
	std::array<std::uint8_t, 4> center{pixels[off + 0], pixels[off + 1], pixels[off + 2], pixels[off + 3]};

	device.destroyFence(fence.value);
	device.destroyCommandPool(pool.value);
	return center;
}

// Wire a PhongPass with a Y-flipped camera over `graph`; returns the scene output edge.
DataHandle make_phong(Graph& graph, veng::passes::PhongPass*& out_pass,
					  std::vector<std::unique_ptr<veng::passes::PhongPass>>& owner)
{
	const auto screen = graph.add_source<veng::rhi::Extent2D>(veng::rhi::Extent2D{SIDE, SIDE});
	const auto vp	  = graph.add_source<glm::mat4>(yflip_view_proj());
	const auto eye	  = graph.add_source<glm::vec4>(glm::vec4(0.0F, 0.0F, 3.0F, 1.0F));
	const auto scene  = graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));
	owner.push_back(std::make_unique<veng::passes::PhongPass>(graph, COLOR, DEPTH, screen, scene, vp, eye));
	out_pass = owner.back().get();
	return scene;
}
} // namespace

TEST_CASE("PhongPass renders front-facing opaque geometry (winding correct)", "[gpu][passes][phong]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	Graph												  graph;
	veng::passes::PhongPass*							  phong = nullptr;
	std::vector<std::unique_ptr<veng::passes::PhongPass>> owner;
	const DataHandle									  scene = make_phong(graph, phong, owner);
	const auto											  model = graph.add_source<glm::mat4>(glm::mat4(1.0F));
	phong->add_object(upload(graph, quad_front(0.0F)), model, glm::vec4(0.9F, 0.1F, 0.1F, 1.0F)); // opaque red

	const auto px = render_center(ctx, graph, scene);
	// The front face survives culling and is lit red — not the dark clear color.
	REQUIRE(px[0] > 60);
	REQUIRE(px[0] > px[1]);
	REQUIRE(px[0] > px[2]);
}

TEST_CASE("PhongPass culls back-facing geometry", "[gpu][passes][phong]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	Graph												  graph;
	veng::passes::PhongPass*							  phong = nullptr;
	std::vector<std::unique_ptr<veng::passes::PhongPass>> owner;
	const DataHandle									  scene = make_phong(graph, phong, owner);
	const auto											  model = graph.add_source<glm::mat4>(glm::mat4(1.0F));
	phong->add_object(upload(graph, quad_back(0.0F)), model, glm::vec4(0.9F, 0.1F, 0.1F, 1.0F)); // back-facing red

	const auto px = render_center(ctx, graph, scene);
	// Back face is culled, so the dark clear color shows through (no bright red surface).
	REQUIRE(px[0] < 30);
}

TEST_CASE("PhongPass blends a translucent object over the opaque geometry behind it", "[gpu][passes][phong]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	Graph												  graph;
	veng::passes::PhongPass*							  phong = nullptr;
	std::vector<std::unique_ptr<veng::passes::PhongPass>> owner;
	const DataHandle									  scene = make_phong(graph, phong, owner);
	const auto											  model = graph.add_source<glm::mat4>(glm::mat4(1.0F));
	// Opaque green far quad, then a translucent red quad nearer the camera covering it. The eye is at
	// +Z, so a LARGER z is nearer: green far at z=-0.5, glass red near at z=+0.5.
	phong->add_object(upload(graph, quad_front(-0.5F)), model, glm::vec4(0.1F, 0.9F, 0.1F, 1.0F)); // green (far)
	phong->add_object(upload(graph, quad_front(0.5F)), model, glm::vec4(0.9F, 0.1F, 0.1F, 0.5F));  // glass red (near)

	const auto px = render_center(ctx, graph, scene);
	CAPTURE(px[0], px[1], px[2], px[3]);
	// The near quad is translucent, so the green behind shows through: both red (from the glass) and
	// green (from the opaque behind it) are present. Pure red would mean blending is off; pure green
	// would mean the glass was wrongly culled/occluded.
	REQUIRE(px[0] > 40); // red from the translucent front
	REQUIRE(px[1] > 40); // green from the opaque behind, seen through it
}
