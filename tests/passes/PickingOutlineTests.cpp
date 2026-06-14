//
// PickingPass + OutlinePass — the two reactive sub-graph passes that previously had zero coverage.
// Both render offscreen (no surface), so they run on any driver.
//
//   * PickingPass: a triangle covering the centre is id-rendered; a pick at the centre decodes to
//     its id on retirement (the draw -> readback -> CPU -> decode -> callback path), and a pick at a
//     corner decodes to "no hit". Mirrors the ScreenshotNode peer-sink readback pattern.
//   * OutlinePass: the silhouette -> blur -> ring chain executes, and the reactive knobs re-run only
//     the affected nodes (set_color -> ring, set_width -> blur), the cached-silhouette claim.
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>
#include <veng/context/Context.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/gpu/Sink.hpp>
#include <veng/gpu/SubmitContext.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/managers/CommandManager.hpp>
#include <veng/managers/QueueKind.hpp>
#include <veng/nodes/MeshNode.hpp>
#include <veng/passes/OutlinePass.hpp>
#include <veng/passes/PickingPass.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/ResourcePool.hpp>

using namespace veng::graph;

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("Picking/Outline Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

// The engine's standard mesh layout the picking/outline vertex shaders reflect: {position, normal,
// color}, tightly packed to 36 bytes. Only position is read; normal/color exist to match the stride.
struct MeshVertex
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color;
};
static_assert(sizeof(MeshVertex) == 36);

constexpr veng::rhi::Format COLOR = veng::rhi::Format::RGBA8_UNORM;
constexpr std::uint32_t		SIDE  = 32;

// A triangle (identity MVP, so positions are NDC) that covers the centre pixel but leaves the
// top corners clear — so a centre pick hits it and a corner pick hits the background.
DataHandle upload_centre_triangle(Graph& graph)
{
	static const std::vector<MeshVertex> verts{
		{{-0.9F, -0.9F, 0.5F}, {0, 0, 1}, {1, 1, 1}},
		{{0.9F, -0.9F, 0.5F}, {0, 0, 1}, {1, 1, 1}},
		{{0.0F, 0.9F, 0.5F}, {0, 0, 1}, {1, 1, 1}},
	};
	static const std::vector<std::uint32_t> idx{0, 1, 2};
	const DataHandle slot = graph.add(std::make_unique<ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));
	graph.set_producer(slot, graph.add(std::make_unique<veng::nodes::MeshNode>(
								 std::span<const MeshVertex>(verts), std::span<const std::uint32_t>(idx), slot)));
	return slot;
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
} // namespace

TEST_CASE("PickingPass decodes the object under a pixel and reports no-hit on the background", "[passes][picking]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			 ctx	= make_context();
	const vk::Device device = ctx.device();

	Graph			 graph;
	auto			 screen = graph.add_source<vk::Extent2D>(vk::Extent2D{SIDE, SIDE});
	auto			 mvp	= graph.add_source<glm::mat4>(glm::mat4(1.0F)); // identity -> positions are NDC
	const DataHandle mesh	= upload_centre_triangle(graph);

	veng::passes::PickingPass picking(graph, screen);
	constexpr std::uint32_t	  OBJECT_ID = 7;
	picking.add_object(mesh, mvp, OBJECT_ID);

	veng::ResourcePool	 pool(ctx.device(), ctx.rhi(), ctx.allocator(), 1);
	veng::CommandManager commands(ctx);
	InlineScheduler		 scheduler;

	// One full pick cycle: request -> resolve (pulls render + readback into the plan) -> execute ->
	// submit -> wait -> on_retired (decodes + fires the callback), exactly as the driver sequences it.
	const auto run_pick = [&](std::uint32_t px, std::uint32_t py, std::uint64_t frame) -> veng::passes::PickResult
	{
		std::optional<veng::passes::PickResult> result;
		picking.pick(px, py, [&](veng::passes::PickResult r) { result = r; });
		REQUIRE(picking.pending());

		pool.begin_frame(frame);
		auto cmd = commands.begin(veng::QueueKind::Graphics, 0);
		REQUIRE(cmd.has_value());
		veng::gpu::GpuExecContext gpu_ctx(graph, ctx, pool, *cmd, 0);
		const auto				  plan = graph.resolve(std::array{picking.done_token()});
		REQUIRE(plan.has_value());
		REQUIRE(plan_contains(*plan, picking.render_node()));
		REQUIRE(plan_contains(*plan, picking.readback_node()));
		REQUIRE(graph.execute(*plan, scheduler, gpu_ctx));
		REQUIRE(cmd->end() == vk::Result::eSuccess);

		const auto fence = device.createFence({});
		REQUIRE(fence.result == vk::Result::eSuccess);
		REQUIRE(ctx.graphics_queue().submit(vk::SubmitInfo().setCommandBuffers(*cmd), fence.value) ==
				vk::Result::eSuccess);

		veng::gpu::SubmitContext post(graph, ctx, 0);
		for (const NodeHandle h : plan->nodes())
		{
			if (auto* sink = dynamic_cast<veng::gpu::Sink*>(graph.get_node(h)))
			{
				sink->on_submitted(post);
			}
		}
		REQUIRE(device.waitForFences(fence.value, vk::True, UINT64_MAX) == vk::Result::eSuccess);
		for (const NodeHandle h : plan->nodes())
		{
			if (auto* sink = dynamic_cast<veng::gpu::Sink*>(graph.get_node(h)))
			{
				sink->on_retired(post); // decodes the queued pick + fires the callback
			}
		}
		device.destroyFence(fence.value);

		REQUIRE(result.has_value());	  // the callback fired on retirement
		REQUIRE_FALSE(picking.pending()); // and the pick is no longer in flight
		commands.reset_frame(0);
		return *result;
	};

	SECTION("a centre pixel decodes to the object id")
	{
		const veng::passes::PickResult hit = run_pick(SIDE / 2, SIDE / 2, 0);
		REQUIRE(hit.hit);
		REQUIRE(hit.id == OBJECT_ID);
		REQUIRE(hit.x == SIDE / 2);
		REQUIRE(hit.y == SIDE / 2);
	}

	SECTION("a background pixel decodes to no-hit")
	{
		const veng::passes::PickResult miss = run_pick(0, 0, 0); // top-left corner: outside the triangle
		REQUIRE_FALSE(miss.hit);
		REQUIRE(miss.id == 0);
	}

	(void)device.waitIdle();
}

TEST_CASE("OutlinePass renders the silhouette->blur->ring chain and re-runs only affected nodes", "[passes][outline]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	Graph			 graph;
	auto			 screen = graph.add_source<vk::Extent2D>(vk::Extent2D{SIDE, SIDE});
	auto			 mvp	= graph.add_source<glm::mat4>(glm::mat4(1.0F));
	const DataHandle mesh	= upload_centre_triangle(graph);
	const DataHandle glow	= graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));

	veng::passes::OutlinePass outline(graph, COLOR, screen, glow);
	outline.add_mesh(mesh, mvp);

	veng::ResourcePool	 pool(ctx.device(), ctx.rhi(), ctx.allocator(), 1);
	veng::CommandManager commands(ctx);
	InlineScheduler		 scheduler;

	const auto render = [&](std::uint64_t frame) -> FramePlan
	{
		pool.begin_frame(frame);
		auto cmd = commands.begin(veng::QueueKind::Graphics, 0);
		REQUIRE(cmd.has_value());
		veng::gpu::GpuExecContext gpu_ctx(graph, ctx, pool, *cmd, 0);
		auto					  plan = graph.resolve(std::array{glow});
		REQUIRE(plan.has_value());
		REQUIRE(graph.execute(*plan, scheduler, gpu_ctx));
		REQUIRE(cmd->end() == vk::Result::eSuccess);
		REQUIRE(ctx.graphics_queue().submit(vk::SubmitInfo().setCommandBuffers(*cmd), {}) == vk::Result::eSuccess);
		REQUIRE(ctx.device().waitIdle() == vk::Result::eSuccess);
		commands.reset_frame(0);
		return std::move(*plan);
	};

	// Cold frame: the whole chain runs.
	const FramePlan cold = render(0);
	REQUIRE(plan_contains(cold, outline.silhouette_node()));
	REQUIRE(plan_contains(cold, outline.blur_node()));
	REQUIRE(plan_contains(cold, outline.ring_node()));

	// Retint: only the ring re-runs; the silhouette and (horizontal) blur stay cached.
	outline.set_color({1.0F, 0.0F, 0.0F});
	const auto recolored = graph.resolve(std::array{glow});
	REQUIRE(recolored.has_value());
	REQUIRE(plan_contains(*recolored, outline.ring_node()));
	REQUIRE_FALSE(plan_contains(*recolored, outline.silhouette_node()));

	// Rescale: the blur re-runs (the ring downstream of it too); the silhouette stays cached.
	outline.set_width(2.5F);
	const auto rescaled = graph.resolve(std::array{glow});
	REQUIRE(rescaled.has_value());
	REQUIRE(plan_contains(*rescaled, outline.blur_node()));
	REQUIRE_FALSE(plan_contains(*rescaled, outline.silhouette_node()));

	(void)ctx.device().waitIdle();
}
