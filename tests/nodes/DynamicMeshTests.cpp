//
// DynamicMeshNode coverage for the paths DebugLineTests (vertex-only, single size) skips: the
// indexed upload path, a buffer that grows across frames (re-acquiring larger pool copies), and the
// zero-element round-up-to-one-stride guard. The node's record() is a pure host memcpy into a mapped
// pool buffer plus a MeshRef publish, so these execute a plan and inspect the published ref — no
// rendering or submit required.
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <veng/context/Context.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/managers/CommandManager.hpp>
#include <veng/managers/QueueKind.hpp>
#include <veng/nodes/DynamicMeshNode.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/ResourcePool.hpp>

using namespace veng::graph;

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("DynamicMesh Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

struct Vertex
{
	float		x, y, z;
	friend bool operator==(const Vertex&, const Vertex&) noexcept = default;
};

std::vector<Vertex> make_vertices(std::uint32_t n)
{
	std::vector<Vertex> v;
	v.reserve(n);
	for (std::uint32_t i = 0; i < n; ++i)
	{
		v.push_back(Vertex{static_cast<float>(i), 0.0F, 0.0F});
	}
	return v;
}

std::vector<std::uint32_t> make_indices(std::uint32_t n)
{
	std::vector<std::uint32_t> idx;
	idx.reserve(n);
	for (std::uint32_t i = 0; i < n; ++i)
	{
		idx.push_back(i);
	}
	return idx;
}
} // namespace

TEST_CASE("DynamicMeshNode uploads an indexed mesh and grows across frames", "[nodes][dynamic_mesh]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	Graph			 graph;
	auto			 verts_src	 = graph.add_source<std::vector<Vertex>>(make_vertices(3));
	auto			 indices_src = graph.add_source<std::vector<std::uint32_t>>(make_indices(3));
	const DataHandle mesh		 = graph.add(std::make_unique<ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));

	graph.set_producer(mesh, graph.add(std::make_unique<veng::nodes::DynamicMeshNode>(verts_src, indices_src, mesh)));

	veng::CommandManager commands(ctx);
	veng::ResourcePool	 pool(ctx.device(), ctx.allocator(), 1);
	InlineScheduler		 scheduler;
	const auto*			 mesh_slot = dynamic_cast<ValueData<veng::gpu::MeshRef>*>(graph.get_data(mesh));
	REQUIRE(mesh_slot != nullptr);

	// Records the demanded plan into a throwaway command buffer (the node's work is a host memcpy).
	const auto run = [&](std::uint64_t frame)
	{
		pool.begin_frame(frame);
		auto cmd = commands.begin(veng::QueueKind::Graphics, 0);
		REQUIRE(cmd.has_value());
		veng::gpu::GpuExecContext gpu_ctx(graph, ctx, pool, *cmd, 0);
		const auto				  plan = graph.resolve(std::array{mesh});
		REQUIRE(plan.has_value());
		REQUIRE(graph.execute(*plan, scheduler, gpu_ctx));
		REQUIRE(cmd->end() == vk::Result::eSuccess);
		commands.reset_frame(0);
	};

	SECTION("indexed upload publishes both buffers and counts")
	{
		run(0);
		const veng::gpu::MeshRef ref = mesh_slot->value();
		REQUIRE(ref.vertex_buffer);
		REQUIRE(ref.index_buffer); // the index edge was provided -> indexed draw
		REQUIRE(ref.vertex_count == 3);
		REQUIRE(ref.index_count == 3);
		REQUIRE(ref.vertex_stride == sizeof(Vertex));
	}

	SECTION("growing the vectors re-acquires larger buffers")
	{
		run(0);
		graph.set(verts_src, make_vertices(64)); // forces a buffer grow in the pool
		graph.set(indices_src, make_indices(96));
		run(1);
		const veng::gpu::MeshRef ref = mesh_slot->value();
		REQUIRE(ref.vertex_count == 64);
		REQUIRE(ref.index_count == 96);
		REQUIRE(ref.vertex_buffer);
		REQUIRE(ref.index_buffer);
	}

	SECTION("a zero-element mesh is legal (rounded up to one stride, no zero-size alloc)")
	{
		graph.set(verts_src, std::vector<Vertex>{});
		graph.set(indices_src, std::vector<std::uint32_t>{});
		run(0);
		const veng::gpu::MeshRef ref = mesh_slot->value();
		REQUIRE(ref.vertex_count == 0);
		REQUIRE(ref.index_count == 0);
		REQUIRE(ref.vertex_buffer); // a (one-stride) buffer was still allocated, never zero-size
	}

	(void)ctx.device().waitIdle();
}
