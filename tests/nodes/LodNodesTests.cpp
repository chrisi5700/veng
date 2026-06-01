//
// Unit tests for the discrete-LOD nodes: MeshSelectorNode (the mux) and CoverageLodNode (the
// screen-coverage metric). Both are pure CPU graph nodes, so these run headless — no GPU — driving
// real graph frames with the InlineScheduler and reading the output slots back. MeshRefs are
// fabricated with distinct vertex counts purely as identity tags.
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <vector>
#include <veng/gpu/MeshRef.hpp>
#include <veng/nodes/CoverageLodNode.hpp>
#include <veng/nodes/MeshSelectorNode.hpp>
#include <veng/rendergraph/Graph.hpp>

using namespace veng::graph;

namespace
{
template <class T>
const T& value_of(const Graph& graph, DataHandle handle)
{
	const auto* slot = dynamic_cast<const ValueData<T>*>(graph.get_data(handle));
	REQUIRE(slot != nullptr);
	return slot->value();
}

veng::gpu::MeshRef tagged_mesh(std::uint32_t tag)
{
	return veng::gpu::MeshRef{.vertex_count = tag, .index_count = tag * 3};
}
} // namespace

TEST_CASE("MeshSelectorNode forwards the mesh chosen by the level edge", "[nodes][lod]")
{
	Graph							graph;
	const std::array<DataHandle, 3> meshes{graph.add_source<veng::gpu::MeshRef>(tagged_mesh(10)),
										   graph.add_source<veng::gpu::MeshRef>(tagged_mesh(20)),
										   graph.add_source<veng::gpu::MeshRef>(tagged_mesh(30))};
	const auto						level = graph.add_source<std::uint32_t>(0U);
	const DataHandle out  = graph.add(std::make_unique<ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));
	const NodeHandle node = graph.add(std::make_unique<veng::nodes::MeshSelectorNode>(meshes, level.handle, out));
	graph.set_producer(out, node);

	InlineScheduler scheduler;

	REQUIRE(graph.frame(out, scheduler).has_value());
	CHECK(value_of<veng::gpu::MeshRef>(graph, out).vertex_count == 10); // level 0 = finest

	graph.set(level, 2U);
	REQUIRE(graph.frame(out, scheduler).has_value());
	CHECK(value_of<veng::gpu::MeshRef>(graph, out).vertex_count == 30); // switched, no re-upload

	graph.set(level, 1U);
	REQUIRE(graph.frame(out, scheduler).has_value());
	CHECK(value_of<veng::gpu::MeshRef>(graph, out).vertex_count == 20);
}

TEST_CASE("MeshSelectorNode clamps an out-of-range level", "[nodes][lod]")
{
	Graph							graph;
	const std::array<DataHandle, 2> meshes{graph.add_source<veng::gpu::MeshRef>(tagged_mesh(1)),
										   graph.add_source<veng::gpu::MeshRef>(tagged_mesh(2))};
	const auto						level = graph.add_source<std::uint32_t>(99U); // past the end
	const DataHandle out  = graph.add(std::make_unique<ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));
	const NodeHandle node = graph.add(std::make_unique<veng::nodes::MeshSelectorNode>(meshes, level.handle, out));
	graph.set_producer(out, node);

	InlineScheduler scheduler;
	REQUIRE(graph.frame(out, scheduler).has_value());
	CHECK(value_of<veng::gpu::MeshRef>(graph, out).vertex_count == 2); // clamped to the coarsest
}

namespace
{
// A graph wired with a CoverageLodNode whose camera distance is adjustable via the view source.
struct CoverageFixture
{
	Graph				   graph;
	TypedHandle<glm::mat4> view;
	TypedHandle<glm::mat4> proj;
	TypedHandle<glm::vec4> sphere;
	DataHandle			   level;
	InlineScheduler		   scheduler;

	CoverageFixture(std::vector<float> thresholds, float hysteresis)
		: view(graph.add_source<glm::mat4>(glm::mat4(1.0F)))
		, proj(graph.add_source<glm::mat4>(glm::perspective(glm::radians(60.0F), 1.0F, 0.1F, 100.0F)))
		, sphere(graph.add_source<glm::vec4>(glm::vec4(0.0F, 0.0F, 0.0F, 1.0F))) // unit sphere at origin
		, level(graph.add(std::make_unique<ValueData<std::uint32_t>>(0U)))
	{
		const NodeHandle node = graph.add(std::make_unique<veng::nodes::CoverageLodNode>(
			view.handle, proj.handle, sphere.handle, level, std::move(thresholds), hysteresis));
		graph.set_producer(level, node);
	}

	// Place the camera so the origin sphere sits at view-space depth `distance`, then run a frame.
	std::uint32_t at_distance(float distance)
	{
		graph.set(view, glm::translate(glm::mat4(1.0F), glm::vec3(0.0F, 0.0F, -distance)));
		REQUIRE(graph.frame(level, scheduler).has_value());
		return value_of<std::uint32_t>(graph, level);
	}
};
} // namespace

TEST_CASE("CoverageLodNode picks coarser levels as the object shrinks on screen", "[nodes][lod]")
{
	// 4 levels, no hysteresis so the mapping is exact. Thresholds are diameter-fractions of height.
	CoverageFixture fix({0.5F, 0.2F, 0.08F}, 0.0F);

	CHECK(fix.at_distance(2.0F) == 0);	// coverage ≈ 0.87 → finest
	CHECK(fix.at_distance(10.0F) == 2); // coverage ≈ 0.17 → mid
	CHECK(fix.at_distance(40.0F) == 3); // coverage ≈ 0.043 → coarsest
}

TEST_CASE("CoverageLodNode hysteresis prevents flicker at a threshold", "[nodes][lod]")
{
	// One threshold at 0.5 with a ±20% dead band → switch up only below 0.4, back down only above 0.6.
	// proj[1][1] ≈ 1.732, sphere radius 1, so coverage = 1.732 / distance.
	CoverageFixture fix({0.5F}, 0.2F);

	CHECK(fix.at_distance(0.5F) == 0);	// coverage ≈ 3.46 → finest
	CHECK(fix.at_distance(3.85F) == 0); // coverage ≈ 0.45: inside the dead band → stays finest
	CHECK(fix.at_distance(4.95F) == 1); // coverage ≈ 0.35 (< 0.4) → drops to coarse
	CHECK(fix.at_distance(3.85F) == 1); // coverage ≈ 0.45 again: dead band → stays coarse (no flicker)
	CHECK(fix.at_distance(2.66F) == 0); // coverage ≈ 0.65 (> 0.6) → back to finest
}
