//
// Unit tests for veng::assets::load_obj. An inline OBJ cube fixture (no fetch, no GPU — the loader
// only adds CPU nodes to a Graph) exercises tinyobjloader parse → de-index → the shared pipeline.
//

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <veng/assets/ObjLoader.hpp>
#include <veng/rendergraph/Graph.hpp>

namespace
{
std::filesystem::path write_cube_obj()
{
	const auto	  path = std::filesystem::temp_directory_path() / "veng_cube.obj";
	std::ofstream out(path);
	out << "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n";
	// Six quad faces (1-indexed); tinyobjloader triangulates them.
	out << "f 1 2 3 4\nf 5 6 7 8\nf 1 2 6 5\nf 4 3 7 8\nf 1 4 8 5\nf 2 3 7 6\n";
	return path;
}
} // namespace

TEST_CASE("load_obj parses + de-indexes a cube", "[assets][obj]")
{
	const auto path = write_cube_obj();

	veng::graph::Graph graph;
	const auto		   mesh = veng::assets::load_obj(graph, path.string());
	REQUIRE(mesh.has_value());
	CHECK(mesh->index_count == 36); // 6 quads → 12 triangles
	CHECK(mesh->vertex_count == 8); // 8 shared corners (no vt/vn to split on)
	CHECK(mesh->bounds_max.x == Catch::Approx(1.0F));
	CHECK(mesh->bounds_min.x == Catch::Approx(0.0F));

	std::filesystem::remove(path);
}

TEST_CASE("load_obj_lods builds a mesh edge per level", "[assets][obj]")
{
	const auto path = write_cube_obj();

	veng::graph::Graph							graph;
	const std::array<veng::assets::LodLevel, 2> levels{{{.target_ratio = 1.0F}, {.target_ratio = 0.5F}}};
	const auto									set = veng::assets::load_obj_lods(graph, path.string(), levels);
	REQUIRE(set.has_value());
	CHECK(set->meshes.size() == 2);
	CHECK(set->radius() > 0.0F);

	std::filesystem::remove(path);
}

TEST_CASE("load_obj reports a clear error for a missing file", "[assets][obj]")
{
	veng::graph::Graph graph;
	const auto		   mesh = veng::assets::load_obj(graph, "/no/such/path/missing.obj");
	REQUIRE_FALSE(mesh.has_value());
	CHECK(mesh.error() == veng::assets::ObjError::FileUnreadable);
}
