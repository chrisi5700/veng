//
// Unit tests for veng::assets::load_ply. An inline ASCII PLY cube fixture (no fetch, no GPU)
// exercises tinyply parse → the shared pipeline.
//

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <veng/assets/PlyLoader.hpp>
#include <veng/rendergraph/Graph.hpp>

namespace
{
std::filesystem::path write_cube_ply()
{
	const auto	  path = std::filesystem::temp_directory_path() / "veng_cube.ply";
	std::ofstream out(path);
	out << "ply\nformat ascii 1.0\n"
		   "element vertex 8\nproperty float x\nproperty float y\nproperty float z\n"
		   "element face 12\nproperty list uchar int vertex_indices\nend_header\n";
	out << "0 0 0\n1 0 0\n1 1 0\n0 1 0\n0 0 1\n1 0 1\n1 1 1\n0 1 1\n";
	// 12 triangles (0-indexed), two per cube face.
	out << "3 0 1 2\n3 0 2 3\n3 4 5 6\n3 4 6 7\n3 0 1 5\n3 0 5 4\n"
		   "3 3 2 6\n3 3 6 7\n3 0 3 7\n3 0 7 4\n3 1 2 6\n3 1 6 5\n";
	return path;
}
} // namespace

TEST_CASE("load_ply parses an ASCII cube", "[assets][ply]")
{
	const auto path = write_cube_ply();

	veng::graph::Graph graph;
	const auto		   mesh = veng::assets::load_ply(graph, path.string());
	REQUIRE(mesh.has_value());
	CHECK(mesh->index_count == 36);
	CHECK(mesh->vertex_count == 8); // indexed, no attribute seams to split
	CHECK(mesh->bounds_max.z == Catch::Approx(1.0F));

	std::filesystem::remove(path);
}

TEST_CASE("load_ply_lods builds a mesh edge per level", "[assets][ply]")
{
	const auto path = write_cube_ply();

	veng::graph::Graph							graph;
	const std::array<veng::assets::LodLevel, 2> levels{{{.target_ratio = 1.0F}, {.target_ratio = 0.5F}}};
	const auto									set = veng::assets::load_ply_lods(graph, path.string(), levels);
	REQUIRE(set.has_value());
	CHECK(set->meshes.size() == 2);

	std::filesystem::remove(path);
}

TEST_CASE("load_ply reports a clear error for a missing file", "[assets][ply]")
{
	veng::graph::Graph graph;
	const auto		   mesh = veng::assets::load_ply(graph, "/no/such/path/missing.ply");
	REQUIRE_FALSE(mesh.has_value());
	CHECK(mesh.error() == veng::assets::PlyError::FileUnreadable);
}
