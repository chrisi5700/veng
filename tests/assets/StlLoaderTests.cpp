//
// Unit tests for veng::assets::load_stl. STL carries only triangle positions, so the loader's job is
// synthesising the rest: welding facet corners, splitting/smoothing normals by the crease angle, and
// generating UVs + tangents. These tests drive that through the public API on tiny in-memory ASCII
// and binary fixtures (no GPU needed — load_stl only adds CPU nodes to a Graph), checking the
// observable contract: triangle/vertex counts under crease smoothing, ASCII↔binary parity, bounds,
// and the error paths. Geometry is generated procedurally so no asset files are required.
//

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <veng/assets/StlLoader.hpp>
#include <veng/rendergraph/Graph.hpp>

namespace
{
using Tri = std::array<glm::vec3, 3>;

// The 12 triangles (2 per face) of an axis-aligned box. Winding is consistent-enough for the tests,
// which only depend on coplanarity grouping, not on outward orientation.
std::vector<Tri> box_triangles(glm::vec3 lo, glm::vec3 hi)
{
	const std::array<glm::vec3, 8> c{glm::vec3(lo.x, lo.y, lo.z), glm::vec3(hi.x, lo.y, lo.z),
									 glm::vec3(hi.x, hi.y, lo.z), glm::vec3(lo.x, hi.y, lo.z),
									 glm::vec3(lo.x, lo.y, hi.z), glm::vec3(hi.x, lo.y, hi.z),
									 glm::vec3(hi.x, hi.y, hi.z), glm::vec3(lo.x, hi.y, hi.z)};
	// Each face as a quad of corner indices; split into two triangles.
	const std::array<std::array<int, 4>, 6> faces{{{0, 1, 2, 3},   // z = lo
												   {4, 5, 6, 7},   // z = hi
												   {0, 1, 5, 4},   // y = lo
												   {3, 2, 6, 7},   // y = hi
												   {0, 3, 7, 4},   // x = lo
												   {1, 2, 6, 5}}}; // x = hi
	std::vector<Tri>						tris;
	for (const auto& f : faces)
	{
		tris.push_back(Tri{c[f[0]], c[f[1]], c[f[2]]});
		tris.push_back(Tri{c[f[0]], c[f[2]], c[f[3]]});
	}
	return tris;
}

std::filesystem::path temp_path(const std::string& name)
{
	return std::filesystem::temp_directory_path() / name;
}

void write_ascii(const std::filesystem::path& path, const std::vector<Tri>& tris)
{
	std::ofstream out(path);
	out << "solid test\n";
	for (const Tri& t : tris)
	{
		out << "facet normal 0 0 0\n outer loop\n";
		for (const glm::vec3& v : t)
		{
			out << "  vertex " << v.x << ' ' << v.y << ' ' << v.z << '\n';
		}
		out << " endloop\nendfacet\n";
	}
	out << "endsolid test\n";
}

void write_binary(const std::filesystem::path& path, const std::vector<Tri>& tris)
{
	std::ofstream		 out(path, std::ios::binary);
	std::array<char, 80> header{};
	out.write(header.data(), header.size());
	const auto count = static_cast<std::uint32_t>(tris.size());
	out.write(static_cast<const char*>(static_cast<const void*>(&count)), sizeof(count));
	for (const Tri& t : tris)
	{
		const std::array<float, 3> n{0.0F, 0.0F, 0.0F};
		out.write(static_cast<const char*>(static_cast<const void*>(n.data())), sizeof(n));
		for (const glm::vec3& v : t)
		{
			const std::array<float, 3> xyz{v.x, v.y, v.z};
			out.write(static_cast<const char*>(static_cast<const void*>(xyz.data())), sizeof(xyz));
		}
		const std::uint16_t attr = 0;
		out.write(static_cast<const char*>(static_cast<const void*>(&attr)), sizeof(attr));
	}
}
} // namespace

TEST_CASE("load_stl welds an ASCII cube and splits its hard edges", "[assets][stl]")
{
	const auto path = temp_path("veng_cube_ascii.stl");
	write_ascii(path, box_triangles(glm::vec3(0.0F), glm::vec3(1.0F)));

	veng::graph::Graph graph;
	const auto		   mesh = veng::assets::load_stl(graph, path.string());
	REQUIRE(mesh.has_value());

	// 12 triangles → 36 indices. Each cube corner is shared by 3 faces meeting at 90° (> 40° crease),
	// so it splits into 3 vertices: 8 corners × 3 = 24 welded/split vertices.
	CHECK(mesh->index_count == 36);
	CHECK(mesh->vertex_count == 24);

	CHECK(mesh->bounds_min.x == Catch::Approx(0.0F));
	CHECK(mesh->bounds_max.x == Catch::Approx(1.0F));
	CHECK(mesh->center().y == Catch::Approx(0.5F));
	CHECK(mesh->radius() == Catch::Approx(glm::length(glm::vec3(1.0F)) * 0.5F));

	std::filesystem::remove(path);
}

TEST_CASE("load_stl parses binary identically to ASCII", "[assets][stl]")
{
	const auto tris	  = box_triangles(glm::vec3(-1.0F), glm::vec3(2.0F));
	const auto a_path = temp_path("veng_cube_ascii2.stl");
	const auto b_path = temp_path("veng_cube_bin.stl");
	write_ascii(a_path, tris);
	write_binary(b_path, tris);

	veng::graph::Graph ga;
	veng::graph::Graph gb;
	const auto		   a = veng::assets::load_stl(ga, a_path.string());
	const auto		   b = veng::assets::load_stl(gb, b_path.string());
	REQUIRE(a.has_value());
	REQUIRE(b.has_value());

	CHECK(a->vertex_count == b->vertex_count);
	CHECK(a->index_count == b->index_count);
	CHECK(b->bounds_min.x == Catch::Approx(-1.0F));
	CHECK(b->bounds_max.z == Catch::Approx(2.0F));

	std::filesystem::remove(a_path);
	std::filesystem::remove(b_path);
}

TEST_CASE("load_stl merges coplanar triangles into shared vertices", "[assets][stl]")
{
	// Two coplanar triangles forming a quad in z = 0: four corners, all within the crease cone, so
	// every corner welds to a single smoothed vertex.
	const std::vector<Tri> quad{Tri{glm::vec3(0, 0, 0), glm::vec3(1, 0, 0), glm::vec3(1, 1, 0)},
								Tri{glm::vec3(0, 0, 0), glm::vec3(1, 1, 0), glm::vec3(0, 1, 0)}};
	const auto			   path = temp_path("veng_quad.stl");
	write_ascii(path, quad);

	veng::graph::Graph graph;
	const auto		   mesh = veng::assets::load_stl(graph, path.string());
	REQUIRE(mesh.has_value());
	CHECK(mesh->index_count == 6);
	CHECK(mesh->vertex_count == 4);

	std::filesystem::remove(path);
}

TEST_CASE("load_stl crease angle controls smoothing", "[assets][stl]")
{
	const auto path = temp_path("veng_cube_smooth.stl");
	write_ascii(path, box_triangles(glm::vec3(0.0F), glm::vec3(1.0F)));

	// A crease angle wide enough to span any dihedral merges all faces at each corner, welding the
	// cube down to its 8 geometric corners.
	veng::graph::Graph graph;
	const auto		   mesh =
		veng::assets::load_stl(graph, path.string(), veng::assets::StlOptions{.crease_angle_deg = 179.0F});
	REQUIRE(mesh.has_value());
	CHECK(mesh->vertex_count == 8);
	CHECK(mesh->index_count == 36);

	std::filesystem::remove(path);
}

TEST_CASE("load_stl reports a clear error for a missing file", "[assets][stl]")
{
	veng::graph::Graph graph;
	const auto		   mesh = veng::assets::load_stl(graph, "/no/such/path/missing.stl");
	REQUIRE_FALSE(mesh.has_value());
	CHECK(mesh.error() == veng::assets::StlError::FileUnreadable);
}

TEST_CASE("load_stl rejects bytes with no triangles", "[assets][stl]")
{
	const auto path = temp_path("veng_garbage.stl");
	std::ofstream(path) << "solid empty\nthis text has no vertex records\nendsolid empty\n";

	veng::graph::Graph graph;
	const auto		   mesh = veng::assets::load_stl(graph, path.string());
	REQUIRE_FALSE(mesh.has_value());
	CHECK(mesh.error() == veng::assets::StlError::ParseFailed);

	std::filesystem::remove(path);
}
