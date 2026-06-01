//
// Unit tests for veng::assets::load_3mf. Writes a cube to a temp .3mf with lib3mf, then loads it
// back — self-contained, no fetched asset. SKIPs if lib3mf can't be loaded at runtime. No GPU
// (the loader only adds CPU nodes to a Graph).
//

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <lib3mf_implicit.hpp>
#include <string>
#include <vector>
#include <veng/assets/ThreemfLoader.hpp>
#include <veng/rendergraph/Graph.hpp>

namespace
{
// Write a unit-cube .3mf via lib3mf; returns false (test SKIPs) if lib3mf is unavailable.
bool write_cube_3mf(const std::filesystem::path& path)
{
	try
	{
		Lib3MF::PWrapper	wrapper = Lib3MF::CWrapper::loadLibrary();
		Lib3MF::PModel		model	= wrapper->CreateModel();
		Lib3MF::PMeshObject mesh	= model->AddMeshObject();

		const std::array<std::array<float, 3>, 8> corners{
			{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}};
		std::vector<Lib3MF::sPosition> verts;
		for (const auto& c : corners)
		{
			verts.push_back(Lib3MF::sPosition{.m_Coordinates = {c[0], c[1], c[2]}});
		}
		const std::array<std::array<std::uint32_t, 3>, 12> faces{{{0, 2, 1},
																  {0, 3, 2},
																  {4, 5, 6},
																  {4, 6, 7},
																  {0, 1, 5},
																  {0, 5, 4},
																  {3, 7, 6},
																  {3, 6, 2},
																  {0, 4, 7},
																  {0, 7, 3},
																  {1, 2, 6},
																  {1, 6, 5}}};
		std::vector<Lib3MF::sTriangle>					   tris;
		for (const auto& f : faces)
		{
			tris.push_back(Lib3MF::sTriangle{.m_Indices = {f[0], f[1], f[2]}});
		}
		mesh->SetGeometry(verts, tris);

		Lib3MF::PWriter writer = model->QueryWriter("3mf");
		writer->WriteToFile(path.string());
		return true;
	}
	catch (...)
	{
		return false;
	}
}
} // namespace

TEST_CASE("load_3mf parses a cube written by lib3mf", "[assets][3mf]")
{
	const auto path = std::filesystem::temp_directory_path() / "veng_cube.3mf";
	if (!write_cube_3mf(path))
	{
		SKIP("lib3mf unavailable at runtime");
	}

	veng::graph::Graph graph;
	const auto		   mesh = veng::assets::load_3mf(graph, path.string());
	REQUIRE(mesh.has_value());
	CHECK(mesh->index_count == 36);	 // 12 triangles
	CHECK(mesh->vertex_count == 24); // soup → welded → crease-split at the cube's hard edges
	CHECK(mesh->bounds_max.x == Catch::Approx(1.0F));

	std::filesystem::remove(path);
}

TEST_CASE("load_3mf reports a clear error for a missing file", "[assets][3mf]")
{
	veng::graph::Graph graph;
	const auto		   mesh = veng::assets::load_3mf(graph, "/no/such/path/missing.3mf");
	REQUIRE_FALSE(mesh.has_value());
	CHECK(mesh.error() == veng::assets::ThreemfError::FileUnreadable);
}
