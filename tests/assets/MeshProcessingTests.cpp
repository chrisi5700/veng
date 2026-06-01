//
// Unit tests for the internal mesh-repair pass (veng::assets::detail::repair_orientation). Drives it
// directly on synthetic cubes — the public StlMesh surface doesn't expose winding/normals, and this
// is format-agnostic anyway. Correctness is checked structurally: the repaired mesh has positive
// signed volume (outward-wound closed shell), repair is idempotent (a second pass changes nothing),
// and duplicate/degenerate triangles are dropped.
//

#include <array>
#include <assets/MeshProcessing.hpp> // internal header (src/ is on the test include path)
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

namespace
{
using veng::assets::detail::Face;

const std::vector<glm::vec3> CUBE_CORNERS{glm::vec3(0, 0, 0), glm::vec3(1, 0, 0), glm::vec3(1, 1, 0),
										  glm::vec3(0, 1, 0), glm::vec3(0, 0, 1), glm::vec3(1, 0, 1),
										  glm::vec3(1, 1, 1), glm::vec3(0, 1, 1)};

// 6 · 2 triangles of the unit cube, each forced to outward winding (normal points away from centre).
std::vector<Face> outward_cube()
{
	const std::array<std::array<std::uint32_t, 4>, 6> quads{
		{{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4}, {3, 2, 6, 7}, {0, 3, 7, 4}, {1, 2, 6, 5}}};
	const glm::vec3	  centre(0.5F);
	std::vector<Face> faces;
	for (const auto& q : quads)
	{
		for (const Face tri : {Face{q[0], q[1], q[2]}, Face{q[0], q[2], q[3]}})
		{
			const glm::vec3 p0 = CUBE_CORNERS[tri[0]];
			const glm::vec3 n  = glm::cross(CUBE_CORNERS[tri[1]] - p0, CUBE_CORNERS[tri[2]] - p0);
			Face			t  = tri;
			if (glm::dot(n, ((p0 + CUBE_CORNERS[tri[1]] + CUBE_CORNERS[tri[2]]) / 3.0F) - centre) < 0.0F)
			{
				std::swap(t[1], t[2]); // make it face outward
			}
			faces.push_back(t);
		}
	}
	return faces;
}

// 6× the enclosed volume; positive when the mesh is wound consistently outward.
double signed_volume(const std::vector<Face>& faces)
{
	double v = 0.0;
	for (const Face& f : faces)
	{
		v += glm::dot(CUBE_CORNERS[f[0]], glm::cross(CUBE_CORNERS[f[1]], CUBE_CORNERS[f[2]]));
	}
	return v;
}
} // namespace

TEST_CASE("repair_orientation leaves an already-clean outward cube alone", "[assets][mesh]")
{
	std::vector<Face> faces = outward_cube();
	const auto		  stats = veng::assets::detail::repair_orientation(CUBE_CORNERS, faces);

	CHECK(stats.faces_reoriented == 0);
	CHECK(stats.duplicate_faces_removed == 0);
	CHECK(faces.size() == 12);
	CHECK(signed_volume(faces) > 0.0);
}

TEST_CASE("repair_orientation unifies scrambled winding", "[assets][mesh]")
{
	std::vector<Face> faces = outward_cube();
	// Flip an arbitrary subset so winding is inconsistent across shared edges.
	for (std::size_t i = 0; i < faces.size(); i += 2)
	{
		std::swap(faces[i][1], faces[i][2]);
	}

	const auto stats = veng::assets::detail::repair_orientation(CUBE_CORNERS, faces);
	CHECK(stats.faces_reoriented > 0);
	CHECK(signed_volume(faces) > 0.0); // consistent AND outward

	// Idempotent: a second pass finds nothing left to fix.
	std::vector<Face> again	 = faces;
	const auto		  stats2 = veng::assets::detail::repair_orientation(CUBE_CORNERS, again);
	CHECK(stats2.faces_reoriented == 0);
	CHECK(stats2.duplicate_faces_removed == 0);
}

TEST_CASE("repair_orientation flips a fully-inverted cube outward", "[assets][mesh]")
{
	std::vector<Face> faces = outward_cube();
	for (Face& f : faces)
	{
		std::swap(f[1], f[2]); // every face wound inward
	}
	REQUIRE(signed_volume(faces) < 0.0);

	const auto stats = veng::assets::detail::repair_orientation(CUBE_CORNERS, faces);
	CHECK(stats.faces_reoriented == 12); // whole shell re-oriented
	CHECK(signed_volume(faces) > 0.0);
}

TEST_CASE("repair_orientation drops duplicate and degenerate triangles", "[assets][mesh]")
{
	std::vector<Face> faces = outward_cube();
	faces.push_back(faces[0]);									  // exact duplicate
	faces.push_back(Face{2, 2, 5});								  // degenerate (repeated index)
	faces.push_back(Face{faces[1][0], faces[1][2], faces[1][1]}); // same triangle, reversed winding

	const auto stats = veng::assets::detail::repair_orientation(CUBE_CORNERS, faces);
	CHECK(stats.duplicate_faces_removed == 3);
	CHECK(faces.size() == 12);
	CHECK(signed_volume(faces) > 0.0);
}

namespace
{
struct Grid
{
	std::vector<glm::vec3>	   positions;
	std::vector<glm::vec3>	   normals;
	std::vector<std::uint32_t> indices;
};

// An n×n quad grid on z=0 (flat, all normals +Z): dense, and collapsible to almost nothing at zero
// geometric error — ideal for asserting decimation hits its target and stays valid.
Grid make_grid(int n)
{
	Grid grid;
	for (int y = 0; y <= n; ++y)
	{
		for (int x = 0; x <= n; ++x)
		{
			grid.positions.emplace_back(static_cast<float>(x), static_cast<float>(y), 0.0F);
			grid.normals.emplace_back(0.0F, 0.0F, 1.0F);
		}
	}
	const auto vid = [n](int x, int y) { return static_cast<std::uint32_t>((y * (n + 1)) + x); };
	for (int y = 0; y < n; ++y)
	{
		for (int x = 0; x < n; ++x)
		{
			grid.indices.insert(grid.indices.end(), {vid(x, y), vid(x + 1, y), vid(x + 1, y + 1), vid(x, y),
													 vid(x + 1, y + 1), vid(x, y + 1)});
		}
	}
	return grid;
}

bool indices_in_range(const std::vector<std::uint32_t>& indices, std::uint32_t vertex_count)
{
	for (const std::uint32_t i : indices)
	{
		if (i >= vertex_count)
		{
			return false;
		}
	}
	return true;
}
} // namespace

TEST_CASE("decimate reduces triangle count and keeps the index buffer valid", "[assets][mesh]")
{
	const Grid grid = make_grid(20); // 800 triangles

	const auto result = veng::assets::detail::decimate(grid.positions, grid.normals, grid.indices,
													   veng::assets::detail::DecimateOptions{.target_ratio = 0.25F});

	CHECK(result.triangle_count < grid.indices.size() / 3); // actually fewer triangles
	CHECK(result.triangle_count <= 200);					// reached ~the 25% target...
	CHECK(result.error < 0.001F);							// ...at near-zero error (the sheet is flat)
	CHECK(result.vertex_count < grid.positions.size());		// unused vertices dropped
	CHECK(result.indices.size() == result.triangle_count * 3);
	CHECK(indices_in_range(result.indices, result.vertex_count));

	// The remap compacts a per-vertex array down to exactly the surviving vertices.
	const auto slim = veng::assets::detail::apply_vertex_remap(grid.positions, result.remap, result.vertex_count);
	CHECK(slim.size() == result.vertex_count);
}

TEST_CASE("decimate aggressive path also reduces and stays valid", "[assets][mesh]")
{
	const Grid grid = make_grid(16);

	const auto result = veng::assets::detail::decimate(
		grid.positions, grid.normals, grid.indices,
		veng::assets::detail::DecimateOptions{.target_ratio = 0.2F, .max_error = 0.1F, .aggressive = true});

	CHECK(result.triangle_count < grid.indices.size() / 3);
	CHECK(result.vertex_count < grid.positions.size());
	CHECK(indices_in_range(result.indices, result.vertex_count));
}
