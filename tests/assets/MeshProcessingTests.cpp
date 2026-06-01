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
