//
// Geometry primitives for the example library: a single `Vertex` layout (position, normal,
// color) and a `Mesh` aggregate (vertices + 32-bit indices), plus factory functions for the
// shapes the examples need. The layout matches `shaders/demo/lit_instanced.vert.slang`, so a
// mesh fed through a `MeshNode` lights correctly under `lit_directional.frag.slang`. Heavier
// shapes can land here later (capsule, cylinder, plane, ...) without touching call sites.
//

#ifndef VENG_EXAMPLE_GEOMETRY_HPP
#define VENG_EXAMPLE_GEOMETRY_HPP

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

namespace example
{
struct Vertex
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color;
};

struct Mesh
{
	std::vector<Vertex>		   vertices;
	std::vector<std::uint32_t> indices;
};

/// Unit cube centered at the origin (edge length 1). 24 verts / 36 indices, per-face flat
/// normals (4 verts per face) so the directional light renders crisp facets. Pass a single
/// color for a uniform look or use `make_cube_faceted()` for the rainbow per-face palette
/// the rigid-body demo uses.
Mesh make_cube(glm::vec3 color = {1.0F, 1.0F, 1.0F});

/// Cube with the classic per-face colour palette (-z red, +z green, -x blue, +x yellow,
/// -y magenta, +y cyan). Useful when you want to see body orientations at a glance.
Mesh make_cube_faceted();

/// UV sphere of `radius` centered at the origin. `latitude_segments` rings between the poles
/// (>=2), `longitude_segments` slices around the equator (>=3). Normals are the position
/// vector — smooth shading. 16x32 is a good default.
Mesh make_sphere(float radius = 0.5F, std::uint32_t latitude_segments = 16, std::uint32_t longitude_segments = 32,
				 glm::vec3 color = {1.0F, 1.0F, 1.0F});

/// Torus centered at the origin in the xz plane (major axis is y). `major_radius` is the
/// ring radius, `minor_radius` the tube radius. Segments parameter the same as the sphere.
/// Smooth normals.
Mesh make_torus(float major_radius = 0.5F, float minor_radius = 0.15F, std::uint32_t major_segments = 32,
				std::uint32_t minor_segments = 16, glm::vec3 color = {1.0F, 1.0F, 1.0F});
} // namespace example

#endif // VENG_EXAMPLE_GEOMETRY_HPP
