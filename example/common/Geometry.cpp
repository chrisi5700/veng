//
// See Geometry.hpp.
//

#include "Geometry.hpp"

#include <array>
#include <cmath>
#include <glm/gtc/constants.hpp>

namespace example
{
namespace
{
Mesh build_cube(const std::array<glm::vec3, 6>& face_colors)
{
	constexpr std::array<glm::vec3, 8> corners{glm::vec3{-0.5F, -0.5F, -0.5F}, glm::vec3{0.5F, -0.5F, -0.5F},
											   glm::vec3{0.5F, 0.5F, -0.5F},   glm::vec3{-0.5F, 0.5F, -0.5F},
											   glm::vec3{-0.5F, -0.5F, 0.5F},  glm::vec3{0.5F, -0.5F, 0.5F},
											   glm::vec3{0.5F, 0.5F, 0.5F},	   glm::vec3{-0.5F, 0.5F, 0.5F}};

	struct Face
	{
		std::array<std::uint32_t, 4> corner;
		glm::vec3					 normal;
	};
	constexpr std::array<Face, 6> faces{Face{{0, 1, 2, 3}, {0.0F, 0.0F, -1.0F}},	// -z
										Face{{5, 4, 7, 6}, {0.0F, 0.0F, 1.0F}},		// +z
										Face{{4, 0, 3, 7}, {-1.0F, 0.0F, 0.0F}},	// -x
										Face{{1, 5, 6, 2}, {1.0F, 0.0F, 0.0F}},		// +x
										Face{{4, 5, 1, 0}, {0.0F, -1.0F, 0.0F}},	// -y
										Face{{3, 2, 6, 7}, {0.0F, 1.0F, 0.0F}}};	// +y

	Mesh mesh;
	mesh.vertices.reserve(24);
	mesh.indices.reserve(36);
	for (std::size_t f = 0; f < faces.size(); ++f)
	{
		const Face& face = faces[f];
		const auto	base = static_cast<std::uint32_t>(mesh.vertices.size());
		for (const std::uint32_t corner : face.corner)
		{
			mesh.vertices.push_back(
				Vertex{.position = corners[corner], .normal = face.normal, .color = face_colors[f]});
		}
		for (const std::uint32_t offset : {0U, 1U, 2U, 0U, 2U, 3U})
		{
			mesh.indices.push_back(base + offset);
		}
	}
	return mesh;
}
} // namespace

Mesh make_cube(glm::vec3 color)
{
	return build_cube({color, color, color, color, color, color});
}

Mesh make_cube_faceted()
{
	return build_cube({glm::vec3{1.0F, 0.25F, 0.25F}, glm::vec3{0.25F, 1.0F, 0.35F}, glm::vec3{0.30F, 0.45F, 1.0F},
					   glm::vec3{1.0F, 0.85F, 0.25F}, glm::vec3{1.0F, 0.35F, 1.0F}, glm::vec3{0.30F, 1.0F, 1.0F}});
}

Mesh make_sphere(float radius, std::uint32_t latitude_segments, std::uint32_t longitude_segments, glm::vec3 color)
{
	Mesh mesh;
	mesh.vertices.reserve(static_cast<std::size_t>((latitude_segments + 1) * (longitude_segments + 1)));
	mesh.indices.reserve(static_cast<std::size_t>(latitude_segments * longitude_segments * 6));

	const float pi = glm::pi<float>();
	for (std::uint32_t lat = 0; lat <= latitude_segments; ++lat)
	{
		// theta in [0, pi]: 0 at north pole, pi at south pole.
		const float theta = pi * static_cast<float>(lat) / static_cast<float>(latitude_segments);
		const float sin_t = std::sin(theta);
		const float cos_t = std::cos(theta);
		for (std::uint32_t lon = 0; lon <= longitude_segments; ++lon)
		{
			// phi in [0, 2pi] around the equator.
			const float		phi	   = 2.0F * pi * static_cast<float>(lon) / static_cast<float>(longitude_segments);
			const glm::vec3 normal = {sin_t * std::cos(phi), cos_t, sin_t * std::sin(phi)};
			mesh.vertices.push_back(Vertex{.position = normal * radius, .normal = normal, .color = color});
		}
	}

	const std::uint32_t ring = longitude_segments + 1;
	for (std::uint32_t lat = 0; lat < latitude_segments; ++lat)
	{
		for (std::uint32_t lon = 0; lon < longitude_segments; ++lon)
		{
			const std::uint32_t a = lat * ring + lon;
			const std::uint32_t b = a + ring;
			mesh.indices.insert(mesh.indices.end(), {a, b, a + 1, a + 1, b, b + 1});
		}
	}
	return mesh;
}

Mesh make_torus(float major_radius, float minor_radius, std::uint32_t major_segments, std::uint32_t minor_segments,
				glm::vec3 color)
{
	Mesh mesh;
	mesh.vertices.reserve(static_cast<std::size_t>((major_segments + 1) * (minor_segments + 1)));
	mesh.indices.reserve(static_cast<std::size_t>(major_segments * minor_segments * 6));

	const float pi = glm::pi<float>();
	for (std::uint32_t i = 0; i <= major_segments; ++i)
	{
		const float u	  = 2.0F * pi * static_cast<float>(i) / static_cast<float>(major_segments);
		const float cos_u = std::cos(u);
		const float sin_u = std::sin(u);
		for (std::uint32_t j = 0; j <= minor_segments; ++j)
		{
			const float v	  = 2.0F * pi * static_cast<float>(j) / static_cast<float>(minor_segments);
			const float cos_v = std::cos(v);
			const float sin_v = std::sin(v);
			// Tube center at (major_radius * cos_u, 0, major_radius * sin_u); tube point offset by
			// minor_radius along (cos_v in the ring plane, sin_v in y). Normal is the offset
			// direction itself — smooth shading along the tube.
			const glm::vec3 normal{cos_v * cos_u, sin_v, cos_v * sin_u};
			const glm::vec3 position{(major_radius + minor_radius * cos_v) * cos_u, minor_radius * sin_v,
									 (major_radius + minor_radius * cos_v) * sin_u};
			mesh.vertices.push_back(Vertex{.position = position, .normal = normal, .color = color});
		}
	}

	const std::uint32_t ring = minor_segments + 1;
	for (std::uint32_t i = 0; i < major_segments; ++i)
	{
		for (std::uint32_t j = 0; j < minor_segments; ++j)
		{
			const std::uint32_t a = i * ring + j;
			const std::uint32_t b = a + ring;
			mesh.indices.insert(mesh.indices.end(), {a, b, a + 1, a + 1, b, b + 1});
		}
	}
	return mesh;
}
} // namespace example
