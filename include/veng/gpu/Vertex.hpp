//
// Created by chris on 5/30/26.
//
// The engine's standard PBR vertex layout — the contract between the vertex shaders that do
// lit/textured shading (shaders/passes/pbr.vert) and whoever uploads the geometry (the glTF
// loader, the example geometry helpers, tests). A plain header with no symbols, so any target
// can include it without a link dependency. Tightly packed to 48 bytes; the consuming
// GraphicsNode/PbrPass asserts this stride against the shader's reflected vertex binding
// (gpu::MeshRef::vertex_stride).
//

#ifndef VENG_VERTEX_HPP
#define VENG_VERTEX_HPP

#include <glm/glm.hpp>

namespace veng::gpu
{
struct PbrVertex
{
	glm::vec3 position; // 0
	glm::vec3 normal;	// 12
	glm::vec4 tangent;	// 24 — xyz tangent, w = handedness (+/-1) for the bitangent
	glm::vec2 uv;		// 40
};
static_assert(sizeof(PbrVertex) == 48, "PbrVertex must stay tightly packed to match the shader's reflected stride");
} // namespace veng::gpu

#endif // VENG_VERTEX_HPP
