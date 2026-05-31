/**
 * @file
 * @author chris
 * @brief The engine's standard PBR vertex layout.
 *
 * Defines the contract between the vertex shaders that do lit and textured shading
 * (`shaders/passes/pbr.vert`) and whoever uploads the geometry (the glTF loader, the example
 * geometry helpers, tests). A plain header with no non-inline symbols, so any target can
 * include it without a link dependency. Tightly packed to 48 bytes; the consuming
 * `GraphicsNode` / `PbrPass` asserts this stride against the shader's reflected vertex binding
 * via @ref veng::gpu::MeshRef::vertex_stride.
 *
 * @ingroup gpu_handles
 */

#ifndef VENG_VERTEX_HPP
#define VENG_VERTEX_HPP

#include <glm/glm.hpp>

namespace veng::gpu
{
/**
 * @brief Standard PBR vertex: position, normal, tangent (with handedness), and UV.
 *
 * Tightly packed to 48 bytes. The `static_assert` below enforces this at compile time so a
 * struct-layout change is caught immediately rather than silently mismatching the shader's
 * reflected stride.
 *
 * @ingroup gpu_handles
 * @see MeshRef::vertex_stride
 */
struct PbrVertex
{
	glm::vec3 position; ///< Object-space position (offset 0).
	glm::vec3 normal;	///< Object-space normal (offset 12).
	glm::vec4 tangent;	///< `xyz` = tangent direction, `w` = bitangent handedness (+1 or -1) (offset 24).
	glm::vec2 uv;		///< Texture coordinates (offset 40).
};
static_assert(sizeof(PbrVertex) == 48, "PbrVertex must stay tightly packed to match the shader's reflected stride");
} // namespace veng::gpu

#endif // VENG_VERTEX_HPP
