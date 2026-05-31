/**
 * @file
 * @author chris
 * @brief The value that flows on a mesh edge of the reactive graph.
 *
 * `MeshRef` is the geometry counterpart to @ref veng::gpu::ImageRef. It is a non-owning reference to GPU
 * vertex (and optional index) buffers, never the vertex data itself. A `MeshNode` uploads the
 * geometry once, owns the backing buffers, and produces a `MeshRef` to them; a `GraphicsNode`
 * reads the ref and binds and draws it. This is what turns the user's
 * `IndexedVertexNode{verts, indices} -> GraphicsNode.add_mesh(...)` into real buffer-backed
 * geometry (versus the `SV_VertexID` path).
 *
 * Equality is value-based, including a `version` the producing `MeshNode` bumps on every
 * publish — so a re-upload (where the underlying buffer handle is reused but contents changed)
 * compares unequal and re-dirties the consuming draw. `MeshNode` normally uploads once cold
 * and caches forever after; the version is the safety net for the rare re-upload case.
 *
 * @ingroup gpu_handles
 */

#ifndef VENG_MESHREF_HPP
#define VENG_MESHREF_HPP

#include <cstdint>
#include <vulkan/vulkan.hpp>

namespace veng::gpu
{
/**
 * @brief Non-owning reference to GPU vertex and (optionally) index buffers.
 *
 * Flows on mesh edges of the reactive graph. The producer (`MeshNode`) owns the backing
 * buffers; consumers such as `GraphicsNode` read this ref to bind and draw the geometry.
 *
 * @ingroup gpu_handles
 * @see ImageRef
 * @see BufferRef
 * @see UniformRef
 * @see VersionedOutput
 */
struct MeshRef
{
	vk::Buffer vertex_buffer{}; ///< Vertex buffer handle.
	vk::Buffer index_buffer{};	///< Index buffer handle; null means draw non-indexed using `vertex_count`.

	std::uint32_t vertex_count = 0;						 ///< Vertices in the vertex buffer (non-indexed draw count).
	std::uint32_t index_count  = 0;						 ///< Indices in the index buffer (indexed draw count).
	vk::IndexType index_type   = vk::IndexType::eUint32; ///< Index element type.

	/// Bytes per vertex (the producing node's `sizeof(Vertex)`). The consuming `GraphicsNode`
	/// asserts this against the stride its bound vertex shader reflects, turning a layout
	/// mismatch (e.g. 36-byte vertices fed to a 56-byte-input shader) into a typed node failure
	/// instead of garbage strided through the buffer. 0 means unknown/unchecked (skips the check).
	std::uint32_t vertex_stride = 0;

	/// Producer-bumped version. Incremented on every publish so a re-upload of the same
	/// underlying buffer handle compares unequal and the change-cutoff fires correctly.
	std::uint64_t version = 0;

	friend bool operator==(const MeshRef&, const MeshRef&) noexcept = default;
};
} // namespace veng::gpu

#endif // VENG_MESHREF_HPP
