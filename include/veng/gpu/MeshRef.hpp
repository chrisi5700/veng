//
// Created by chris on 5/26/26.
//
// The value that flows on a *mesh* edge of the reactive graph (design.md §L4) — the
// geometry counterpart to ImageRef. It is a non-owning *reference* to GPU vertex (and
// optional index) buffers, never the vertex data itself. A MeshNode uploads the geometry
// once, owns the backing `Buffer`s, and produces a `MeshRef` to them; a GraphicsNode reads
// the ref and binds + draws it. This is what turns the user's `IndexedVertexNode{verts,
// indices} -> GraphicsNode.add_mesh(...)` into real buffer-backed geometry (vs the
// SV_VertexID path).
//
// Like ImageRef, deliberately NOT equality-comparable: `ValueData<T>` then treats every
// re-produce as "changed" (Data.hpp). A MeshNode normally uploads once and is cached
// forever after, so this only matters on the rare re-upload — where the buffer contents
// genuinely changed and must propagate to the draw.
//

#ifndef VENG_MESHREF_HPP
#define VENG_MESHREF_HPP

#include <cstdint>
#include <vulkan/vulkan.hpp>

namespace veng::gpu
{
struct MeshRef
{
	vk::Buffer vertex_buffer{};
	vk::Buffer index_buffer{}; // null => draw non-indexed with `vertex_count`

	std::uint32_t vertex_count = 0; // vertices in the vertex buffer (non-indexed draw count)
	std::uint32_t index_count  = 0; // indices in the index buffer (indexed draw count)
	vk::IndexType index_type   = vk::IndexType::eUint32;

	// No operator== on purpose — see the file header.
};
} // namespace veng::gpu

#endif // VENG_MESHREF_HPP
