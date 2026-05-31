/**
 * @file
 * @author chris
 * @brief The value that flows on a storage-buffer edge of the reactive graph.
 *
 * `BufferRef` is the SSBO/instance-buffer counterpart to @ref veng::gpu::UniformRef. It is a non-owning
 * reference to a GPU buffer plus the metadata a consumer needs: the reflected binding `name`
 * it fills, the per-element `stride` (used by instance-rate vertex bindings), and the live
 * element `count` (used as the `instanceCount` of an instanced draw). A `StorageBufferNode`
 * owns the backing `Buffer`, uploads the new array contents, and produces a `BufferRef`; a
 * `GraphicsNode::add_storage_buffer` reads the ref, matches `name` to the shader's reflected
 * descriptor (the `StructuredBuffer` variable name), and writes the buffer into a descriptor
 * set. The count travels with the ref because the same edge that supplies the per-instance
 * data naturally drives the instance count of the draw that reads it — a resize of the body
 * array is a single producer write, not two coupled inputs.
 *
 * Equality is value-based and includes a `version` the producing `StorageBufferNode` bumps on
 * every produce: the underlying `VkBuffer` handle is stable across re-uploads, so without a
 * version a structural compare would call two distinct upload values "equal" and silently cache
 * away the consuming instanced draw (the same trap as @ref veng::gpu::UniformRef).
 *
 * @ingroup gpu_handles
 */

#ifndef VENG_BUFFERREF_HPP
#define VENG_BUFFERREF_HPP

#include <cstdint>
#include <string>
#include <vulkan/vulkan.hpp>

namespace veng::gpu
{
/**
 * @brief Non-owning reference to a GPU storage buffer and the metadata needed by a consumer.
 *
 * Flows on storage-buffer edges of the reactive graph. The producer (`StorageBufferNode`) owns
 * the backing buffer; the consumer reads this ref to bind the buffer into a descriptor set and
 * to drive the `instanceCount` of an instanced draw.
 *
 * @ingroup gpu_handles
 * @see UniformRef
 * @see MeshRef
 * @see VersionedOutput
 */
struct BufferRef
{
	vk::Buffer	   buffer{};
	vk::DeviceSize size	  = 0; ///< Allocated byte range to bind (>= one stride; never 0, see `StorageBufferNode`).
	std::uint32_t  stride = 0; ///< Bytes per element.
	std::uint32_t  count  = 0; ///< Number of elements; drives `instanceCount` of consuming draws.
	std::string	   name;	   ///< The reflected descriptor binding this fills.

	/// Pool id of the backing buffer (like @ref veng::gpu::ImageRef::pool_id). A consumer reading this ref
	/// while its producer is cached must call `pool.consume(ref)` so the N-buffered copy it
	/// points at is retained for the in-flight window — otherwise the pool can recycle or
	/// destroy it out from under the read.
	static constexpr std::uint32_t INVALID_POOL_ID = ~0U;
	std::uint32_t pool_id = INVALID_POOL_ID; ///< @ref veng::ResourcePool id, or `INVALID_POOL_ID` if unmanaged.

	/// Producer-bumped version. Incremented on every publish so two consecutive produces of the
	/// same underlying buffer handle compare unequal and the change-cutoff in `ValueData<BufferRef>`
	/// fires correctly.
	std::uint64_t version = 0;

	friend bool operator==(const BufferRef&, const BufferRef&) noexcept = default;
};
} // namespace veng::gpu

#endif // VENG_BUFFERREF_HPP
