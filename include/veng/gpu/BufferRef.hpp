//
// The value that flows on a *storage-buffer* edge of the reactive graph (design.md §L4) —
// the SSBO/instance-buffer counterpart to UniformRef. It is a non-owning *reference* to a
// GPU buffer plus the metadata a consumer needs: the reflected binding `name` it fills, the
// per-element `stride` (used by instance-rate vertex bindings), and the live element `count`
// (used as the `instanceCount` of an instanced draw). A StorageBufferNode owns the backing
// `Buffer`, uploads the new array contents, and produces a `BufferRef`; a
// `GraphicsNode::add_storage_buffer` reads the ref, matches `name` to the shader's reflected
// descriptor (the `StructuredBuffer` variable name), and writes the buffer into a
// descriptor set. The count travels with the ref because the same edge that supplies the
// per-instance data naturally drives the instance count of the draw that reads it — a
// resize of the body array is a single producer write, not two coupled inputs.
//
// Equality is value-based and includes a `version` the producing `StorageBufferNode` bumps
// on every `produce`: the underlying VkBuffer handle is stable across re-uploads, so without
// a version a structural compare would call two distinct upload values "equal" and silently
// cache away the consuming instanced draw (same trap as UniformRef).
//

#ifndef VENG_BUFFERREF_HPP
#define VENG_BUFFERREF_HPP

#include <cstdint>
#include <string>
#include <vulkan/vulkan.hpp>

namespace veng::gpu
{
struct BufferRef
{
	vk::Buffer	   buffer{};
	vk::DeviceSize size	  = 0; // total byte size = stride * count
	std::uint32_t  stride = 0; // bytes per element
	std::uint32_t  count  = 0; // number of elements (drives instanceCount of consuming draws)
	std::string	   name;	   // the reflected descriptor binding this fills

	// Producer-bumped version (see the file header).
	std::uint64_t version = 0;

	friend bool operator==(const BufferRef&, const BufferRef&) noexcept = default;
};
} // namespace veng::gpu

#endif // VENG_BUFFERREF_HPP
