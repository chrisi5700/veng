/**
 * @file
 * @author chris
 * @brief The value that flows on a uniform edge of the reactive graph.
 *
 * `UniformRef` is the descriptor-set counterpart to @ref veng::gpu::MeshRef. It is a non-owning reference
 * to a GPU uniform buffer plus the reflected binding name it fills. A `UniformNode` owns the
 * backing `Buffer`, uploads the value, and produces a `UniformRef`; a
 * `GraphicsNode::add_uniform` reads it, matches `name` to the shader's reflected descriptor
 * (`DescriptorInfo.name`), and writes the buffer into a descriptor set. This is the user's
 * `UniformNode{value, "name"} -> GraphicsNode.add_uniform(...)`.
 *
 * Equality is value-based, including a `version` the producing `UniformNode` bumps on every
 * publish — critical because the buffer handle is stable across re-uploads (contents change in
 * place), so without a version a structural comparison would call two distinct upload values
 * "equal" and silently cache away the consuming draw.
 *
 * @ingroup gpu_handles
 */

#ifndef VENG_UNIFORMREF_HPP
#define VENG_UNIFORMREF_HPP

#include <cstdint>
#include <string>
#include <veng/rhi/Handles.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::gpu
{
/**
 * @brief Non-owning reference to a GPU uniform buffer and the descriptor binding it fills.
 *
 * Flows on uniform edges of the reactive graph. The producer (`UniformNode`) owns the backing
 * buffer; consumers such as `GraphicsNode::add_uniform` read this ref to write the buffer into
 * a descriptor set, matched by reflected binding `name`.
 *
 * @ingroup gpu_handles
 * @see MeshRef
 * @see BufferRef
 * @see ImageRef
 * @see VersionedOutput
 */
struct UniformRef
{
	rhi::BufferHandle buffer{}; ///< Opaque handle to the uniform buffer (resolve via rhi::Device).
	vk::DeviceSize	  size = 0; ///< Byte size of the buffer.
	std::string		  name;		///< The reflected descriptor binding this fills (`DescriptorInfo.name`).

	/// Producer-bumped version. Incremented on every publish so two consecutive produces of the
	/// same underlying buffer handle (whose contents changed in place) compare unequal and the
	/// change-cutoff in `ValueData<UniformRef>` fires correctly.
	std::uint64_t version = 0;

	friend bool operator==(const UniformRef&, const UniformRef&) noexcept = default;
};
} // namespace veng::gpu

#endif // VENG_UNIFORMREF_HPP
