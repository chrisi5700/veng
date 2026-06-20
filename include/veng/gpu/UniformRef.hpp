/**
 * @file
 * @author chris
 * @brief The value that flows on a uniform edge of the reactive graph.
 *
 * `UniformRef` is the descriptor-set counterpart to @ref veng::gpu::MeshRef. It is a non-owning reference
 * to a GPU uniform buffer plus the reflected binding name it fills. A `UniformNode` declares an
 * N-buffered, pool-owned uniform buffer, writes this frame's physical copy, and produces a
 * `UniformRef`; a `GraphicsNode::add_uniform` reads it, matches `name` to the shader's reflected
 * descriptor (`DescriptorInfo.name`), and writes the buffer into a descriptor set. This is the
 * user's `UniformNode{value, "name"} -> GraphicsNode.add_uniform(...)`.
 *
 * Because the backing buffer is pool-owned and N-buffered (see @ref veng::ResourcePool), the ref
 * carries the producer's `pool_id`: a consumer reading this ref while its producer is cached must
 * call `pool.consume(ref)` so the copy it points at is retained for the in-flight window —
 * otherwise the pool can recycle that copy out from under the descriptor set still referencing it.
 *
 * Equality is value-based, including a `version` the producing `UniformNode` bumps on every
 * publish — critical because a re-upload may reuse the same physical copy (same buffer handle, new
 * contents), so without a version a structural comparison would call two distinct upload values
 * "equal" and silently cache away the consuming draw.
 *
 * @ingroup gpu_handles
 */

#ifndef VENG_UNIFORMREF_HPP
#define VENG_UNIFORMREF_HPP

#include <cstdint>
#include <string>
#include <veng/rhi/Handles.hpp>

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
	std::uint64_t	  size = 0; ///< Byte size of the buffer.
	std::string		  name;		///< The reflected descriptor binding this fills (`DescriptorInfo.name`).

	/// Pool id of the backing buffer (like @ref veng::gpu::BufferRef::pool_id). A consumer reading this ref
	/// while its producer is cached must call `pool.consume(ref)` so the N-buffered copy it points
	/// at is retained for the in-flight window — otherwise the pool can recycle it out from under
	/// the descriptor set still referencing it.
	static constexpr std::uint32_t INVALID_POOL_ID = ~0U;
	std::uint32_t pool_id = INVALID_POOL_ID; ///< @ref veng::ResourcePool id, or `INVALID_POOL_ID` if unmanaged.

	/// Producer-bumped version. Incremented on every publish so two consecutive produces (which may
	/// reuse the same physical copy: same buffer handle, new contents) compare unequal and the
	/// change-cutoff in `ValueData<UniformRef>` fires correctly.
	std::uint64_t version = 0;

	friend bool operator==(const UniformRef&, const UniformRef&) noexcept = default;
};
} // namespace veng::gpu

#endif // VENG_UNIFORMREF_HPP
