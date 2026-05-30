//
// L4 upload node (design.md §L4) — the array counterpart to UniformNode. You give it a
// *value edge* (a `TypedHandle<std::vector<T>>` source/transform) and the reflected binding
// name; each time the vector changes it uploads the new bytes into its persistent storage
// buffer and publishes a `gpu::BufferRef` carrying the buffer handle, per-element stride,
// element count, and binding name. A `GraphicsNode::add_storage_buffer(handle)` consumes
// the ref, matches the name to the shader's reflected descriptor, and binds it. The count
// also drives `instanceCount` of every draw on the consuming node — this is the engine's
// "many bodies" path: one `std::vector<Body>` source -> one upload -> one instanced draw.
//
// Like UniformNode this is genuinely reactive: the vector is an *input* edge, so changing
// it re-runs the upload and re-renders every dependent GraphicsNode. The buffer is
// pool-owned (N-buffered, persistently mapped) so frame N+1 never stomps the copy frame N's
// GPU work is still reading. The buffer grows as `vector.size()` grows — a larger element
// count reallocates the pool copies (do it only at frame boundaries; the pool's retirement
// window keeps the in-flight copies alive).
//

#ifndef VENG_STORAGEBUFFERNODE_HPP
#define VENG_STORAGEBUFFERNODE_HPP

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/gpu/VersionedOutput.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <veng/resources/Buffer.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::nodes
{
class StorageBufferNode final : public gpu::GpuNode
{
	 public:
	/// Upload the contents of `value` (a `TypedHandle<std::vector<T>>`) into a storage buffer
	/// and publish a `gpu::BufferRef` on `output` (a `ValueData<gpu::BufferRef>`).
	/// `binding_name` must match the reflected descriptor name in the consuming shader (the
	/// `StructuredBuffer<T>` / `RWStructuredBuffer<T>` variable name). `T`'s memory layout must
	/// match the shader's element type (std430 for storage buffers — tightly packed for the
	/// usual POD physics-body structs). `T` is deduced from `value`.
	template <class T>
	StorageBufferNode(graph::TypedHandle<std::vector<T>> value, std::string binding_name,
					  graph::DataHandle output) noexcept
		: m_value(value.handle)
		, m_output(output)
		, m_name(std::move(binding_name))
		, m_stride(static_cast<std::uint32_t>(sizeof(T)))
		, m_read(
			  [handle = value.handle](graph::ExecContext& ctx) -> Reading
			  {
				  const auto* slot = dynamic_cast<graph::ValueData<std::vector<T>>*>(ctx.data(handle));
				  if (slot == nullptr)
				  {
					  return Reading{.resolved = false};
				  }
				  // An empty vector is a valid state (the sim has spawned no bodies yet, or all
				  // were despawned); we publish a count=0 BufferRef so consuming draws emit zero
				  // instances. v.data() may legitimately be null in that case, so the `resolved`
				  // flag — not the byte pointer — is how the cpp distinguishes the missing-slot
				  // failure path from the legitimate empty-array path.
				  const std::vector<T>& v = slot->value();
				  return Reading{.resolved = true,
								 .bytes	   = static_cast<const void*>(v.data()),
								 .count	   = static_cast<std::uint32_t>(v.size())};
			  })
	{
	}

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return {&m_value, 1}; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	 protected:
	[[nodiscard]] std::expected<bool, graph::ExecError> record(gpu::GpuExecContext& ctx) override;

	 private:
	// Erased view of the input vector: the captured lambda resolves the typed slot and hands
	// back the bytes + count. `resolved` distinguishes a missing/wrong-type input slot
	// (MISSING_INPUT) from a legitimately empty vector (count=0, bytes possibly null).
	struct Reading
	{
		bool		  resolved = false;
		const void*	  bytes	   = nullptr;
		std::uint32_t count	   = 0;
	};

	graph::DataHandle							m_value; // the reactive vector<T> edge
	graph::DataHandle							m_output;
	std::string									m_name;
	std::uint32_t								m_stride; // sizeof(T) — bytes per element
	std::function<Reading(graph::ExecContext&)> m_read;
	bool										m_declared	= false; // m_buffer_id declared in the pool?
	BufferId									m_buffer_id = 0;	 // pool-owned, N-buffered storage buffer
	gpu::VersionedOutput						m_versioned; // owns the per-upload version bump for the BufferRef
};
} // namespace veng::nodes

#endif // VENG_STORAGEBUFFERNODE_HPP
