//
// Created by chris on 5/30/26.
//
// The producer-side "stamp a fresh version, then produce" step every upload/producer node
// repeats (design.md §L4). The reference types that flow on graph edges (MeshRef, UniformRef,
// BufferRef, ImageRef) carry a `version` the producer bumps on every publish so two
// structurally identical refs from consecutive runs still compare unequal — which is what
// makes `ValueData<Ref>` change-cutoff fire when a stable handle (a reused buffer, a recycled
// swapchain image) is re-filled with new contents (see ImageRef.hpp).
//
// Before this, MeshNode/UniformNode/StorageBufferNode/DynamicMeshNode/BlitNode/GraphicsNode each
// carried their own `std::uint64_t m_version` and hand-wrote `++m_version; out->produce({…,
// .version = m_version})`. VersionedOutput owns that counter and the publish step in one place,
// making the "a producer always publishes a fresh ref" contract explicit and unforgettable.
//

#ifndef VENG_VERSIONEDOUTPUT_HPP
#define VENG_VERSIONEDOUTPUT_HPP

#include <cstdint>
#include <utility>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/nodes/Node.hpp>

namespace veng::gpu
{
/// Bundles a monotonic version counter with the stamp-and-produce a producer node does each
/// time it publishes its output ref. `RefT` must have a `std::uint64_t version` member.
class VersionedOutput
{
	 public:
	/// Stamp the next version into `ref` and write it to the `output` slot. Defensively a no-op
	/// (and does not bump) if `output` does not resolve to a `ValueData<RefT>` — the same guard
	/// the nodes had inline. The produce result is intentionally discarded: a producer always
	/// publishes a fresh ref (the bumped version guarantees the change), so it returns `true`
	/// from its own `record`/`execute` regardless.
	template <class RefT>
	void publish(graph::ExecContext& ctx, graph::DataHandle output, RefT ref)
	{
		if (auto* out = dynamic_cast<graph::ValueData<RefT>*>(ctx.data(output)); out != nullptr)
		{
			ref.version = ++m_version;
			static_cast<void>(out->produce(std::move(ref)));
		}
	}

	/// The current version (the value stamped into the most recent publish). A test/introspection
	/// lens; also lets a node that publishes its ref by hand stay in lockstep with this counter.
	[[nodiscard]] std::uint64_t version() const noexcept { return m_version; }

	 private:
	std::uint64_t m_version = 0;
};
} // namespace veng::gpu

#endif // VENG_VERSIONEDOUTPUT_HPP
