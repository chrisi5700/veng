/**
 * @file
 * @author chris
 * @brief Helper that centralises the "stamp a fresh version, then produce" step for producer nodes.
 *
 * The reference types that flow on graph edges — @ref veng::gpu::MeshRef, @ref veng::gpu::UniformRef, @ref
 * veng::gpu::BufferRef,
 * @ref veng::gpu::ImageRef — each carry a `version` the producer bumps on every publish so two structurally
 * identical refs from consecutive runs still compare unequal. This is what makes
 * `ValueData<Ref>`'s change-cutoff fire when a stable handle (a reused buffer, a recycled
 * swapchain image) is re-filled with new contents.
 *
 * Before this helper, `MeshNode`, `UniformNode`, `StorageBufferNode`, `DynamicMeshNode`,
 * `BlitNode`, and `GraphicsNode` each carried their own `std::uint64_t m_version` and
 * hand-wrote `++m_version; out->produce({…, .version = m_version})`. @ref veng::gpu::VersionedOutput
 * owns that counter and the publish step in one place, making the "a producer always publishes
 * a fresh ref" contract explicit and unforgettable.
 *
 * @ingroup gpu_handles
 */

#ifndef VENG_VERSIONEDOUTPUT_HPP
#define VENG_VERSIONEDOUTPUT_HPP

#include <cstdint>
#include <utility>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/nodes/Node.hpp>

namespace veng::gpu
{
/**
 * @brief Monotonic version counter with a combined stamp-and-produce publish step.
 *
 * Owned by producer nodes. On each frame where the producer has new data, it calls @ref publish
 * instead of manually incrementing its own counter and calling `out->produce`. The `RefT`
 * template parameter must have a `std::uint64_t version` member.
 *
 * @ingroup gpu_handles
 * @see ImageRef
 * @see MeshRef
 * @see UniformRef
 * @see BufferRef
 */
class VersionedOutput
{
	 public:
	/**
	 * @brief Stamp the next version into `ref` and write it to the `output` data slot.
	 *
	 * Defensively a no-op (and does not bump the counter) if `output` does not resolve to a
	 * `ValueData<RefT>` — the same guard the nodes had inline. The produce result is
	 * intentionally discarded: a producer always publishes a fresh ref (the bumped version
	 * guarantees the change), so it returns `true` from its own `record`/`execute` regardless.
	 *
	 * @tparam RefT  A ref type with a `std::uint64_t version` member (e.g. @ref veng::gpu::ImageRef).
	 * @param  ctx    The execution context used to resolve `output` to a `ValueData<RefT>`.
	 * @param  output Data handle for the output slot to write into.
	 * @param  ref    The ref to publish; `version` is overwritten with the bumped counter.
	 */
	template <class RefT>
	void publish(graph::ExecContext& ctx, graph::DataHandle output, RefT ref)
	{
		if (auto* out = dynamic_cast<graph::ValueData<RefT>*>(ctx.data(output)); out != nullptr)
		{
			ref.version = ++m_version;
			static_cast<void>(out->produce(std::move(ref)));
		}
	}

	/**
	 * @brief The current version (the value stamped into the most recent publish).
	 *
	 * A test and introspection lens; also lets a node that publishes its ref by hand stay in
	 * lockstep with this counter.
	 *
	 * @return The version stamped by the last @ref publish call (0 if never published).
	 */
	[[nodiscard]] std::uint64_t version() const noexcept { return m_version; }

	 private:
	std::uint64_t m_version = 0; ///< Monotonically increasing counter, bumped on every @ref publish.
};
} // namespace veng::gpu

#endif // VENG_VERSIONEDOUTPUT_HPP
