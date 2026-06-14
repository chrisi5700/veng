/**
 * @file
 * @author chris
 * @brief The L3-to-Vulkan seam: execution context for GPU render-graph frames.
 *
 * `GpuExecContext` is the @ref veng::graph::ExecContext the driver hands to `Graph::execute` when
 * the frame plan contains GPU work. It resolves data handles against the graph (like the core
 * CPU context) and additionally exposes the RHI command encoder, the engine @ref veng::Context, and
 * the frame-in-flight slot — no `vk::` type appears on its surface. @ref veng::gpu::GpuNode performs
 * the single localized cast to it (see `GpuNode.hpp`).
 *
 * @ingroup gpu_handles
 */

#ifndef VENG_GPUEXECCONTEXT_HPP
#define VENG_GPUEXECCONTEXT_HPP

#include <cstddef>
#include <veng/context/Context.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/rendergraph/nodes/Node.hpp>
#include <veng/resources/ResourcePool.hpp>
#include <veng/rhi/CommandEncoder.hpp>

namespace veng::gpu
{
/**
 * @brief Execution context supplied to GPU nodes during a frame's `Graph::execute` traversal.
 *
 * Extends the engine-neutral `graph::ExecContext` with the RHI surface needed to record commands:
 * the @ref veng::rhi::CommandEncoder, the engine @ref veng::Context (device/allocator/queues/RHI
 * registry), the @ref veng::ResourcePool of N-buffered transient resources, and the current frame
 * slot index.
 *
 * @ingroup gpu_handles
 * @see GpuNode
 * @see SubmitContext
 * @see ResourcePool
 */
class GpuExecContext final : public graph::ExecContext
{
	 public:
	/**
	 * @brief Construct a GPU execution context for one frame.
	 *
	 * The driver (`FrameExecutor`) builds the @ref veng::rhi::CommandEncoder from the frame's command
	 * buffer and hands it in, so this seam exposes the RHI encoder rather than a `vk::CommandBuffer`.
	 *
	 * @param graph      The graph being executed (used for data handle resolution).
	 * @param context    The engine-wide context (device, allocator, queues, RHI registry).
	 * @param pool       The transient resource pool for this frame.
	 * @param encoder    The RHI command encoder over this frame's recording command buffer.
	 * @param frame_slot The in-flight frame slot index (0..N-1).
	 */
	GpuExecContext(const graph::Graph& graph, const Context& context, ResourcePool& pool, rhi::CommandEncoder encoder,
				   std::size_t frame_slot) noexcept
		: m_graph(&graph)
		, m_context(&context)
		, m_pool(&pool)
		, m_encoder(encoder)
		, m_frame_slot(frame_slot)
	{
	}

	/// @brief Resolve a data handle to the underlying `graph::Data` node in the graph.
	/// @param handle The data handle to look up.
	/// @return Pointer to the data node, or `nullptr` if the handle is invalid.
	[[nodiscard]] graph::Data* data(graph::DataHandle handle) const override { return m_graph->get_data(handle); }

	/// @brief Return the graph's current revision, used by the change-cutoff scheduler.
	/// @return The revision counter of the executing graph.
	[[nodiscard]] graph::Revision revision() const noexcept override { return m_graph->current_revision(); }

	/**
	 * @brief Auto-barrier hook: insert image-layout transitions for a node's declared usages.
	 *
	 * Definition lives in `GpuNode.hpp` where @ref veng::gpu::GpuNode is fully defined, to avoid a
	 * circular include. Walks the node's `image_usages()` list and calls
	 * `ResourcePool::transition_image` on the recording command buffer for each entry.
	 * The CPU context's `prepare_for` is a no-op.
	 *
	 * @param node The node about to be recorded; queried for its image usages.
	 */
	void prepare_for(graph::Node& node) noexcept override;

	/// @brief Access the engine-wide context.
	/// @return Reference to the @ref veng::Context owning the device, allocator, and queues.
	[[nodiscard]] const Context& context() const noexcept { return *m_context; }

	/// @brief The RHI command encoder for this frame — the mid-level recording surface nodes use.
	/// @return The @ref veng::rhi::CommandEncoder wrapping this frame's command buffer.
	[[nodiscard]] rhi::CommandEncoder& encoder() noexcept { return m_encoder; }

	/// @brief The RHI handle registry — resolves an edge's `TextureHandle`/`BufferHandle` to vk objects.
	/// @return The @ref veng::rhi::Device from the engine @ref veng::Context.
	[[nodiscard]] rhi::Device& rhi() const noexcept { return m_context->rhi(); }

	/// @brief The in-flight frame slot index for this frame (0..N-1).
	/// @return The slot index used to select the N-buffered copy of each resource.
	[[nodiscard]] std::size_t frame_slot() const noexcept { return m_frame_slot; }

	/**
	 * @brief The engine's transient-resource pool.
	 *
	 * Where GPU nodes obtain their N-buffered render targets and uniform buffers. The driver
	 * advances the pool once per frame via `begin_frame`.
	 *
	 * @return Reference to the @ref veng::ResourcePool for this frame.
	 */
	[[nodiscard]] ResourcePool& pool() const noexcept { return *m_pool; }

	 private:
	const graph::Graph* m_graph;	  ///< Graph being executed.
	const Context*		m_context;	  ///< Engine-wide context (device/allocator/queues/RHI registry).
	ResourcePool*		m_pool;		  ///< Transient resource pool for this frame.
	rhi::CommandEncoder m_encoder;	  ///< Mid-level recording surface over this frame's command buffer.
	std::size_t			m_frame_slot; ///< In-flight slot index (0..N-1).
};
} // namespace veng::gpu

#endif // VENG_GPUEXECCONTEXT_HPP
