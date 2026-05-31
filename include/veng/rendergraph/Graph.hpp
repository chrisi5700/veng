/**
 * @file
 * @author chris
 * @brief The reactive render graph: @ref veng::graph::TypedHandle, @ref veng::graph::FramePlan, and @ref
 * veng::graph::Graph.
 *
 * The @ref veng::graph::Graph is a persistent dependency graph evaluated by demand-driven incremental
 * computation. It owns all @ref veng::graph::Node and @ref veng::graph::Data slots, applies frame-boundary source
 * snapshots, resolves the demanded cone into a topologically-ordered @ref veng::graph::FramePlan, and
 * dispatches it through an injected @ref veng::graph::Scheduler. Entirely Vulkan-agnostic.
 *
 * @ingroup rendergraph
 */

#ifndef VENG_GRAPH_HPP
#define VENG_GRAPH_HPP

#include <concepts>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/nodes/Node.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>

namespace veng::graph
{
/**
 * @brief Compile-time-typed wrapper around a @ref veng::graph::DataHandle.
 *
 * Provides typed edges via templates while keeping runtime handles underneath.
 * Converts implicitly to the untyped @ref veng::graph::DataHandle so it can be used directly
 * wherever a sink is expected.
 *
 * @ingroup rendergraph
 * @tparam T The value type stored in the referenced @ref veng::graph::ValueData slot.
 * @see Graph::add_source
 * @see Graph::add_transform
 * @see Graph::add_history
 */
template <class T>
struct TypedHandle
{
	DataHandle handle{}; ///< The underlying untyped handle.

	/// @brief Returns true if the underlying handle is valid (not the sentinel).
	[[nodiscard]] bool valid() const noexcept { return handle.valid(); }

	/// @brief Implicit conversion to @ref veng::graph::DataHandle for use as a graph sink.
	constexpr explicit(false) operator DataHandle() const noexcept { return handle; }
};

/**
 * @brief The dirty ∩ demanded node set for one frame, ordered by topological height.
 *
 * A shared dependency appears before either consumer so that @ref veng::graph::Graph::execute can
 * dispatch independent nodes within a height band in parallel without synchronisation.
 * Produced by @ref veng::graph::Graph::resolve; consumed by @ref veng::graph::Graph::execute.
 *
 * @ingroup rendergraph
 * @see Graph::resolve
 * @see Graph::execute
 */
class FramePlan
{
	 public:
	/// @brief Returns the ordered sequence of @ref veng::graph::NodeHandle values in this plan.
	[[nodiscard]] std::span<const NodeHandle> nodes() const noexcept { return m_order; }

	/// @brief Returns the number of nodes in this plan.
	[[nodiscard]] std::size_t size() const noexcept { return m_order.size(); }

	/// @brief Returns true if no nodes need to execute this frame.
	[[nodiscard]] bool empty() const noexcept { return m_order.empty(); }

	/// @brief Returns the @ref veng::graph::Revision at which this plan was built.
	[[nodiscard]] Revision revision() const noexcept { return m_revision; }

	 private:
	friend class Graph;
	std::vector<NodeHandle>	   m_order;	  ///< Dirty ∩ demanded nodes, ascending topological height.
	std::vector<std::uint32_t> m_heights; ///< Topological height, parallel to @ref m_order.
	Revision				   m_revision = 0;
};

/**
 * @brief The reactive render graph: owns all nodes and data, plans frames, and executes them.
 *
 * The @ref veng::graph::Graph drives demand-driven incremental computation: only the dirty ∩ demanded
 * subset of nodes executes each frame. Sources are mutated between frames via @ref set (queued)
 * or @ref set_now (immediate, for execute-time driver resources like a swapchain image).
 *
 * **Frame lifecycle:**
 * 1. @ref resolve — apply queued source mutations, bump the @ref veng::graph::Revision, walk the demanded
 *    cone, return a height-sorted @ref veng::graph::FramePlan.
 * 2. @ref execute — dispatch the plan through a @ref veng::graph::Scheduler, applying change-cutoff and
 *    CAS-based completion per node.
 *
 * Or call @ref frame for the combined convenience wrapper.
 *
 * @ingroup rendergraph
 * @see FramePlan
 * @see Node
 * @see Data
 * @see Scheduler
 */
class Graph
{
	 public:
	// --- Low-level construction (graph takes ownership) -------------------------

	/**
	 * @brief Transfer ownership of a @ref veng::graph::Node into the graph.
	 * @param node The node to own.
	 * @return A @ref veng::graph::NodeHandle referencing the newly added slot.
	 */
	NodeHandle add(std::unique_ptr<Node> node);

	/**
	 * @brief Transfer ownership of a @ref veng::graph::Data slot into the graph.
	 *
	 * Sets the slot's @ref veng::graph::Data::m_graph and @ref veng::graph::Data::m_self back-references.
	 *
	 * @param data The data slot to own.
	 * @return A @ref veng::graph::DataHandle referencing the newly added slot.
	 */
	DataHandle add(std::unique_ptr<Data> data);

	/**
	 * @brief Declare that `producer` writes `data`.
	 *
	 * The typed @ref add_transform helper calls this for you. Custom node types built via
	 * the low-level @ref add overloads call it to register their outputs — and it is the
	 * seam tests use to deliberately build cycles for cycle-detection coverage.
	 *
	 * @param data     The @ref veng::graph::DataHandle of the output slot.
	 * @param producer The @ref veng::graph::NodeHandle of the node that writes it.
	 */
	void set_producer(DataHandle data, NodeHandle producer);

	// --- Typed authoring helpers ------------------------------------------------

	/**
	 * @brief Add an externally-mutated source value.
	 *
	 * Mutate the returned handle later via @ref set, which queues the new value until the
	 * next frame boundary. Use @ref set_now for execute-time writes that must not trigger
	 * replanning.
	 *
	 * @tparam T The value type.
	 * @param initial The initial value held by this source before the first mutation.
	 * @return A @ref veng::graph::TypedHandle<T> referencing the new source slot.
	 */
	template <class T>
	TypedHandle<T> add_source(T initial)
	{
		return TypedHandle<T>{add(std::make_unique<ValueData<T>>(std::move(initial)))};
	}

	/**
	 * @brief Queue a typed source mutation, applied at the next frame boundary.
	 *
	 * The one place the @ref veng::graph::ValueData downcast lives, so driver/author code mutates
	 * sources through the compile-time-typed handle and never touches a `dynamic_cast`.
	 * A no-op if the handle does not resolve to a @ref veng::graph::ValueData — a type mismatch
	 * the typed handle makes unrepresentable for handles obtained from @ref add_source<T>.
	 *
	 * @tparam T The value type.
	 * @param handle The typed handle returned by @ref add_source<T>.
	 * @param value  The new value to queue.
	 */
	template <class T>
	void set(TypedHandle<T> handle, T value)
	{
		if (auto* data = dynamic_cast<ValueData<T>*>(get_data(handle.handle)); data != nullptr)
		{
			data->set(std::move(value));
		}
	}

	/**
	 * @brief Write a source's value *immediately* for the frame currently being executed.
	 *
	 * NOT a queued, frame-boundary mutation and NOT a dirtiness pulse — it does not stamp
	 * `changed_at`, so it never re-triggers planning next frame. This is the seam for an
	 * execute-time resource the driver provides *after* planning has decided to render —
	 * e.g. the just-acquired swapchain image handed to the present/blit node. Use @ref set
	 * (not this) for reactive inputs whose change should drive recomputation. Only valid for
	 * a source (no producer); the caller must order it before the reading node executes.
	 *
	 * @tparam T The value type.
	 * @param handle The typed handle returned by @ref add_source<T>.
	 * @param value  The value to publish immediately into the live slot.
	 */
	template <class T>
	void set_now(TypedHandle<T> handle, T value)
	{
		if (auto* data = dynamic_cast<ValueData<T>*>(get_data(handle.handle)); data != nullptr)
		{
			static_cast<void>(data->produce(std::move(value)));
		}
	}

	/**
	 * @brief Add a one-frame-delayed *history* (feedback) edge shadowing `live`.
	 *
	 * The returned handle reads, during frame F, whatever `live` held as of frame F-1 —
	 * and presents to the planner as a source, so feeding it as an input to the node that
	 * produces `live` forms no cycle. This is the engine's temporal primitive: ping-pong,
	 * TAA history, and progressive accumulation are wired as
	 * `consumer(history(O), …) -> O` instead of a hand-managed `needs_refresh` flag and
	 * manual stamp compare. `initial` is what reads see on frame 0 (before `live` has ever
	 * produced). See @ref veng::graph::HistoryData for the convergence semantics.
	 *
	 * @tparam T The value type; must match `live`'s @ref veng::graph::ValueData.
	 * @param live    The typed handle of the live output slot to shadow.
	 * @param initial Value returned by readers on frame 0.
	 * @return A @ref veng::graph::TypedHandle<T> referencing the new history slot.
	 */
	template <class T>
	TypedHandle<T> add_history(TypedHandle<T> live, T initial)
	{
		const DataHandle handle = add(std::make_unique<HistoryData<T>>(live.handle, std::move(initial)));
		m_history.push_back(handle);
		return TypedHandle<T>{handle};
	}

	/**
	 * @brief Add a pure CPU transform: `func` is called with the resolved input values and
	 *        its result becomes a new graph-owned output slot.
	 *
	 * @tparam F    A callable with signature compatible with `R(const Args&...)`.
	 * @tparam Args Input value types; each must have a corresponding @ref veng::graph::TypedHandle<Args> argument.
	 * @param func   The pure function to wrap.
	 * @param inputs The typed input handles, one per `Args`.
	 * @return A @ref veng::graph::TypedHandle<R> referencing the new output slot.
	 */
	template <class F, class... Args>
	auto add_transform(F&& func, TypedHandle<Args>... inputs) -> TypedHandle<std::invoke_result_t<F&, const Args&...>>
	{
		using R = std::invoke_result_t<F&, const Args&...>;
		static_assert(std::default_initializable<R>, "transform output type must be default-constructible");

		const DataHandle							  output = add(std::make_unique<ValueData<R>>(R{}));
		const std::array<DataHandle, sizeof...(Args)> input_handles{inputs.handle...};

		auto node = std::make_unique<TransformNode<R(Args...)>>(std::function<R(const Args&...)>(std::forward<F>(func)),
																input_handles, output);
		const NodeHandle node_handle = add(std::move(node));

		set_producer(output, node_handle);
		return TypedHandle<R>{output};
	}

	// --- Frame lifecycle --------------------------------------------------------

	/**
	 * @brief Queue a source to be re-snapshotted at the next frame boundary.
	 *
	 * Normally driven for you by @ref veng::graph::ValueData::set; idempotent within a frame (duplicate
	 * entries are ignored).
	 *
	 * @param handle The @ref veng::graph::DataHandle of the source slot to mark pending.
	 */
	void mutate_source(DataHandle handle);

	/**
	 * @brief Synchronous planning pass: produce a @ref veng::graph::FramePlan from the demanded sinks.
	 *
	 * Applies queued source mutations, bumps the @ref veng::graph::Revision, walks the demanded cone from
	 * `sinks`, and returns the dirty ∩ demanded plan sorted by topological height. Also
	 * re-snapshots all @ref veng::graph::HistoryData slots (feedback edges). Does no node execution.
	 *
	 * @param sinks The @ref veng::graph::DataHandle values of the graph outputs the caller wants this frame.
	 * @return The frame plan on success, or @ref veng::graph::GraphError::CYCLE_DETECTED /
	 *         @ref veng::graph::GraphError::INVALID_SINK on failure.
	 */
	[[nodiscard]] std::expected<FramePlan, GraphError> resolve(std::span<const DataHandle> sinks);

	/**
	 * @brief Asynchronous execution pass: dispatch the plan through `scheduler`.
	 *
	 * Applies change-cutoff and CAS-based completion per node. The first overload uses the
	 * built-in CPU @ref veng::graph::ExecContext; the second injects one — the GPU path passes a
	 * `gpu::GpuExecContext` so @ref veng::graph::Node subclasses see command buffers and resources.
	 * `ctx` must outlive the call (execute is frame-synchronous).
	 *
	 * Returns true if every node in the plan ended in @ref veng::graph::ExecutionState::VALID; false if
	 * any ended in @ref veng::graph::ExecutionState::FAILED. A GPU-path caller **must** check this before
	 * submitting the recorded command buffer — with auto-barrier insertion, a failed node may
	 * have produced state the rest of the recording assumed; submitting a half-baked frame is
	 * the wrong recovery.
	 *
	 * @param plan      The @ref veng::graph::FramePlan produced by @ref resolve.
	 * @param scheduler The @ref veng::graph::Scheduler that will run node tasks.
	 * @return True if all nodes completed successfully; false if any node failed.
	 */
	[[nodiscard]] bool execute(const FramePlan& plan, Scheduler& scheduler);

	/**
	 * @brief Overload that injects a custom @ref veng::graph::ExecContext (e.g. `gpu::GpuExecContext`).
	 * @param plan      The @ref veng::graph::FramePlan produced by @ref resolve.
	 * @param scheduler The @ref veng::graph::Scheduler that will run node tasks.
	 * @param ctx       The execution context; must outlive this call.
	 * @return True if all nodes completed successfully; false if any node failed.
	 */
	[[nodiscard]] bool execute(const FramePlan& plan, Scheduler& scheduler, ExecContext& ctx);

	/**
	 * @brief Convenience wrapper: one full frame — @ref resolve then @ref execute.
	 * @param sinks     The demanded graph outputs.
	 * @param scheduler The scheduler to use for execution.
	 * @return The @ref veng::graph::FramePlan (node states observable via @ref get_node), or a
	 *         @ref veng::graph::GraphError if planning failed.
	 */
	[[nodiscard]] std::expected<FramePlan, GraphError> frame(std::span<const DataHandle> sinks, Scheduler& scheduler);

	/**
	 * @brief Single-sink convenience overload of @ref frame.
	 * @param sink      The single demanded graph output.
	 * @param scheduler The scheduler to use for execution.
	 * @return The @ref veng::graph::FramePlan, or a @ref veng::graph::GraphError if planning failed.
	 */
	[[nodiscard]] std::expected<FramePlan, GraphError> frame(DataHandle sink, Scheduler& scheduler);

	// --- Queries / tooling ------------------------------------------------------

	/// @brief Returns the current global @ref veng::graph::Revision (bumped by each @ref resolve call).
	[[nodiscard]] Revision current_revision() const noexcept { return m_revision; }

	/// @brief Returns the number of @ref veng::graph::Node slots owned by this graph.
	[[nodiscard]] std::size_t node_count() const noexcept { return m_nodes.size(); }

	/// @brief Returns the number of @ref veng::graph::Data slots owned by this graph.
	[[nodiscard]] std::size_t data_count() const noexcept { return m_data.size(); }

	/**
	 * @brief Resolve a @ref veng::graph::NodeHandle to a live @ref veng::graph::Node pointer.
	 * @param handle The handle to resolve.
	 * @return Pointer to the node, or nullptr if the handle is invalid or the generation mismatches.
	 */
	[[nodiscard]] Node* get_node(NodeHandle handle) const;

	/**
	 * @brief Resolve a @ref veng::graph::DataHandle to a live @ref veng::graph::Data pointer.
	 * @param handle The handle to resolve.
	 * @return Pointer to the slot, or nullptr if the handle is invalid or the generation mismatches.
	 */
	[[nodiscard]] Data* get_data(DataHandle handle) const;

	/**
	 * @brief Dump the graph topology to a GraphViz DOT string, coloured by @ref veng::graph::ExecutionState.
	 * @return A DOT-format string suitable for rendering with Graphviz.
	 */
	[[nodiscard]] std::string to_dot() const;

	 private:
	/// @brief Per-node scratch state for the recursive demanded-cone walk in @ref resolve.
	struct ResolveState
	{
		std::vector<std::uint8_t>  color;  ///< DFS colour: 0 = white, 1 = gray, 2 = black.
		std::vector<std::uint32_t> height; ///< Topological height of each node.
		std::vector<std::uint8_t>  dirty;  ///< 1 if the node is dirty and must execute.
		std::vector<NodeHandle>	   order;  ///< Nodes added to the plan in visit order.
	};

	/**
	 * @brief Recursive DFS visit for cycle detection, height computation, and dirty propagation.
	 * @param index Node slot index to visit.
	 * @param state Mutable per-resolve scratch state.
	 * @return void on success, @ref veng::graph::GraphError::CYCLE_DETECTED if a back-edge is found.
	 */
	[[nodiscard]] std::expected<void, GraphError> visit(std::uint32_t index, ResolveState& state);

	/**
	 * @brief Execute one planned node: apply change-cutoff, call @ref veng::graph::Node::execute, then
	 *        CAS-complete the node and stamp output revisions.
	 * @param node       The node to run.
	 * @param ctx        The execution context for handle resolution and barrier insertion.
	 * @param revision   The frame @ref veng::graph::Revision (from the plan).
	 * @param first_time True if this node has never produced a valid output before.
	 */
	void run_node(Node& node, ExecContext& ctx, Revision revision, bool first_time) const;

	std::vector<std::unique_ptr<Node>> m_nodes;
	std::vector<std::unique_ptr<Data>> m_data;
	/// @brief Generation per slot, validated by @ref get_node / @ref get_data.
	///
	/// A stale handle is rejected rather than silently aliasing a future occupant (UAF/ABA
	/// guard). v1 is append-only so these stay 0; they become load-bearing once eviction bumps
	/// a slot's generation on reuse.
	std::vector<std::uint32_t> m_node_generations;
	std::vector<std::uint32_t> m_data_generations; ///< Generation per data slot; parallel to @ref m_data.
	std::vector<DataHandle>	   m_pending_sources;  ///< Sources queued for commit at the next frame boundary.
	/// @brief @ref veng::graph::HistoryData slots, re-snapshotted from their live slot every frame in @ref resolve
	///        so a node's previous-frame output can flow back as an input without a planning cycle.
	std::vector<DataHandle> m_history;
	Revision				m_revision = 0; ///< Global logical clock; bumped once per @ref resolve call.
};
} // namespace veng::graph

#endif // VENG_GRAPH_HPP
