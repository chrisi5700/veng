/**
 * @file
 * @author chris
 * @brief Reactive graph nodes, execution-pass interfaces (@ref veng::graph::ExecContext, @ref veng::graph::Scheduler),
 *        and the generic CPU @ref veng::graph::TransformNode.
 *
 * Vulkan-agnostic GPU nodes live in the L4 layer (`veng::gpu`) and plug into these same
 * interfaces. The separation keeps the core graph testable without a GPU.
 *
 * @ingroup rendergraph
 */

#ifndef VENG_NODE_HPP
#define VENG_NODE_HPP

#include <array>
#include <atomic>
#include <expected>
#include <functional>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>

namespace veng::graph
{
class Node; // defined below; ExecContext::prepare_for takes Node&

/**
 * @brief Resolves handles to live data during the execute pass.
 *
 * An interface so the L4 GPU implementation (`gpu::GpuExecContext`) can layer command
 * recording and resource lookup on top while the core and its tests use the plain
 * @ref veng::graph::Graph "Graph-backed" implementation.
 *
 * @ingroup rendergraph
 * @see Scheduler
 * @see Graph::execute
 */
class ExecContext
{
	 public:
	ExecContext()							   = default;
	ExecContext(const ExecContext&)			   = default;
	ExecContext& operator=(const ExecContext&) = default;
	ExecContext(ExecContext&&)				   = default;
	ExecContext& operator=(ExecContext&&)	   = default;
	virtual ~ExecContext()					   = default;

	/**
	 * @brief Resolve a @ref veng::graph::DataHandle to its live @ref veng::graph::Data slot.
	 * @param handle The handle to resolve.
	 * @return Pointer to the live slot, or nullptr if the handle is invalid.
	 */
	[[nodiscard]] virtual Data* data(DataHandle handle) const = 0;

	/// @brief Returns the @ref veng::graph::Revision associated with the current frame being executed.
	[[nodiscard]] virtual Revision revision() const noexcept = 0;

	/**
	 * @brief Pre-execute hook called once per node by `Graph::run_node` *before*
	 *        `node.execute(ctx)`.
	 *
	 * The default is a no-op. The GPU context (`gpu::GpuExecContext`) overrides it to query
	 * the node's declared `image_usages()` and insert the necessary layout/barrier transitions
	 * on the recording command buffer. This lets the executor — not the node — own resource
	 * transitions, so node implementations no longer record their own `image_barrier` calls
	 * for pool-backed targets.
	 *
	 * @param node The node about to execute.
	 */
	virtual void prepare_for(Node& node) noexcept { (void)node; }
};

/**
 * @brief Dependency-injection seam for task execution.
 *
 * The real implementation is @ref veng::graph::ThreadPoolScheduler (a fixed-size worker pool); tests
 * inject the deterministic @ref veng::graph::InlineScheduler.
 *
 * @ingroup rendergraph
 * @see ThreadPoolScheduler
 * @see InlineScheduler
 * @see Graph::execute
 */
class Scheduler
{
	 public:
	Scheduler()							   = default;
	Scheduler(const Scheduler&)			   = default;
	Scheduler& operator=(const Scheduler&) = default;
	Scheduler(Scheduler&&)				   = default;
	Scheduler& operator=(Scheduler&&)	   = default;
	virtual ~Scheduler()				   = default;

	/**
	 * @brief Enqueue a unit of work.
	 * @param task The callable to execute, with no arguments and no return value.
	 */
	virtual void submit(std::function<void()> task) = 0;
};

/**
 * @brief A @ref veng::graph::Scheduler that runs each submitted task immediately on the calling thread.
 *
 * The deterministic scheduler used in unit tests and as a perfectly valid default for
 * single-threaded use. Because @ref veng::graph::Graph::execute submits in topological-height order,
 * a node's dependencies have already completed by the time it runs here.
 *
 * @ingroup rendergraph
 * @see Scheduler
 * @see ThreadPoolScheduler
 */
class InlineScheduler final : public Scheduler
{
	 public:
	void submit(std::function<void()> task) override { task(); }
};

/**
 * @brief Type-erased base for a unit of work in the reactive graph.
 *
 * A node is modelled as a pure function of its resolved inputs producing its outputs
 * (determinism/purity). The resolve pass reads @ref inputs / @ref outputs to walk and
 * topo-sort the graph; the execute pass calls @ref execute and interprets the returned
 * changed flag to drive change-cutoff downstream.
 *
 * @ingroup rendergraph
 * @see ExecContext
 * @see Graph
 * @see TransformNode
 */
class Node
{
	 public:
	Node()						 = default;
	Node(const Node&)			 = delete;
	Node& operator=(const Node&) = delete;
	Node(Node&&)				 = delete;
	Node& operator=(Node&&)		 = delete;
	virtual ~Node()				 = default;

	/**
	 * @brief Returns the handles of all @ref veng::graph::Data slots this node reads (resolve pass).
	 * @return A span over the node's input @ref veng::graph::DataHandle set.
	 */
	[[nodiscard]] virtual std::span<const DataHandle> inputs() const = 0;

	/**
	 * @brief Returns the handles of all @ref veng::graph::Data slots this node writes (resolve pass).
	 * @return A span over the node's output @ref veng::graph::DataHandle set.
	 */
	[[nodiscard]] virtual std::span<const DataHandle> outputs() const = 0;

	/**
	 * @brief Perform the node's work during the execute pass.
	 *
	 * Returns whether the output actually changed, which drives change-cutoff: a false
	 * return stops the dirty wave from propagating to downstream consumers.
	 *
	 * @param ctx The execution context providing handle resolution and frame @ref veng::graph::Revision.
	 * @return Whether the node's output changed, or an @ref veng::graph::ExecError on failure.
	 */
	[[nodiscard]] virtual std::expected<bool, ExecError> execute(ExecContext& ctx) = 0;

	/**
	 * @brief Demand-refresh hook: return true to be re-evaluated every demanded frame
	 *        even when inputs are unchanged.
	 *
	 * Used by progressive accumulators that have not yet converged: each frame they read
	 * their own previous output (a temporal self-edge via @ref veng::graph::HistoryData) and add one more
	 * sample. Default is false: an ordinary pure node re-runs only when an input actually
	 * changes, so a static scene costs nothing.
	 *
	 * @return Whether this node must re-run regardless of input stamps.
	 */
	[[nodiscard]] virtual bool needs_refresh() const noexcept { return false; }

	/// @brief Returns the current @ref veng::graph::ExecutionState of this node (atomic, acquire).
	[[nodiscard]] ExecutionState state() const noexcept { return m_state.load(std::memory_order_acquire); }

	/**
	 * @brief Force this node back to @ref veng::graph::ExecutionState::INVALID so the next @ref
	 * veng::graph::Graph::resolve re-plans it regardless of input stamps.
	 *
	 * Use when a *runtime* mutation extends the node's input set or changes how it draws —
	 * a new push-constant edge, a new mesh draw, a different sampled-image binding — because
	 * those edits are not observable to the planner's `input.changed_at > verified_at` check
	 * and would leave the node silently stale until something else dirties it.
	 */
	void mark_dirty() noexcept { m_state.store(ExecutionState::INVALID, std::memory_order_release); }

	 protected:
	friend class Graph;

	/**
	 * @brief Returns the @ref veng::graph::Revision at which this node last produced output (or was
	 *        confirmed current).
	 *
	 * @note Temporal nodes MUST detect "did input X change since I last ran" by comparing
	 *       `data.changed_at() > last_run_revision()` — never `== current revision`, which
	 *       misses a change made on a frame this node was not demanded.
	 *
	 * @return The revision stamp set by the most recent successful `Graph::run_node` call.
	 */
	[[nodiscard]] Revision last_run_revision() const noexcept { return m_verified_at; }

	std::atomic<ExecutionState> m_state{ExecutionState::INVALID};
	Revision					m_verified_at = 0; ///< Revision this node was last confirmed valid at.
};

/**
 * @brief The workhorse CPU node: wraps a pure callable over typed inputs into the graph.
 *
 * Edges are stored as @ref veng::graph::DataHandle values; the concrete @ref veng::graph::ValueData types are
 * recovered at execute time via the @ref veng::graph::ExecContext. The equality-gated @ref
 * veng::graph::ValueData::produce call in @ref veng::graph::Node::execute drives change-cutoff automatically.
 *
 * @ingroup rendergraph
 * @tparam Signature A function-type of the form `R(Args...)` that describes the callable.
 * @see Graph::add_transform
 * @see ExecContext
 */
template <class Signature>
class TransformNode;

/**
 * @brief Specialization of @ref veng::graph::TransformNode for a callable `R(const Args&...)`.
 * @ingroup rendergraph
 * @tparam R Return type; stored in the output @ref veng::graph::ValueData<R> slot.
 * @tparam Args Input types; resolved from @ref veng::graph::ValueData<Args> slots at execute time.
 */
template <class R, class... Args>
class TransformNode<R(Args...)> final : public Node
{
	 public:
	/// @brief The function type this node wraps.
	using Function = std::function<R(const Args&...)>;

	/**
	 * @brief Construct with the callable, input handles, and output handle.
	 * @param func    The pure function to invoke with the resolved input values.
	 * @param inputs  Handles to the @ref veng::graph::ValueData<Args> input slots, in parameter order.
	 * @param output  Handle to the @ref veng::graph::ValueData<R> output slot.
	 */
	TransformNode(Function func, std::array<DataHandle, sizeof...(Args)> inputs, DataHandle output)
		: m_func(std::move(func))
		, m_inputs(inputs)
		, m_output(output)
	{
	}

	[[nodiscard]] std::span<const DataHandle> inputs() const override { return m_inputs; }
	[[nodiscard]] std::span<const DataHandle> outputs() const override { return {&m_output, 1}; }

	[[nodiscard]] std::expected<bool, ExecError> execute(ExecContext& ctx) override
	{
		return [&]<std::size_t... Is>(std::index_sequence<Is...>) -> std::expected<bool, ExecError>
		{
			std::tuple<ValueData<Args>*...> args{dynamic_cast<ValueData<Args>*>(ctx.data(m_inputs[Is]))...};
			if (((std::get<Is>(args) == nullptr) || ...))
			{
				return std::unexpected(ExecError::MISSING_INPUT);
			}
			auto* out = dynamic_cast<ValueData<R>*>(ctx.data(m_output));
			if (out == nullptr)
			{
				return std::unexpected(ExecError::MISSING_INPUT);
			}
			return out->produce(m_func(std::get<Is>(args)->value()...));
		}(std::make_index_sequence<sizeof...(Args)>{});
	}

	 private:
	Function								m_func;
	std::array<DataHandle, sizeof...(Args)> m_inputs;
	DataHandle								m_output;
};
} // namespace veng::graph

#endif // VENG_NODE_HPP
