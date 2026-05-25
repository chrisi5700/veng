//
// Created by chris on 1/24/26.
//
// L3 Reactive Core — Node (a unit of work) plus the execution-pass interfaces
// (ExecContext, Scheduler) and the generic CPU transform node. Vulkan-agnostic
// (design.md §L3): GPU nodes live in L4 and plug into these same interfaces.
//

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
/// Resolves handles to live data during the execute pass. An interface so the L4
/// GPU implementation can layer command recording / resource lookup on top while
/// the core and its tests use the plain Graph-backed one (design.md §L3).
class ExecContext
{
	 public:
	ExecContext()							   = default;
	ExecContext(const ExecContext&)			   = default;
	ExecContext& operator=(const ExecContext&) = default;
	ExecContext(ExecContext&&)				   = default;
	ExecContext& operator=(ExecContext&&)	   = default;
	virtual ~ExecContext()					   = default;

	[[nodiscard]] virtual Data*	   data(DataHandle handle) const = 0;
	[[nodiscard]] virtual Revision revision() const noexcept	 = 0;
};

/// The dependency-injection seam for execution (design.md §5, §L5, §10). The real
/// implementation is a work-stealing job pool; tests inject a deterministic
/// single-threaded runner (see InlineScheduler).
class Scheduler
{
	 public:
	Scheduler()							   = default;
	Scheduler(const Scheduler&)			   = default;
	Scheduler& operator=(const Scheduler&) = default;
	Scheduler(Scheduler&&)				   = default;
	Scheduler& operator=(Scheduler&&)	   = default;
	virtual ~Scheduler()				   = default;

	virtual void submit(std::function<void()> task) = 0;
};

/// Runs each submitted task immediately on the calling thread. The deterministic
/// scheduler the design calls for in unit tests, and a perfectly valid default for
/// single-threaded use. Because the Graph submits in topological-height order, a
/// node's dependencies have already completed by the time it runs here.
class InlineScheduler final : public Scheduler
{
	 public:
	void submit(std::function<void()> task) override { task(); }
};

/// Type-erased base for a unit of work. A node is a pure function of its resolved
/// inputs producing its outputs (design.md §8: determinism/purity).
class Node
{
	 public:
	Node()						 = default;
	Node(const Node&)			 = delete;
	Node& operator=(const Node&) = delete;
	Node(Node&&)				 = delete;
	Node& operator=(Node&&)		 = delete;
	virtual ~Node()				 = default;

	// Resolve pass (sync): report edges so the planner can walk and topo-sort.
	[[nodiscard]] virtual std::span<const DataHandle> inputs() const  = 0;
	[[nodiscard]] virtual std::span<const DataHandle> outputs() const = 0;

	// Execute pass (async): do the work. Returns whether the output actually
	// changed, which drives change-cutoff (design.md §2.4).
	[[nodiscard]] virtual std::expected<bool, ExecError> execute(ExecContext& ctx) = 0;

	// Demand-refresh hook (design.md §L4 temporal / accumulation). Return true to be
	// re-evaluated on every demanded frame even when inputs are unchanged — e.g. a
	// progressive accumulator that has not yet converged, which reads its own previous
	// output (a temporal self-edge) and adds one more sample. Default false: an
	// ordinary pure node re-runs only when an input actually changes, so a static
	// scene keeps costing nothing.
	[[nodiscard]] virtual bool needs_refresh() const noexcept { return false; }

	[[nodiscard]] ExecutionState state() const noexcept { return m_state.load(std::memory_order_acquire); }

	 protected:
	friend class Graph;

	/// The revision at which this node last produced (or was confirmed current).
	/// Temporal nodes MUST detect "did input X change since I last ran" by comparing
	/// `data.changed_at() > last_run_revision()` — never `== current revision`, which
	/// misses a change made on a frame this node was not demanded (the C1/M7 trap).
	[[nodiscard]] Revision last_run_revision() const noexcept { return m_verified_at; }

	std::atomic<ExecutionState> m_state{ExecutionState::INVALID};
	Revision					m_verified_at = 0; // revision this node was last confirmed at
};

/// The workhorse CPU node (design.md §L4 "Generic transform node"): wraps a pure
/// callable over typed inputs into the graph. Edges are stored as handles; the
/// concrete `ValueData<T>` types are recovered at execute via the ExecContext.
template <class Signature>
class TransformNode;

template <class R, class... Args>
class TransformNode<R(Args...)> final : public Node
{
	 public:
	using Function = std::function<R(const Args&...)>;

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
