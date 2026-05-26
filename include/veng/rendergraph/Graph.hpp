//
// Created by chris on 5/25/26.
//
// L3 Reactive Core — the Graph: a persistent dependency graph evaluated by
// demand-driven incremental computation (design.md §2). Owns all nodes and data,
// applies frame-boundary source snapshots, resolves the demanded cone into a
// topologically-ordered frame plan, and dispatches it through an injected
// Scheduler. Entirely Vulkan-agnostic (design.md §L3).
//

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
/// A compile-time-typed wrapper around a `DataHandle` (design.md §L6: "compile-time
/// typed edges via templates, runtime handles underneath"). Converts implicitly to
/// the untyped handle so it can be used directly as a sink.
template <class T>
struct TypedHandle
{
	DataHandle handle{};

	[[nodiscard]] bool		  valid() const noexcept { return handle.valid(); }
	constexpr explicit(false) operator DataHandle() const noexcept { return handle; }
};

/// The dirty ∩ demanded node set for one frame, ordered by topological height so a
/// shared dependency is resolved before either consumer (design.md §2.6, diamonds).
class FramePlan
{
	 public:
	[[nodiscard]] std::span<const NodeHandle> nodes() const noexcept { return m_order; }
	[[nodiscard]] std::size_t				  size() const noexcept { return m_order.size(); }
	[[nodiscard]] bool						  empty() const noexcept { return m_order.empty(); }
	[[nodiscard]] Revision					  revision() const noexcept { return m_revision; }

	 private:
	friend class Graph;
	std::vector<NodeHandle>	   m_order;	  // dirty ∩ demanded nodes, ascending height
	std::vector<std::uint32_t> m_heights; // topological height, parallel to m_order
	Revision				   m_revision = 0;
};

class Graph
{
	 public:
	// --- Low-level construction (graph takes ownership) -------------------------
	NodeHandle add(std::unique_ptr<Node> node);
	DataHandle add(std::unique_ptr<Data> data);

	/// Declare that `producer` writes `data`. The typed `add_transform` helper does
	/// this for you; custom node types built via the low-level `add` overloads call
	/// it to register their outputs (and it is the seam tests use to build cycles).
	void set_producer(DataHandle data, NodeHandle producer);

	// --- Typed authoring helpers (design.md §L6) --------------------------------

	/// Add an externally-mutated source value. Mutate it later via the returned
	/// handle and `ValueData<T>::set`, which queues until the next frame boundary.
	template <class T>
	TypedHandle<T> add_source(T initial)
	{
		return TypedHandle<T>{add(std::make_unique<ValueData<T>>(std::move(initial)))};
	}

	/// Queue a typed source mutation (applied at the next frame boundary). The one place
	/// the `ValueData<T>` downcast lives, so driver/author code mutates sources through
	/// the compile-time-typed handle and never touches a `dynamic_cast`. A no-op if the
	/// handle does not resolve to a `ValueData<T>` (a type mismatch the typed handle makes
	/// unrepresentable for handles obtained from `add_source<T>`).
	template <class T>
	void set(TypedHandle<T> handle, T value)
	{
		if (auto* data = dynamic_cast<ValueData<T>*>(get_data(handle.handle)); data != nullptr)
		{
			data->set(std::move(value));
		}
	}

	/// Write a source's value *immediately*, for the frame currently being executed —
	/// NOT a queued, frame-boundary mutation and NOT a dirtiness pulse (it does not stamp
	/// `changed_at`, so it never re-triggers planning next frame). This is the seam for an
	/// execute-time resource the driver provides *after* planning has decided to render —
	/// e.g. the just-acquired swapchain image handed to the present/blit. Use `set` (not
	/// this) for reactive inputs whose change should drive recomputation. Only valid for a
	/// source (no producer); the caller must order it before the reading node executes.
	template <class T>
	void set_now(TypedHandle<T> handle, T value)
	{
		if (auto* data = dynamic_cast<ValueData<T>*>(get_data(handle.handle)); data != nullptr)
		{
			static_cast<void>(data->produce(std::move(value)));
		}
	}

	/// Add a pure CPU transform: `func` is invoked with the (const-ref) values of
	/// `inputs` and its result becomes a new graph-owned output value.
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

	// --- Frame lifecycle (design.md §2.6) ---------------------------------------

	/// Queue a source to be re-snapshotted at the next frame boundary. Normally
	/// driven for you by `ValueData<T>::set`; idempotent within a frame.
	void mutate_source(DataHandle handle);

	/// Sync planning pass: apply queued mutations, bump the revision, walk the
	/// demanded cone from `sinks`, and return the dirty ∩ demanded plan (or a cycle
	/// error). Does no work itself.
	[[nodiscard]] std::expected<FramePlan, GraphError> resolve(std::span<const DataHandle> sinks);

	/// Async execution pass: dispatch the plan through `scheduler`, applying
	/// change-cutoff (design.md §2.4) and CAS completion (design.md §L3) per node.
	/// The first overload uses the built-in CPU context; the second injects one — the
	/// L4/L5 GPU path passes a `GpuExecContext` so `GpuNode`s see command buffers /
	/// resources (design.md §L4). `ctx` must outlive the call (execute is synchronous).
	void execute(const FramePlan& plan, Scheduler& scheduler);
	void execute(const FramePlan& plan, Scheduler& scheduler, ExecContext& ctx);

	/// Convenience: one full frame — resolve then execute.
	[[nodiscard]] std::expected<FramePlan, GraphError> frame(std::span<const DataHandle> sinks, Scheduler& scheduler);
	[[nodiscard]] std::expected<FramePlan, GraphError> frame(DataHandle sink, Scheduler& scheduler);

	// --- Queries / tooling ------------------------------------------------------
	[[nodiscard]] Revision	  current_revision() const noexcept { return m_revision; }
	[[nodiscard]] std::size_t node_count() const noexcept { return m_nodes.size(); }
	[[nodiscard]] std::size_t data_count() const noexcept { return m_data.size(); }

	[[nodiscard]] Node* get_node(NodeHandle handle) const;
	[[nodiscard]] Data* get_data(DataHandle handle) const;

	/// Dump the topology to GraphViz, colored by execution state (design.md §L6).
	[[nodiscard]] std::string to_dot() const;

	 private:
	// Recursive demanded-cone walk: cycle detection + height + dirty propagation.
	struct ResolveState
	{
		std::vector<std::uint8_t>  color; // 0 white, 1 gray, 2 black
		std::vector<std::uint32_t> height;
		std::vector<std::uint8_t>  dirty;
		std::vector<NodeHandle>	   order;
	};
	[[nodiscard]] std::expected<void, GraphError> visit(std::uint32_t index, ResolveState& state);

	// Run one planned node: within-frame cutoff, execute, CAS completion, stamping.
	void run_node(Node& node, ExecContext& ctx, Revision revision, bool first_time) const;

	std::vector<std::unique_ptr<Node>> m_nodes;
	std::vector<std::unique_ptr<Data>> m_data;
	// Generation per slot, validated by get_node/get_data so a stale handle is
	// rejected rather than silently aliasing a future occupant (UAF/ABA guard). v1 is
	// append-only so these stay 0; they become load-bearing once eviction (§6) bumps a
	// slot's generation on reuse.
	std::vector<std::uint32_t> m_node_generations;
	std::vector<std::uint32_t> m_data_generations;
	std::vector<DataHandle>	   m_pending_sources;
	Revision				   m_revision = 0;
};
} // namespace veng::graph

#endif // VENG_GRAPH_HPP
