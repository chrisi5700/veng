//
// Created by chris on 1/24/26.
//
// L3 Reactive Core — Data (a value slot / graph edge). Owns its memoized value and
// the revision stamp that drives incremental re-evaluation (design.md §2.2, §L3).
//

#ifndef VENG_DATA_HPP
#define VENG_DATA_HPP

#include <concepts>
#include <optional>
#include <utility>
#include <veng/rendergraph/RenderGraphCommon.hpp>

namespace veng::graph
{
class Graph;

/// Type-erased base for a value slot on the graph. A `Data` is either a *source*
/// (no producer; mutated externally) or the *output* of exactly one node. Concrete
/// typed storage lives in `ValueData<T>` so the core algorithm stays type-agnostic.
class Data
{
	 public:
	Data()						 = default;
	Data(const Data&)			 = delete;
	Data& operator=(const Data&) = delete;
	Data(Data&&)				 = delete;
	Data& operator=(Data&&)		 = delete;
	virtual ~Data()				 = default;

	/// The node that writes this slot, or an invalid handle for a source.
	[[nodiscard]] NodeHandle producer() const noexcept { return m_producer; }
	[[nodiscard]] bool		 is_source() const noexcept { return !m_producer.valid(); }

	/// Revision at which this value last *actually* changed (design.md §2.2).
	[[nodiscard]] Revision changed_at() const noexcept { return m_changed_at; }

	 protected:
	friend class Graph;

	/// Apply a queued source mutation at the frame boundary (design.md §2.5).
	/// Returns true iff the value actually changed — the equality gate that backs
	/// change-cutoff at the source (design.md §2.4). Non-sources have nothing to do.
	virtual bool commit_pending() { return false; }

	/// Register this source as dirtied; the Graph applies the pending value and
	/// bumps the revision at the next frame boundary. Defined in Graph.cpp because
	/// it reaches back into the owning Graph.
	void notify_mutation();

	/// Resolve a sibling slot on the owning graph (or nullptr). The one place a `Data`
	/// reaches another by handle — used by `HistoryData` to read the live slot it shadows.
	/// Defined in Graph.cpp (this header only forward-declares `Graph`).
	[[nodiscard]] Data* peer(DataHandle handle) const;

	NodeHandle m_producer{}; // invalid => source
	Revision   m_changed_at = 0;
	Graph*	   m_graph		= nullptr; // owning graph (set in Graph::add)
	DataHandle m_self{};			   // this slot's own handle (set in Graph::add)
};

/// Typed value storage. Doubles as both a source (use `set`) and a produced output
/// (the producing node calls `produce` during execute). Change-cutoff is
/// equality-gated when `T` is comparable; otherwise every write counts as a change.
template <class T>
class ValueData : public Data
{
	 public:
	explicit ValueData(T initial)
		: m_value(std::move(initial))
	{
	}

	[[nodiscard]] const T& value() const noexcept { return m_value; }

	/// Source-side mutation. Queued and applied at the next frame boundary so the
	/// input set is immutable within a frame (design.md §2.5).
	void set(T next)
	{
		m_pending = std::move(next);
		notify_mutation();
	}

	/// Producer-side write during execute. Returns whether the value changed, which
	/// the node forwards to drive change-cutoff (design.md §2.4).
	[[nodiscard]] bool produce(T next)
	{
		if (values_equal(m_value, next))
		{
			return false;
		}
		m_value = std::move(next);
		return true;
	}

	 protected:
	bool commit_pending() override
	{
		if (!m_pending.has_value())
		{
			return false;
		}
		T next = std::move(*m_pending);
		m_pending.reset();
		if (values_equal(m_value, next))
		{
			return false;
		}
		m_value = std::move(next);
		return true;
	}

	 private:
	static bool values_equal(const T& lhs, const T& rhs)
	{
		if constexpr (std::equality_comparable<T>)
		{
			return lhs == rhs;
		}
		else
		{
			return false; // no cheap comparison: conservatively treat as changed
		}
	}

	T				 m_value;
	std::optional<T> m_pending{};
};

/// A one-frame-delayed *history* (feedback) edge — the engine's temporal primitive
/// (design.md §L4 temporal / accumulation). It shadows a live slot but presents to the
/// planner as a *source* (no producer), so wiring it as an input to the very node that
/// produces the live slot forms no cycle: the DAG invariant is preserved while the value
/// flows backwards in time by one frame.
///
/// How the one-frame delay falls out for free: at each frame boundary `commit_pending`
/// snapshots the live slot's *current* value. Because frame-boundary commits run before the
/// frame executes, the live producer has not run yet this frame — its slot still holds what
/// it produced *last* frame. So during frame F a consumer reads the live value as of frame
/// F-1. For a self-edge (consumer reads `history(O)` and produces `O`) that is exactly
/// `O_{F-1}`, the classic ping-pong / accumulator feedback.
///
/// Convergence is automatic and needs no `needs_refresh` hook: the snapshot is equality-gated
/// (it reuses `produce`), so once the live value stops changing the history stops bumping its
/// `changed_at`, the consumer is no longer dirtied, and the graph goes quiet — a progressive
/// accumulator runs exactly until it converges, then costs nothing. For GPU ping-pong (TAA,
/// history buffers) the value is a `gpu::ImageRef`; the ResourcePool's N-buffering keeps the
/// previous frame's physical copy alive for the read (the consumer still `consume`s it).
template <class T>
class HistoryData final : public ValueData<T>
{
	 public:
	HistoryData(DataHandle live, T initial)
		: ValueData<T>(std::move(initial))
		, m_live(live)
	{
	}

	 protected:
	/// Re-snapshot the live slot's current (= previous-frame) value. Returns whether it
	/// changed, so the Graph bumps this slot's `changed_at` and re-demands consumers.
	bool commit_pending() override
	{
		const auto* live = dynamic_cast<const ValueData<T>*>(this->peer(m_live));
		return live != nullptr && this->produce(live->value());
	}

	 private:
	DataHandle m_live; // the slot whose previous-frame value this one shadows
};
} // namespace veng::graph

#endif // VENG_DATA_HPP
