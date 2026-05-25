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

	NodeHandle m_producer{}; // invalid => source
	Revision   m_changed_at = 0;
	Graph*	   m_graph		= nullptr; // owning graph (set in Graph::add)
	DataHandle m_self{};			   // this slot's own handle (set in Graph::add)
};

/// Typed value storage. Doubles as both a source (use `set`) and a produced output
/// (the producing node calls `produce` during execute). Change-cutoff is
/// equality-gated when `T` is comparable; otherwise every write counts as a change.
template <class T>
class ValueData final : public Data
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
} // namespace veng::graph

#endif // VENG_DATA_HPP
