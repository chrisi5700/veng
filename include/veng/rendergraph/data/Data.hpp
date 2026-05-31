/**
 * @file
 * @author chris
 * @brief Reactive data slots (graph edges): type-erased @ref veng::graph::Data base, typed
 *        @ref veng::graph::ValueData, and the one-frame-delayed @ref veng::graph::HistoryData feedback edge.
 *
 * Each slot owns its memoized value and the @ref veng::graph::Revision stamp that drives incremental
 * re-evaluation. A @ref veng::graph::Data is either a *source* (no producer; mutated externally via
 * @ref veng::graph::ValueData::set) or the *output* of exactly one @ref veng::graph::Node. Concrete typed storage
 * lives in @ref veng::graph::ValueData so the core graph algorithm stays type-agnostic.
 *
 * @ingroup rendergraph
 */

#ifndef VENG_DATA_HPP
#define VENG_DATA_HPP

#include <concepts>
#include <optional>
#include <utility>
#include <veng/rendergraph/RenderGraphCommon.hpp>

namespace veng::graph
{
class Graph;

/**
 * @brief Type-erased base for a value slot on the graph.
 *
 * A @ref veng::graph::Data is either a *source* (no producer; mutated externally) or the *output*
 * of exactly one @ref veng::graph::Node. Concrete typed storage lives in @ref veng::graph::ValueData so the
 * core algorithm stays type-agnostic.
 *
 * @ingroup rendergraph
 * @see ValueData
 * @see HistoryData
 * @see Graph::add
 */
class Data
{
	 public:
	Data()						 = default;
	Data(const Data&)			 = delete;
	Data& operator=(const Data&) = delete;
	Data(Data&&)				 = delete;
	Data& operator=(Data&&)		 = delete;
	virtual ~Data()				 = default;

	/// @brief Returns the @ref veng::graph::NodeHandle of the node that writes this slot, or an invalid
	///        handle for a source.
	[[nodiscard]] NodeHandle producer() const noexcept { return m_producer; }

	/// @brief Returns true if this slot has no producer (i.e. it is a source).
	[[nodiscard]] bool is_source() const noexcept { return !m_producer.valid(); }

	/// @brief Returns the @ref veng::graph::Revision at which this value last *actually* changed.
	[[nodiscard]] Revision changed_at() const noexcept { return m_changed_at; }

	 protected:
	friend class Graph;

	/**
	 * @brief Apply a queued source mutation at the frame boundary.
	 *
	 * Returns true if and only if the value actually changed — the equality gate that backs
	 * change-cutoff at the source. Non-sources have nothing to do and return false.
	 *
	 * @return Whether the committed value differs from the previously held value.
	 */
	virtual bool commit_pending() { return false; }

	/**
	 * @brief Register this source as dirtied.
	 *
	 * The @ref veng::graph::Graph applies the pending value and bumps the revision at the next frame
	 * boundary. Defined in Graph.cpp because it reaches back into the owning @ref veng::graph::Graph.
	 */
	void notify_mutation();

	/**
	 * @brief Resolve a sibling slot on the owning graph (or nullptr).
	 *
	 * The one place a @ref veng::graph::Data reaches another by handle — used by @ref veng::graph::HistoryData to
	 * read the live slot it shadows. Defined in Graph.cpp (this header only forward-declares
	 * @ref veng::graph::Graph).
	 *
	 * @param handle The @ref veng::graph::DataHandle of the sibling slot to resolve.
	 * @return Pointer to the sibling @ref veng::graph::Data, or nullptr if the handle is invalid.
	 */
	[[nodiscard]] Data* peer(DataHandle handle) const;

	NodeHandle m_producer{};		   ///< Invalid handle means this slot is a source.
	Revision   m_changed_at = 0;	   ///< Frame revision at which the value last changed.
	Graph*	   m_graph		= nullptr; ///< Owning graph (set in @ref veng::graph::Graph::add).
	DataHandle m_self{};			   ///< This slot's own handle (set in @ref veng::graph::Graph::add).
};

/**
 * @brief Typed value storage: a source *and* a produced output in one.
 *
 * Doubles as a source (mutate via @ref set, applied at the next frame boundary) and as a
 * produced output (the producing @ref veng::graph::Node calls @ref produce during execute). Change-cutoff
 * is equality-gated when `T` is `std::equality_comparable`; otherwise every write is
 * conservatively treated as a change.
 *
 * @ingroup rendergraph
 * @tparam T The value type stored in this slot.
 * @see Data
 * @see HistoryData
 * @see Graph::add_source
 */
template <class T>
class ValueData : public Data
{
	 public:
	/**
	 * @brief Construct with an initial value.
	 * @param initial The value this slot holds before any source mutation or node execution.
	 */
	explicit ValueData(T initial)
		: m_value(std::move(initial))
	{
	}

	/// @brief Returns a const reference to the currently memoized value.
	[[nodiscard]] const T& value() const noexcept { return m_value; }

	/**
	 * @brief Queue a source-side mutation, applied at the next frame boundary.
	 *
	 * Keeps the input set immutable within a frame. Calls @ref notify_mutation so the
	 * owning @ref veng::graph::Graph schedules the commit and bumps the revision.
	 *
	 * @param next The new value to queue.
	 */
	void set(T next)
	{
		m_pending = std::move(next);
		notify_mutation();
	}

	/**
	 * @brief Producer-side write during execute.
	 *
	 * Equality-gated: if the new value compares equal to the current one, the slot is
	 * untouched and the node signals no change, which stops the dirty wave from propagating
	 * downstream.
	 *
	 * @param next The value the producing node computed.
	 * @return Whether the slot's value actually changed.
	 */
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

/**
 * @brief A one-frame-delayed *history* (feedback) edge — the engine's temporal primitive.
 *
 * Shadows a live @ref veng::graph::DataHandle but presents to the planner as a *source* (no producer),
 * so wiring it as an input to the very node that produces the live slot forms no cycle: the
 * DAG invariant is preserved while the value flows backwards in time by one frame.
 *
 * **How the one-frame delay falls out for free:** at each frame boundary @ref commit_pending
 * snapshots the live slot's *current* value. Because frame-boundary commits run before the
 * frame executes, the live producer has not run yet this frame — its slot still holds what it
 * produced *last* frame. So during frame F a consumer reads the live value as of frame F-1.
 * For a self-edge (consumer reads `history(O)` and produces `O`) that is exactly `O_{F-1}`,
 * the classic ping-pong / accumulator feedback.
 *
 * **Convergence is automatic:** the snapshot is equality-gated (reusing @ref veng::graph::ValueData::produce),
 * so once the live value stops changing the history stops bumping its `changed_at`, the consumer
 * is no longer dirtied, and the graph goes quiet — a progressive accumulator runs exactly until
 * it converges, then costs nothing. For GPU ping-pong (TAA, history buffers) the value is a
 * `gpu::ImageRef`; the ResourcePool's N-buffering keeps the previous frame's physical copy alive
 * for the read.
 *
 * @ingroup rendergraph
 * @tparam T The value type; must match the live slot's @ref veng::graph::ValueData.
 * @see Graph::add_history
 * @see ValueData
 */
template <class T>
class HistoryData final : public ValueData<T>
{
	 public:
	/**
	 * @brief Construct, linking this history slot to its live counterpart.
	 * @param live   @ref veng::graph::DataHandle of the live slot whose previous-frame value this shadows.
	 * @param initial Value returned by readers on frame 0, before the live slot has ever produced.
	 */
	HistoryData(DataHandle live, T initial)
		: ValueData<T>(std::move(initial))
		, m_live(live)
	{
	}

	 protected:
	/**
	 * @brief Re-snapshot the live slot's current (= previous-frame) value.
	 *
	 * Called at every frame boundary by @ref veng::graph::Graph::resolve. Returns whether the snapshot
	 * changed, so the @ref veng::graph::Graph bumps this slot's `changed_at` and re-demands consumers.
	 *
	 * @return Whether the live slot's value differs from the previously snapshotted value.
	 */
	bool commit_pending() override
	{
		const auto* live = dynamic_cast<const ValueData<T>*>(this->peer(m_live));
		return live != nullptr && this->produce(live->value());
	}

	 private:
	DataHandle m_live; ///< The slot whose previous-frame value this one shadows.
};
} // namespace veng::graph

#endif // VENG_DATA_HPP
