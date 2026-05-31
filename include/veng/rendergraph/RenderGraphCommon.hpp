/**
 * @file
 * @author chris
 * @brief Shared primitives for the reactive render graph core: revision clock,
 *        execution state, generation-tagged handles, and error enumerations.
 *
 * Deliberately Vulkan-agnostic: this header (and everything else in `veng::graph`)
 * must compile and be unit-tested without a GPU. All graph-layer types that refer
 * to nodes or data slots do so through @ref veng::graph::NodeHandle / @ref veng::graph::DataHandle rather than
 * raw pointers, keeping the graph stable across pool reallocation and safe against
 * async work outliving a node.
 *
 * @ingroup rendergraph
 */

#ifndef VENG_RENDERGRAPHCOMMON_HPP
#define VENG_RENDERGRAPHCOMMON_HPP

#include <cstdint>
#include <string_view>

namespace veng::graph
{
/**
 * @brief Global logical clock, bumped once per frame after source mutations are applied.
 *
 * A @ref veng::graph::Data slot's `changed_at` and a @ref veng::graph::Node's `verified_at` are both stamps on
 * this clock. Comparing them is the core mechanism that decides whether a node's output
 * is still current without re-executing it.
 *
 * @ingroup rendergraph
 */
using Revision = std::uint64_t;

/**
 * @brief The asynchronous-execution axis of a node, orthogonal to logical validity.
 *
 * Revisions answer "is the output logically up to date?"; @ref veng::graph::ExecutionState answers
 * "has the async (re)compute landed?". The two axes are independent: a node can be
 * logically current (`verified_at == revision`) while its async work is still
 * @ref veng::graph::ExecutionState::PROCESSING.
 *
 * @ingroup rendergraph
 */
enum class ExecutionState : std::uint8_t
{
	VALID,		///< Memoized output computed and current.
	PROCESSING, ///< Async (re)compute in flight.
	INVALID,	///< Needs (re)compute.
	FAILED,		///< Last compute errored.
};

/**
 * @brief Stringify an @ref veng::graph::ExecutionState for logging and GraphViz export.
 * @param state The state to render.
 * @return A string-view constant naming the state.
 */
[[nodiscard]] constexpr std::string_view to_string(ExecutionState state) noexcept
{
	switch (state)
	{
		case ExecutionState::VALID: return "VALID";
		case ExecutionState::PROCESSING: return "PROCESSING";
		case ExecutionState::INVALID: return "INVALID";
		case ExecutionState::FAILED: return "FAILED";
	}
	return "UNKNOWN";
}

/**
 * @brief Generation-tagged, type-safe handle to a node or data slot in the graph.
 *
 * Edges reference nodes and data by handle rather than raw pointer: handles are stable
 * across pool reallocation, serializable, and safe against async work outliving a node
 * (no UAF/ABA). The generation field lets a future eviction/removal pass invalidate
 * stale handles; v1 only ever appends, so generations remain 0 throughout the slot's
 * lifetime.
 *
 * @ingroup rendergraph
 * @tparam Tag A phantom type that makes @ref veng::graph::NodeHandle and @ref veng::graph::DataHandle
 *             distinct types at compile time.
 * @see NodeHandle
 * @see DataHandle
 */
template <class Tag>
struct Handle
{
	static constexpr std::uint32_t INVALID_INDEX = ~0U; ///< Sentinel for an unset handle.

	std::uint32_t index		 = INVALID_INDEX; ///< Slot index within the owning pool.
	std::uint32_t generation = 0;			  ///< Generation counter for ABA/UAF protection.

	/// @brief Returns true if this handle refers to a real slot (index is not the sentinel).
	[[nodiscard]] constexpr bool valid() const noexcept { return index != INVALID_INDEX; }

	[[nodiscard]] friend constexpr bool operator==(const Handle&, const Handle&) noexcept = default;
};

/// @brief Handle to a @ref veng::graph::Node slot in the owning @ref veng::graph::Graph.
/// @ingroup rendergraph
using NodeHandle = Handle<struct NodeTag>;

/// @brief Handle to a @ref veng::graph::Data slot in the owning @ref veng::graph::Graph.
/// @ingroup rendergraph
using DataHandle = Handle<struct DataTag>;

/**
 * @brief Failure modes of the synchronous resolve/planning pass (@ref veng::graph::Graph::resolve).
 * @ingroup rendergraph
 * @see Graph::resolve
 * @see GraphError::CYCLE_DETECTED
 * @see GraphError::INVALID_SINK
 */
enum class GraphError : std::uint8_t
{
	CYCLE_DETECTED, ///< A back-edge was found while planning; the graph is not a DAG.
	INVALID_SINK,	///< A requested sink handle does not resolve to live data.
};

/**
 * @brief Stringify a @ref veng::graph::GraphError for error reporting.
 * @param error The error to render.
 * @return A descriptive string-view constant.
 */
[[nodiscard]] constexpr std::string_view to_string(GraphError error) noexcept
{
	switch (error)
	{
		case GraphError::CYCLE_DETECTED: return "cycle detected in render graph";
		case GraphError::INVALID_SINK: return "sink handle does not resolve to live data";
	}
	return "unknown graph error";
}

/**
 * @brief Failure modes of a single node's asynchronous execute pass (@ref veng::graph::Node::execute).
 * @ingroup rendergraph
 * @see Node::execute
 * @see ExecError::MISSING_INPUT
 * @see ExecError::NODE_FAILED
 * @see ExecError::WRONG_CONTEXT
 */
enum class ExecError : std::uint8_t
{
	MISSING_INPUT, ///< An input handle did not resolve to data of the expected type.
	NODE_FAILED,   ///< The node's work reported failure.
	WRONG_CONTEXT, ///< A node needed a context it was not dispatched with (e.g. a GPU node on the CPU path).
};

/**
 * @brief Stringify an @ref veng::graph::ExecError for error reporting.
 * @param error The error to render.
 * @return A descriptive string-view constant.
 */
[[nodiscard]] constexpr std::string_view to_string(ExecError error) noexcept
{
	switch (error)
	{
		case ExecError::MISSING_INPUT: return "node input did not resolve to the expected data";
		case ExecError::NODE_FAILED: return "node execution failed";
		case ExecError::WRONG_CONTEXT: return "node dispatched without the execution context it requires";
	}
	return "unknown execution error";
}
} // namespace veng::graph

#endif // VENG_RENDERGRAPHCOMMON_HPP
