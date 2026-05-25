//
// Created by chris on 1/25/26.
//
// L3 Reactive Core — shared primitives. Deliberately Vulkan-agnostic: this header
// (and everything else in veng::graph) must compile and be unit-tested without a
// GPU. See design.md §3 (layering) and §2 (mental model).
//

#ifndef VENG_RENDERGRAPHCOMMON_HPP
#define VENG_RENDERGRAPHCOMMON_HPP

#include <cstdint>
#include <string_view>

namespace veng::graph
{
/// Global logical clock, bumped once per frame after source mutations are applied
/// (design.md §2.2). A value's `changed_at` / a node's `verified_at` are stamps on
/// this clock.
using Revision = std::uint64_t;

/// The asynchronous-execution axis, orthogonal to logical validity (design.md §2.3).
/// Revisions answer "is the output logically up to date?"; this answers "has the
/// async (re)compute landed?".
enum class ExecutionState : std::uint8_t
{
	VALID,		// memoized output computed and current
	PROCESSING, // async (re)compute in flight
	INVALID,	// needs (re)compute
	FAILED,		// last compute errored (design.md §9)
};

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

/// Generation-tagged handle (design.md §1, §L1). Edges reference nodes and data by
/// handle rather than raw pointer: stable across pool reallocation, serializable,
/// and safe against async work outliving a node (no UAF/ABA). The generation lets a
/// future eviction/removal pass invalidate stale handles; v1 only ever appends, so
/// generations stay 0.
template <class Tag>
struct Handle
{
	static constexpr std::uint32_t INVALID_INDEX = ~0U;

	std::uint32_t index		 = INVALID_INDEX;
	std::uint32_t generation = 0;

	[[nodiscard]] constexpr bool valid() const noexcept { return index != INVALID_INDEX; }

	[[nodiscard]] friend constexpr bool operator==(const Handle&, const Handle&) noexcept = default;
};

using NodeHandle = Handle<struct NodeTag>;
using DataHandle = Handle<struct DataTag>;

/// Failure modes of the synchronous resolve/planning pass (design.md §2.6, §8).
enum class GraphError : std::uint8_t
{
	CYCLE_DETECTED, // a back-edge was found while planning; the graph is not a DAG
	INVALID_SINK,	// a requested sink handle does not resolve to live data
};

[[nodiscard]] constexpr std::string_view to_string(GraphError error) noexcept
{
	switch (error)
	{
		case GraphError::CYCLE_DETECTED: return "cycle detected in render graph";
		case GraphError::INVALID_SINK: return "sink handle does not resolve to live data";
	}
	return "unknown graph error";
}

/// Failure modes of a single node's asynchronous execute pass (design.md §9).
enum class ExecError : std::uint8_t
{
	MISSING_INPUT, // an input handle did not resolve to data of the expected type
	NODE_FAILED,   // the node's work reported failure
	WRONG_CONTEXT, // a node needed a context it was not dispatched with (e.g. a GPU node on the CPU path)
};

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
