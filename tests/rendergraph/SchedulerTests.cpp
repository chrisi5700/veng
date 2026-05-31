//
// L5 job-system tests. Drives the same reactive core through a
// real multi-threaded ThreadPoolScheduler to prove the concurrency machinery — CAS
// completion, atomic wait/notify, and height-batched dispatch — is correct, that a
// fixed pool never deadlocks regardless of graph depth, and that results match the
// deterministic inline scheduler exactly.
//

#include <array>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <veng/rendergraph/Graph.hpp>
#include <veng/rendergraph/ThreadPoolScheduler.hpp>

using namespace veng::graph;

namespace
{
template <class T>
const T& value_of(const Graph& graph, DataHandle handle)
{
	const auto* slot = dynamic_cast<const ValueData<T>*>(graph.get_data(handle));
	REQUIRE(slot != nullptr);
	return slot->value();
}
} // namespace

TEST_CASE("the thread pool produces the same diamond result as the inline scheduler", "[scheduler][diamond]")
{
	auto  shared_runs = std::make_shared<std::atomic<int>>(0);
	Graph graph;
	auto  root	 = graph.add_source<int>(3);
	auto  shared = graph.add_transform(
		[counter = shared_runs](const int& x)
		{
			counter->fetch_add(1);
			return x * x;
		},
		root);
	auto left  = graph.add_transform([](const int& x) { return x + 1; }, shared);
	auto right = graph.add_transform([](const int& x) { return x - 1; }, shared);
	auto sink  = graph.add_transform([](const int& l, const int& r) { return l + r; }, left, right);

	ThreadPoolScheduler scheduler(4);
	const auto			plan = graph.frame(sink, scheduler);

	REQUIRE(plan.has_value());
	REQUIRE(plan->size() == 4);
	REQUIRE(*shared_runs == 1); // the shared dependency resolves exactly once
	REQUIRE(value_of<int>(graph, sink) == 18);
}

TEST_CASE("a wide band of independent nodes runs in parallel and aggregates correctly", "[scheduler][wide]")
{
	Graph graph;
	auto  a = graph.add_source<int>(1);
	auto  b = graph.add_source<int>(2);
	auto  c = graph.add_source<int>(3);
	auto  d = graph.add_source<int>(4);

	// Four independent doublers (all height 1) feed one 4-input sink (height 2).
	auto da	  = graph.add_transform([](const int& x) { return x * 2; }, a);
	auto db	  = graph.add_transform([](const int& x) { return x * 2; }, b);
	auto dc	  = graph.add_transform([](const int& x) { return x * 2; }, c);
	auto dd	  = graph.add_transform([](const int& x) { return x * 2; }, d);
	auto sink = graph.add_transform([](const int& w, const int& x, const int& y, const int& z)
									{ return w + x + y + z; }, da, db, dc, dd);

	ThreadPoolScheduler scheduler(4);
	REQUIRE(graph.frame(sink, scheduler).has_value());
	REQUIRE(value_of<int>(graph, sink) == 2 + 4 + 6 + 8);
}

TEST_CASE("a chain deeper than the pool size does not deadlock", "[scheduler][deep]")
{
	Graph graph;
	auto  source = graph.add_source<int>(0);

	// 32 sequential +1 stages, executed by a 2-thread pool. Height-batched dispatch
	// processes one stage per band, so no task ever blocks on another -> no deadlock.
	TypedHandle<int> current = source;
	constexpr int	 DEPTH	 = 32;
	for (int i = 0; i < DEPTH; ++i)
	{
		current = graph.add_transform([](const int& x) { return x + 1; }, current);
	}

	ThreadPoolScheduler scheduler(2);
	REQUIRE(graph.frame(current, scheduler).has_value());
	REQUIRE(value_of<int>(graph, current) == DEPTH);
}

TEST_CASE("repeated frames under the pool keep caching and cutoff correct", "[scheduler][stress]")
{
	auto  runs = std::make_shared<std::atomic<int>>(0);
	Graph graph;
	auto  a	  = graph.add_source<int>(0);
	auto  out = graph.add_transform(
		[counter = runs](const int& x)
		{
			counter->fetch_add(1);
			return x + 1;
		},
		a);

	ThreadPoolScheduler scheduler(4);

	// First frame computes; many clean frames must execute nothing (cached).
	REQUIRE(graph.frame(out, scheduler).has_value());
	REQUIRE(*runs == 1);
	for (int i = 0; i < 50; ++i)
	{
		const auto plan = graph.frame(out, scheduler);
		REQUIRE(plan.has_value());
		REQUIRE(plan->empty());
	}
	REQUIRE(*runs == 1);

	// Each genuine mutation recomputes exactly once and yields the right value.
	for (int i = 1; i <= 20; ++i)
	{
		dynamic_cast<ValueData<int>*>(graph.get_data(a))->set(i);
		REQUIRE(graph.frame(out, scheduler).has_value());
		REQUIRE(value_of<int>(graph, out) == i + 1);
	}
	REQUIRE(*runs == 21); // 1 initial + 20 mutations
}

TEST_CASE("a throwing node fails the frame instead of hanging or terminating", "[scheduler][error]")
{
	// H1: an exception thrown by a node must not strand the band barrier (frame hang)
	// nor escape a worker thread (std::terminate). It is converted to a FAILED node.
	Graph graph;
	auto  a	  = graph.add_source<int>(1);
	auto  bad = graph.add_transform([](const int&) -> int { throw std::runtime_error("boom"); }, a);

	ThreadPoolScheduler scheduler(4);
	const auto			plan = graph.frame(bad, scheduler); // must return, not hang/terminate

	REQUIRE(plan.has_value());
	REQUIRE(graph.get_node(graph.get_data(bad)->producer())->state() == ExecutionState::FAILED);
}

TEST_CASE("a single-thread pool still drains every band", "[scheduler][single]")
{
	Graph graph;
	auto  a	  = graph.add_source<int>(5);
	auto  mid = graph.add_transform([](const int& x) { return x * 3; }, a);
	auto  out = graph.add_transform([](const int& x) { return x + 7; }, mid);

	ThreadPoolScheduler scheduler(1);
	REQUIRE(graph.frame(out, scheduler).has_value());
	REQUIRE(value_of<int>(graph, out) == (5 * 3) + 7);
}
