//
// L3 Reactive Core unit tests. The core is Vulkan-free and is the
// priority for coverage, so these tests drive it with a deterministic inline
// scheduler and plain CPU transform nodes — exercising invalidation propagation,
// diamonds, change-cutoff, cycle rejection, lazy evaluation, and CAS completion.
//

#include <array>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <functional>
#include <memory>
#include <string>
#include <veng/rendergraph/Graph.hpp>

using namespace veng::graph;

namespace
{
/// Runs tasks inline (deterministic) while recording how many were dispatched, so
/// tests can assert exactly which nodes ran.
class RecordingScheduler final : public Scheduler
{
	 public:
	void submit(std::function<void()> task) override
	{
		++dispatched;
		task();
	}

	std::size_t dispatched = 0;
};

/// Wraps a transform body so the test can count how many times it actually runs —
/// the lens for asserting that caching and cutoff skip work. The returned generic
/// callable lets `add_transform` deduce the edge types from its inputs.
template <class F>
auto counting(std::shared_ptr<std::atomic<int>> counter, F body)
{
	return [counter = std::move(counter), body = std::move(body)](const auto&... args)
	{
		counter->fetch_add(1);
		return body(args...);
	};
}

template <class T>
const T& value_of(const Graph& graph, DataHandle handle)
{
	const auto* slot = dynamic_cast<const ValueData<T>*>(graph.get_data(handle));
	REQUIRE(slot != nullptr);
	return slot->value();
}

template <class T>
void set_source(const Graph& graph, DataHandle handle, T next)
{
	auto* slot = dynamic_cast<ValueData<T>*>(graph.get_data(handle));
	REQUIRE(slot != nullptr);
	slot->set(std::move(next));
}
} // namespace

TEST_CASE("a freshly built graph executes every demanded node once", "[graph][resolve]")
{
	Graph graph;
	auto  a	  = graph.add_source<int>(2);
	auto  b	  = graph.add_source<int>(3);
	auto  sum = graph.add_transform([](const int& x, const int& y) { return x + y; }, a, b);

	RecordingScheduler scheduler;
	const auto		   plan = graph.frame(sum, scheduler);

	REQUIRE(plan.has_value());
	REQUIRE(plan->size() == 1);
	REQUIRE(scheduler.dispatched == 1);
	REQUIRE(value_of<int>(graph, sum) == 5);
	REQUIRE(graph.get_node(graph.get_data(sum)->producer())->state() == ExecutionState::VALID);
}

TEST_CASE("a clean graph re-presents without re-running anything", "[graph][caching]")
{
	auto  runs = std::make_shared<std::atomic<int>>(0);
	Graph graph;
	auto  a	  = graph.add_source<int>(10);
	auto  out = graph.add_transform(counting(runs, [](const int& x) { return x * 2; }), a);

	RecordingScheduler scheduler;
	REQUIRE(graph.frame(out, scheduler).has_value());
	REQUIRE(*runs == 1);

	// Nothing changed: the second frame plans and executes zero nodes.
	const auto plan = graph.frame(out, scheduler);
	REQUIRE(plan.has_value());
	REQUIRE(plan->empty());
	REQUIRE(*runs == 1);
}

TEST_CASE("mutating a source re-runs only its dependents on the next frame", "[graph][invalidation]")
{
	auto  runs = std::make_shared<std::atomic<int>>(0);
	Graph graph;
	auto  a	  = graph.add_source<int>(1);
	auto  out = graph.add_transform(counting(runs, [](const int& x) { return x + 100; }), a);

	RecordingScheduler scheduler;
	REQUIRE(graph.frame(out, scheduler).has_value());
	REQUIRE(*runs == 1);

	// Source mutation is queued and applied at the next frame boundary.
	set_source<int>(graph, a, 5);
	const auto plan = graph.frame(out, scheduler);

	REQUIRE(plan.has_value());
	REQUIRE(plan->size() == 1);
	REQUIRE(*runs == 2);
	REQUIRE(value_of<int>(graph, out) == 105);
}

TEST_CASE("setting a source to an equal value is gated out", "[graph][cutoff][source]")
{
	auto  runs = std::make_shared<std::atomic<int>>(0);
	Graph graph;
	auto  a	  = graph.add_source<int>(7);
	auto  out = graph.add_transform(counting(runs, [](const int& x) { return x; }), a);

	RecordingScheduler scheduler;
	REQUIRE(graph.frame(out, scheduler).has_value());
	REQUIRE(*runs == 1);

	// Equality-gated source: same value => no change => no recompute.
	set_source<int>(graph, a, 7);
	const auto plan = graph.frame(out, scheduler);
	REQUIRE(plan.has_value());
	REQUIRE(plan->empty());
	REQUIRE(*runs == 1);
}

TEST_CASE("change-cutoff stops a no-op recompute from rippling downstream", "[graph][cutoff]")
{
	auto upstream_runs	 = std::make_shared<std::atomic<int>>(0);
	auto downstream_runs = std::make_shared<std::atomic<int>>(0);

	Graph graph;
	auto  a = graph.add_source<int>(4);
	// abs() collapses +n and -n to the same output: changing the sign re-runs the
	// upstream node, but its output value is unchanged, so the consumer is cut off.
	auto mid = graph.add_transform(counting(upstream_runs, [](const int& x) { return x < 0 ? -x : x; }), a);
	auto out = graph.add_transform(counting(downstream_runs, [](const int& x) { return x + 1; }), mid);

	RecordingScheduler scheduler;
	REQUIRE(graph.frame(out, scheduler).has_value());
	REQUIRE(*upstream_runs == 1);
	REQUIRE(*downstream_runs == 1);

	// 4 -> -4: |-4| == 4, so mid recomputes (it is in the dirty cone) but its value
	// is unchanged, so its stamp is not bumped.
	set_source<int>(graph, a, -4);
	REQUIRE(graph.frame(out, scheduler).has_value());
	REQUIRE(*upstream_runs == 2);

	// The NEXT frame is where cutoff pays off: mid did not bump its stamp, so the
	// downstream node drops out of the plan entirely.
	const auto plan = graph.frame(out, scheduler);
	REQUIRE(plan.has_value());
	REQUIRE(plan->empty());
	REQUIRE(*downstream_runs == 1);
}

TEST_CASE("a diamond resolves its shared dependency once and orders by height", "[graph][diamond]")
{
	auto  shared_runs = std::make_shared<std::atomic<int>>(0);
	Graph graph;
	auto  root	 = graph.add_source<int>(3);
	auto  shared = graph.add_transform(counting(shared_runs, [](const int& x) { return x * x; }), root);
	auto  left	 = graph.add_transform([](const int& x) { return x + 1; }, shared);
	auto  right	 = graph.add_transform([](const int& x) { return x - 1; }, shared);
	auto  sink	 = graph.add_transform([](const int& l, const int& r) { return l + r; }, left, right);

	RecordingScheduler scheduler;
	const auto		   plan = graph.frame(sink, scheduler);

	REQUIRE(plan.has_value());
	REQUIRE(plan->size() == 4);				   // shared, left, right, sink
	REQUIRE(*shared_runs == 1);				   // resolved once despite two consumers
	REQUIRE(value_of<int>(graph, sink) == 18); // (9+1) + (9-1)

	// Height ordering: the shared dependency runs first, the sink last.
	const auto nodes = plan->nodes();
	REQUIRE(nodes.front() == graph.get_data(shared)->producer());
	REQUIRE(nodes.back() == graph.get_data(sink)->producer());
}

TEST_CASE("a cycle is rejected at resolve time", "[graph][cycle]")
{
	Graph			 graph;
	const DataHandle d0 = graph.add(std::make_unique<ValueData<int>>(0));
	const DataHandle d1 = graph.add(std::make_unique<ValueData<int>>(0));

	const auto passthrough = [](DataHandle in, DataHandle out)
	{ return std::make_unique<TransformNode<int(int)>>([](const int& x) { return x; }, std::array{in}, out); };

	// n0 reads d1, writes d0; n1 reads d0, writes d1 — a 2-cycle d0->n1->d1->n0->d0.
	const NodeHandle n0 = graph.add(passthrough(d1, d0));
	const NodeHandle n1 = graph.add(passthrough(d0, d1));
	graph.set_producer(d0, n0);
	graph.set_producer(d1, n1);

	const std::array sinks{d0};
	const auto		 plan = graph.resolve(sinks);
	REQUIRE_FALSE(plan.has_value());
	REQUIRE(plan.error() == GraphError::CYCLE_DETECTED);
}

TEST_CASE("get_node/get_data reject handles with a stale generation", "[graph][handles]")
{
	Graph	   graph;
	const auto a = graph.add_source<int>(1);

	// A correctly-generationed handle resolves; the same index with a bumped
	// generation must not (the UAF/ABA guard for future eviction).
	REQUIRE(graph.get_data(a.handle) != nullptr);
	REQUIRE(graph.get_data(DataHandle{a.handle.index, a.handle.generation + 1}) == nullptr);
	REQUIRE(graph.get_node(NodeHandle{0, 99}) == nullptr);
}

TEST_CASE("resolving an invalid sink reports INVALID_SINK", "[graph][error]")
{
	Graph			 graph;
	const std::array sinks{DataHandle{}};
	const auto		 plan = graph.resolve(sinks);
	REQUIRE_FALSE(plan.has_value());
	REQUIRE(plan.error() == GraphError::INVALID_SINK);
}

TEST_CASE("a node fed the wrong input type lands in FAILED without crashing", "[graph][error]")
{
	Graph			 graph;
	auto			 text = graph.add_source<std::string>("hello"); // ValueData<std::string>
	const DataHandle out  = graph.add(std::make_unique<ValueData<int>>(0));

	// Deliberately wire a string slot into an int transform via the low-level API.
	auto			 node = std::make_unique<TransformNode<int(int)>>([](const int& x) { return x; },
																	  std::array{static_cast<DataHandle>(text)}, out);
	const NodeHandle nh	  = graph.add(std::move(node));
	graph.set_producer(out, nh);

	RecordingScheduler scheduler;
	const auto		   plan = graph.frame(out, scheduler);

	REQUIRE(plan.has_value()); // planning does not type-check
	REQUIRE(plan->size() == 1);
	REQUIRE(graph.get_node(nh)->state() == ExecutionState::FAILED); // execute caught the mismatch
}

TEST_CASE("an undemanded dirty subtree does not execute (lazy evaluation)", "[graph][lazy]")
{
	auto used_runs	 = std::make_shared<std::atomic<int>>(0);
	auto unused_runs = std::make_shared<std::atomic<int>>(0);

	Graph graph;
	auto  a		 = graph.add_source<int>(1);
	auto  used	 = graph.add_transform(counting(used_runs, [](const int& x) { return x; }), a);
	auto  unused = graph.add_transform(counting(unused_runs, [](const int& x) { return x; }), a);
	(void)unused;

	RecordingScheduler scheduler;
	// Demand only `used`; `unused` is dirty (never computed) but not demanded.
	const auto plan = graph.frame(used, scheduler);

	REQUIRE(plan.has_value());
	REQUIRE(plan->size() == 1);
	REQUIRE(*used_runs == 1);
	REQUIRE(*unused_runs == 0);
}

TEST_CASE("revision advances once per frame and stamps changed data", "[graph][revisions]")
{
	Graph graph;
	auto  a	  = graph.add_source<int>(1);
	auto  out = graph.add_transform([](const int& x) { return x; }, a);

	RecordingScheduler scheduler;
	REQUIRE(graph.current_revision() == 0);
	REQUIRE(graph.frame(out, scheduler).has_value());
	REQUIRE(graph.current_revision() == 1);

	set_source<int>(graph, a, 2);
	REQUIRE(graph.frame(out, scheduler).has_value());
	REQUIRE(graph.current_revision() == 2);
	REQUIRE(graph.get_data(a)->changed_at() == 2);
	REQUIRE(graph.get_data(out)->changed_at() == 2);
}

TEST_CASE("non-comparable outputs are treated as always-changed", "[graph][cutoff]")
{
	// A type with no operator== cannot be equality-gated, so every recompute counts
	// as a change.
	struct Opaque
	{
		int payload = 0;
	};

	auto  downstream_runs = std::make_shared<std::atomic<int>>(0);
	Graph graph;
	auto  a = graph.add_source<int>(1);
	// `mid`'s output is a constant Opaque{0} regardless of input. With a comparable
	// type this would cut off the downstream node when the source changes; because
	// Opaque has no operator==, every recompute counts as a change instead.
	auto mid = graph.add_transform([](const int&) { return Opaque{0}; }, a);
	auto out = graph.add_transform(
		[counter = downstream_runs](const Opaque& o)
		{
			counter->fetch_add(1);
			return o.payload;
		},
		mid);

	RecordingScheduler scheduler;
	REQUIRE(graph.frame(out, scheduler).has_value());
	REQUIRE(*downstream_runs == 1);

	// Source changes => mid recomputes to the same Opaque{0}, but the lack of a cheap
	// comparison forces the change to propagate, so the downstream node runs again.
	set_source<int>(graph, a, 2);
	REQUIRE(graph.frame(out, scheduler).has_value());
	REQUIRE(*downstream_runs == 2);
}

TEST_CASE("a node dirtied while undemanded recomputes when next demanded", "[graph][cutoff][regression]")
{
	// Regression for C1: the execute-time cutoff baseline must match resolve's
	// dirtiness test, or a node dirtied on a frame where it was not demanded gets
	// planned-then-skipped and is left permanently stale.
	Graph graph;
	auto  src = graph.add_source<int>(1);
	auto  a	  = graph.add_transform([](const int& x) { return x * 10; }, src);	// sink A
	auto  b	  = graph.add_transform([](const int& x) { return x + 100; }, src); // sink B

	RecordingScheduler scheduler;
	REQUIRE(graph.frame(a, scheduler).has_value());
	REQUIRE(value_of<int>(graph, a) == 10);

	// Mutate the shared source, then demand ONLY b. `a` is now dirty but undemanded.
	set_source<int>(graph, src, 2);
	REQUIRE(graph.frame(b, scheduler).has_value());
	REQUIRE(value_of<int>(graph, b) == 102);

	// Re-demand a: it must reflect the mutation it missed, not the stale cached value.
	const auto plan = graph.frame(a, scheduler);
	REQUIRE(plan.has_value());
	REQUIRE(plan->size() == 1); // a was replanned and actually ran
	REQUIRE(value_of<int>(graph, a) == 20);
}

TEST_CASE("a node dirty across several undemanded frames still catches up", "[graph][cutoff][regression]")
{
	auto  runs = std::make_shared<std::atomic<int>>(0);
	Graph graph;
	auto  src	  = graph.add_source<int>(0);
	auto  tracked = graph.add_transform(counting(runs, [](const int& x) { return x + 1; }), src);
	auto  other	  = graph.add_transform([](const int& x) { return x; }, src);

	RecordingScheduler scheduler;
	REQUIRE(graph.frame(tracked, scheduler).has_value());
	REQUIRE(*runs == 1);
	REQUIRE(value_of<int>(graph, tracked) == 1);

	// Several frames mutate src but demand only `other`; `tracked` stays undemanded.
	for (int i = 1; i <= 3; ++i)
	{
		set_source<int>(graph, src, i);
		REQUIRE(graph.frame(other, scheduler).has_value());
	}
	REQUIRE(*runs == 1); // tracked never ran while undemanded

	// Demanding tracked again must catch it up to the latest source value (3 -> 4).
	REQUIRE(graph.frame(tracked, scheduler).has_value());
	REQUIRE(value_of<int>(graph, tracked) == 4);
	REQUIRE(*runs == 2);
}

TEST_CASE("to_dot emits nodes and edges for the built graph", "[graph][tooling]")
{
	Graph graph;
	auto  a	  = graph.add_source<int>(1);
	auto  out = graph.add_transform([](const int& x) { return x + 1; }, a);
	(void)out;

	const std::string dot = graph.to_dot();
	REQUIRE(dot.starts_with("digraph veng"));
	REQUIRE(dot.find("->") != std::string::npos);
	REQUIRE(dot.find("INVALID") != std::string::npos); // node state rendered before any frame
}
