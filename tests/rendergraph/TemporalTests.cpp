//
// L3 temporal / accumulation tests (design.md §L4 "Temporal / history", §8). These
// exercise the reactive core's demand-refresh hook (Node::needs_refresh) and the
// temporal self-edge pattern — a node reading its own previous output without forming
// a cycle — which together implement the design's headline accumulation showcase:
// "accumulates samples while idle and resets on any input invalidation; converges
// when the scene is still."
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <expected>
#include <memory>
#include <span>
#include <veng/rendergraph/Graph.hpp>

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

void set_source(const Graph& graph, DataHandle handle, int next)
{
	auto* slot = dynamic_cast<ValueData<int>*>(graph.get_data(handle));
	REQUIRE(slot != nullptr);
	slot->set(next);
}

/// The accumulation showcase reduced to the mechanism it stresses. Each demanded
/// frame adds one sample (count + 1), reading its own previous output — a *temporal
/// self-edge*: the output is read via the ExecContext but is deliberately NOT listed
/// in inputs(), so resolve sees no dependency and no cycle. It resets to sample 1
/// whenever its scene input has changed *since it last ran* (using last_run_revision,
/// not "== current revision" — that would miss a change made while undemanded), and
/// stops refreshing (drops out of the plan) once it reaches `max_samples` — i.e.
/// converges and then costs nothing.
class AccumulationNode final : public Node
{
	 public:
	AccumulationNode(DataHandle scene, DataHandle output, int max_samples)
		: m_scene(scene)
		, m_output(output)
		, m_max(max_samples)
	{
	}

	[[nodiscard]] std::span<const DataHandle> inputs() const override { return {&m_scene, 1}; }
	[[nodiscard]] std::span<const DataHandle> outputs() const override { return {&m_output, 1}; }

	[[nodiscard]] bool needs_refresh() const noexcept override { return m_count < m_max; }

	[[nodiscard]] std::expected<bool, ExecError> execute(ExecContext& ctx) override
	{
		auto* scene = dynamic_cast<ValueData<int>*>(ctx.data(m_scene));
		auto* out	= dynamic_cast<ValueData<int>*>(ctx.data(m_output));
		if (scene == nullptr || out == nullptr)
		{
			return std::unexpected(ExecError::MISSING_INPUT);
		}

		// Reset when the scene changed since this node last ran — NOT "== revision",
		// which would miss a change made on a frame the accumulator was undemanded (M7).
		const bool reset = m_count == 0 || scene->changed_at() > last_run_revision();
		const int  prev	 = out->value(); // temporal self-read: last frame's accumulator
		const int  next	 = reset ? 1 : prev + 1;
		m_count			 = next;
		return out->produce(next);
	}

	 private:
	DataHandle m_scene;
	DataHandle m_output;
	int		   m_max;
	int		   m_count = 0;
};

struct AccumulatorFixture
{
	Graph			 graph;
	TypedHandle<int> scene;
	DataHandle		 output;
	NodeHandle		 node;
	InlineScheduler	 scheduler;

	explicit AccumulatorFixture(int max_samples)
		: scene(graph.add_source<int>(0))
		, output(graph.add(std::make_unique<ValueData<int>>(0)))
		, node(graph.add(std::make_unique<AccumulationNode>(static_cast<DataHandle>(scene), output, max_samples)))
	{
		graph.set_producer(output, node);
	}
};
} // namespace

TEST_CASE("an accumulator advances one sample per idle frame, then converges", "[graph][temporal][accumulation]")
{
	AccumulatorFixture fix(4);

	// Frames 1..4 add samples 1,2,3,4 even though the scene never changes — the node
	// stays dirty via needs_refresh, not via any input change.
	for (int sample = 1; sample <= 4; ++sample)
	{
		const auto plan = fix.graph.frame(fix.output, fix.scheduler);
		REQUIRE(plan.has_value());
		REQUIRE(plan->size() == 1);
		REQUIRE(value_of<int>(fix.graph, fix.output) == sample);
	}

	// Frame 5: converged (reached max) -> stops refreshing -> nothing to execute, and
	// the converged value holds. A still scene now costs nothing (design.md §1).
	const auto converged = fix.graph.frame(fix.output, fix.scheduler);
	REQUIRE(converged.has_value());
	REQUIRE(converged->empty());
	REQUIRE(value_of<int>(fix.graph, fix.output) == 4);
}

TEST_CASE("an accumulator resets to the first sample when its input changes", "[graph][temporal][accumulation]")
{
	AccumulatorFixture fix(4);

	for (int i = 0; i < 4; ++i)
	{
		REQUIRE(fix.graph.frame(fix.output, fix.scheduler).has_value());
	}
	REQUIRE(value_of<int>(fix.graph, fix.output) == 4);
	REQUIRE(fix.graph.frame(fix.output, fix.scheduler)->empty()); // converged

	// Any input invalidation resets accumulation the instant it is re-demanded.
	set_source(fix.graph, fix.scene, 99);
	const auto after_reset = fix.graph.frame(fix.output, fix.scheduler);
	REQUIRE(after_reset.has_value());
	REQUIRE(after_reset->size() == 1);
	REQUIRE(value_of<int>(fix.graph, fix.output) == 1); // reset to sample 1

	// ...and it resumes converging from there.
	REQUIRE(fix.graph.frame(fix.output, fix.scheduler).has_value());
	REQUIRE(value_of<int>(fix.graph, fix.output) == 2);
}

TEST_CASE("an accumulator resets even when its input changed while it was undemanded", "[graph][temporal][regression]")
{
	// M7 regression: reset detection must compare against the node's last-run revision,
	// not "== current revision". Here the scene changes on a frame where the accumulator
	// is not demanded; when it is re-demanded it must still reset (a path tracer that
	// missed this would ghost the previous scene onto the new one).
	Graph			 graph;
	auto			 scene	 = graph.add_source<int>(0);
	const DataHandle acc_out = graph.add(std::make_unique<ValueData<int>>(0));
	const NodeHandle acc = graph.add(std::make_unique<AccumulationNode>(static_cast<DataHandle>(scene), acc_out, 8));
	graph.set_producer(acc_out, acc);
	auto			other = graph.add_transform([](const int& x) { return x; }, scene);
	InlineScheduler scheduler;

	// Accumulate two samples while demanded.
	REQUIRE(graph.frame(acc_out, scheduler).has_value());
	REQUIRE(value_of<int>(graph, acc_out) == 1);
	REQUIRE(graph.frame(acc_out, scheduler).has_value());
	REQUIRE(value_of<int>(graph, acc_out) == 2);

	// A frame where the scene changes but only the *other* sink is demanded.
	set_source(graph, scene, 42);
	REQUIRE(graph.frame(other, scheduler).has_value());
	REQUIRE(value_of<int>(graph, acc_out) == 2); // accumulator did not run

	// Re-demand it: the scene changed while it was idle, so it must reset to sample 1.
	REQUIRE(graph.frame(acc_out, scheduler).has_value());
	REQUIRE(value_of<int>(graph, acc_out) == 1); // (was 3 before the M7 fix — missed reset)
}

TEST_CASE("an undemanded accumulator never refreshes", "[graph][temporal][lazy]")
{
	// Demand-driven evaluation still governs refreshing nodes: an accumulator that is
	// not pulled this frame does no work, even though it "wants" to refresh.
	Graph			 graph;
	auto			 scene	 = graph.add_source<int>(0);
	const DataHandle acc_out = graph.add(std::make_unique<ValueData<int>>(0));
	const NodeHandle acc = graph.add(std::make_unique<AccumulationNode>(static_cast<DataHandle>(scene), acc_out, 8));
	graph.set_producer(acc_out, acc);

	// A separate, unrelated sink that we will actually demand.
	auto			other = graph.add_transform([](const int& x) { return x; }, scene);
	InlineScheduler scheduler;

	for (int i = 0; i < 5; ++i)
	{
		REQUIRE(graph.frame(other, scheduler).has_value());
	}

	// The accumulator was never demanded, so it never ran: still the initial value.
	REQUIRE(value_of<int>(graph, acc_out) == 0);
	REQUIRE(graph.get_node(acc)->state() == ExecutionState::INVALID);
}
