//
// Created by chris on 5/25/26.
//
// L3 Reactive Core implementation. See Graph.hpp and design.md §2.
//

#include <algorithm>
#include <atomic>
#include <format>
#include <ranges>
#include <veng/rendergraph/Graph.hpp>

namespace veng::graph
{
namespace
{
constexpr std::uint8_t WHITE = 0;
constexpr std::uint8_t GRAY	 = 1;
constexpr std::uint8_t BLACK = 2;

/// The concrete ExecContext handed to nodes during execute: resolves handles
/// against the graph and reports the frame revision (design.md §L3).
class ExecContextImpl final : public ExecContext
{
	 public:
	ExecContextImpl(const Graph& graph, Revision revision)
		: m_graph(&graph)
		, m_revision(revision)
	{
	}

	[[nodiscard]] Data*	   data(DataHandle handle) const override { return m_graph->get_data(handle); }
	[[nodiscard]] Revision revision() const noexcept override { return m_revision; }

	 private:
	const Graph* m_graph;
	Revision	 m_revision;
};
} // namespace

void Data::notify_mutation()
{
	if (m_graph != nullptr)
	{
		m_graph->mutate_source(m_self);
	}
}

NodeHandle Graph::add(std::unique_ptr<Node> node)
{
	const auto index = static_cast<std::uint32_t>(m_nodes.size());
	m_nodes.push_back(std::move(node));
	m_node_generations.push_back(0);
	return NodeHandle{index, 0};
}

DataHandle Graph::add(std::unique_ptr<Data> data)
{
	const auto		 index = static_cast<std::uint32_t>(m_data.size());
	const DataHandle handle{index, 0};
	data->m_graph = this;
	data->m_self  = handle;
	m_data.push_back(std::move(data));
	m_data_generations.push_back(0);
	return handle;
}

void Graph::set_producer(DataHandle data, NodeHandle producer)
{
	if (Data* slot = get_data(data); slot != nullptr)
	{
		slot->m_producer = producer;
	}
}

Node* Graph::get_node(NodeHandle handle) const
{
	if (!handle.valid() || handle.index >= m_nodes.size() || m_node_generations[handle.index] != handle.generation)
	{
		return nullptr;
	}
	return m_nodes[handle.index].get();
}

Data* Graph::get_data(DataHandle handle) const
{
	if (!handle.valid() || handle.index >= m_data.size() || m_data_generations[handle.index] != handle.generation)
	{
		return nullptr;
	}
	return m_data[handle.index].get();
}

void Graph::mutate_source(DataHandle handle)
{
	if (std::ranges::find(m_pending_sources, handle) == m_pending_sources.end())
	{
		m_pending_sources.push_back(handle);
	}
}

std::expected<void, GraphError> Graph::visit(std::uint32_t index, ResolveState& state)
{
	if (state.color[index] == BLACK)
	{
		return {};
	}
	if (state.color[index] == GRAY)
	{
		return std::unexpected(GraphError::CYCLE_DETECTED); // back-edge => not a DAG
	}
	state.color[index] = GRAY;

	Node* node = m_nodes[index].get();
	// Dirty if never computed/failed, or if it asks to keep refreshing (a progressive
	// accumulator that has not converged — design.md §L4). Input-driven dirtiness is
	// added in the loop below.
	bool		  dirty	 = node->state() != ExecutionState::VALID || node->needs_refresh();
	std::uint32_t height = 0;

	for (const DataHandle input : node->inputs())
	{
		const Data* slot = get_data(input);
		if (slot == nullptr)
		{
			return std::unexpected(GraphError::INVALID_SINK); // dangling edge
		}
		// A source mutated this frame, or an upstream output that changed in a
		// previous frame we never re-verified, shows up as a newer stamp.
		if (slot->changed_at() > node->m_verified_at)
		{
			dirty = true;
		}
		const NodeHandle producer = slot->producer();
		if (producer.valid())
		{
			if (auto walked = visit(producer.index, state); !walked)
			{
				return walked;
			}
			height = std::max(height, state.height[producer.index] + 1);
			// A dependency recomputing this frame may not have bumped its stamp yet
			// (cutoff is decided at execute), so propagate dirtiness explicitly.
			if (state.dirty[producer.index] != 0)
			{
				dirty = true;
			}
		}
	}

	state.color[index]	= BLACK;
	state.height[index] = height;
	state.dirty[index]	= dirty ? 1 : 0;
	// NOTE: verified_at is advanced by execute (run_node), never here. Advancing it
	// during planning would mark a node "up to date at R" before it actually runs,
	// which lets the execute-time cutoff drop a genuinely dirty node and leave it
	// permanently stale (the C1 demand-switching bug).
	if (dirty)
	{
		state.order.push_back(NodeHandle{index, 0});
	}
	return {};
}

std::expected<FramePlan, GraphError> Graph::resolve(std::span<const DataHandle> sinks)
{
	// 1. Apply queued source mutations, then bump the revision once (design.md §2.5/§2.6).
	std::vector<DataHandle> changed_sources;
	for (const DataHandle handle : m_pending_sources)
	{
		if (Data* slot = get_data(handle); slot != nullptr && slot->commit_pending())
		{
			changed_sources.push_back(handle);
		}
	}
	m_pending_sources.clear();
	++m_revision;
	for (const DataHandle handle : changed_sources)
	{
		get_data(handle)->m_changed_at = m_revision;
	}

	// 2. Walk the demanded cone from each sink (design.md §2.2).
	const std::size_t count = m_nodes.size();
	ResolveState	  state{
		.color	= std::vector<std::uint8_t>(count, WHITE),
		.height = std::vector<std::uint32_t>(count, 0),
		.dirty	= std::vector<std::uint8_t>(count, 0),
		.order	= {},
	};

	for (const DataHandle sink : sinks)
	{
		const Data* slot = get_data(sink);
		if (slot == nullptr)
		{
			return std::unexpected(GraphError::INVALID_SINK);
		}
		const NodeHandle producer = slot->producer();
		if (producer.valid())
		{
			if (auto walked = visit(producer.index, state); !walked)
			{
				return std::unexpected(walked.error());
			}
		}
	}

	// 3. Order the plan by topological height so deps precede consumers (diamonds),
	//    and so execute can dispatch independent nodes one height band at a time.
	std::ranges::stable_sort(state.order, [&](NodeHandle lhs, NodeHandle rhs)
							 { return state.height[lhs.index] < state.height[rhs.index]; });

	FramePlan plan;
	plan.m_revision = m_revision;
	plan.m_heights.reserve(state.order.size());
	for (const NodeHandle node : state.order)
	{
		plan.m_heights.push_back(state.height[node.index]);
	}
	plan.m_order = std::move(state.order);
	return plan;
}

void Graph::run_node(Node& node, ExecContext& ctx, Revision revision, bool first_time) const
{
	// Change-cutoff (design.md §2.4): skip a node iff none of its inputs have changed
	// since the revision at which this node last produced (m_verified_at). The
	// baseline MUST match resolve's dirtiness test (input.changed_at > verified_at) —
	// using "changed this exact frame" instead silently drops an update that resolve
	// planned because the node was dirty-but-undemanded on an earlier frame (C1).
	// Inputs live in strictly lower height bands, already completed at the execute
	// barrier, so their stamps are settled and need no per-node wait here.
	// A refreshing node (progressive accumulator) must re-run every demanded frame
	// even with unchanged inputs, so it is never cut off (design.md §L4).
	bool inputs_current = !first_time && !node.needs_refresh();
	for (const DataHandle input : node.inputs())
	{
		if (const Data* slot = get_data(input); slot != nullptr && slot->changed_at() > node.m_verified_at)
		{
			inputs_current = false;
		}
	}

	ExecutionState expected = ExecutionState::PROCESSING;
	if (inputs_current)
	{
		// Confirmed still up to date at this revision without recomputing — this is
		// what stops sub-pixel jitter or a backdated value from rippling downstream.
		node.m_verified_at = revision;
		node.m_state.compare_exchange_strong(expected, ExecutionState::VALID, std::memory_order_acq_rel,
											 std::memory_order_relaxed);
		node.m_state.notify_all();
		return;
	}

	std::expected<bool, ExecError> result;
	try
	{
		result = node.execute(ctx);
	}
	catch (...)
	{
		// A throwing node must not strand the band barrier (frame hang) or escape the
		// worker (std::terminate). Treat any exception as failure (design.md §9, H1).
		result = std::unexpected(ExecError::NODE_FAILED);
	}

	// Completion is a CAS, not a store, so re-invalidation during execute is detected
	// rather than clobbered (design.md §L3, fixes the prototype's lost-update race).
	if (result.has_value())
	{
		if (*result)
		{
			// Output actually changed: stamp it so consumers re-evaluate next.
			for (const DataHandle output : node.outputs())
			{
				get_data(output)->m_changed_at = revision;
			}
		}
		node.m_verified_at = revision; // brought up to date this frame
		node.m_state.compare_exchange_strong(expected, ExecutionState::VALID, std::memory_order_acq_rel,
											 std::memory_order_relaxed);
	}
	else
	{
		// Leave m_verified_at untouched so a FAILED node stays detectably dirty and is
		// replanned next frame (retry; see M2 in feedback for the longer-term policy).
		node.m_state.compare_exchange_strong(expected, ExecutionState::FAILED, std::memory_order_acq_rel,
											 std::memory_order_relaxed);
	}
	node.m_state.notify_all();
}

void Graph::execute(const FramePlan& plan, Scheduler& scheduler)
{
	const Revision revision = plan.revision();
	// Shared so tasks outlive this call even under an asynchronous scheduler.
	const auto ctx	 = std::make_shared<ExecContextImpl>(*this, revision);
	const auto nodes = plan.nodes();

	// Capture prior validity for the whole plan up front: a node that has never
	// produced a valid output must always run; a previously-valid one can be cut off
	// if its inputs turn out unchanged. Lower bands never touch higher-band node
	// state, so this snapshot is stable across the band loop.
	std::vector<std::uint8_t> was_valid(nodes.size());
	for (std::size_t i = 0; i < nodes.size(); ++i)
	{
		was_valid[i] = (get_node(nodes[i])->state() == ExecutionState::VALID) ? 1 : 0;
	}

	// Height-batched dispatch (design.md §L5). The plan is sorted by ascending
	// height, so equal-height nodes form contiguous bands; nodes within a band are
	// mutually independent. Dispatch a whole band, then barrier on it before the
	// next. Because no task ever blocks on another task, a fixed-size worker pool
	// cannot deadlock regardless of graph depth.
	for (std::size_t begin = 0; begin < nodes.size();)
	{
		std::size_t end = begin;
		while (end < nodes.size() && plan.m_heights[end] == plan.m_heights[begin])
		{
			++end;
		}

		for (std::size_t i = begin; i < end; ++i)
		{
			Node* node = get_node(nodes[i]);
			node->m_state.store(ExecutionState::PROCESSING, std::memory_order_release);
		}
		for (std::size_t i = begin; i < end; ++i)
		{
			Node*	   node		  = get_node(nodes[i]);
			const bool first_time = was_valid[i] == 0;
			scheduler.submit([this, node, ctx, revision, first_time]()
							 { run_node(*node, *ctx, revision, first_time); });
		}
		// Barrier: wait for every node in the band to leave PROCESSING before moving
		// on, so the next band sees settled inputs (and so a frame is synchronous).
		for (std::size_t i = begin; i < end; ++i)
		{
			Node*		   node	 = get_node(nodes[i]);
			ExecutionState state = node->m_state.load(std::memory_order_acquire);
			while (state == ExecutionState::PROCESSING)
			{
				node->m_state.wait(state, std::memory_order_acquire);
				state = node->m_state.load(std::memory_order_acquire);
			}
		}

		begin = end;
	}
}

std::expected<FramePlan, GraphError> Graph::frame(std::span<const DataHandle> sinks, Scheduler& scheduler)
{
	auto plan = resolve(sinks);
	if (plan.has_value())
	{
		execute(*plan, scheduler);
	}
	return plan;
}

std::expected<FramePlan, GraphError> Graph::frame(DataHandle sink, Scheduler& scheduler)
{
	return frame(std::span<const DataHandle>{&sink, 1}, scheduler);
}

std::string Graph::to_dot() const
{
	std::string out = "digraph veng {\n  rankdir=LR;\n  node [fontname=\"monospace\"];\n";

	for (std::uint32_t index = 0; index < m_data.size(); ++index)
	{
		const Data* slot  = m_data[index].get();
		const char* color = slot->is_source() ? "lightblue" : "white";
		out += std::format("  d{} [shape=ellipse,style=filled,fillcolor={},label=\"d{} @{}\"];\n", index, color, index,
						   slot->changed_at());
	}

	for (std::uint32_t index = 0; index < m_nodes.size(); ++index)
	{
		const Node* node  = m_nodes[index].get();
		const char* color = "white";
		switch (node->state())
		{
			case ExecutionState::VALID: color = "palegreen"; break;
			case ExecutionState::PROCESSING: color = "khaki"; break;
			case ExecutionState::INVALID: color = "lightgray"; break;
			case ExecutionState::FAILED: color = "lightcoral"; break;
		}
		out += std::format("  n{} [shape=box,style=filled,fillcolor={},label=\"n{} {}\"];\n", index, color, index,
						   to_string(node->state()));

		for (const DataHandle input : node->inputs())
		{
			out += std::format("  d{} -> n{};\n", input.index, index);
		}
		for (const DataHandle output : node->outputs())
		{
			out += std::format("  n{} -> d{};\n", index, output.index);
		}
	}

	out += "}\n";
	return out;
}
} // namespace veng::graph
