//
// Reactive-core demo: the IFS prototype from the original playground, reborn on the
// lifted L3 graph (design.md §11.3). Shows demand-driven evaluation, cross-frame
// caching, and equality-gated invalidation — no Vulkan involved.
//

#include <cstddef>
#include <print>
#include <string>
#include <vector>
#include <veng/rendergraph/Graph.hpp>

using namespace veng::graph;

namespace
{
const std::vector<int>& particles_of(const Graph& graph, DataHandle handle)
{
	return dynamic_cast<const ValueData<std::vector<int>>*>(graph.get_data(handle))->value();
}
} // namespace

int main()
{
	Graph			graph;
	InlineScheduler scheduler;

	// Sources: externally-mutated reactive roots.
	auto count = graph.add_source<std::size_t>(8);
	auto name  = graph.add_source<std::string>("Triangle");

	// IFS node: (count, name) -> particle buffer.
	auto particles = graph.add_transform(
		[](const std::size_t& n, const std::string& ifs_name)
		{
			std::println("  [IFS]    generating {} particles for '{}'", n, ifs_name);
			return std::vector<int>(n, static_cast<int>(ifs_name.size()));
		},
		count, name);

	// Mapper: offsets every particle by the IFS name length.
	auto mapped = graph.add_transform(
		[](const std::vector<int>& data, const std::string& ifs_name)
		{
			std::println("  [mapper] offsetting {} particles by len('{}')", data.size(), ifs_name);
			std::vector<int> out = data;
			for (int& value : out)
			{
				value += static_cast<int>(ifs_name.size());
			}
			return out;
		},
		particles, name);

	const auto present = [&](const char* label)
	{
		const auto plan = graph.frame(mapped, scheduler);
		std::println("{}: frame {} executed {} node(s); first particle = {}", label, graph.current_revision(),
					 plan->size(), particles_of(graph, mapped).front());
	};

	present("initial   "); // both nodes run
	present("re-present"); // nothing changed -> 0 nodes (cached)

	dynamic_cast<ValueData<std::string>*>(graph.get_data(name))->set("Sierpinski");
	present("rename    "); // name feeds both nodes -> both re-run

	dynamic_cast<ValueData<std::size_t>*>(graph.get_data(count))->set(8);
	present("same count"); // equality-gated -> 0 nodes

	std::println("\n--- graph (GraphViz) ---\n{}", graph.to_dot());
	return 0;
}
