//
// See OutlinePass.hpp.
//

#include "OutlinePass.hpp"

#include <memory>
#include <utility>
#include <veng/gpu/ImageRef.hpp>
#include <veng/rendergraph/data/Data.hpp>

namespace demo
{
void OutlinePass::add_mesh(veng::graph::DataHandle mesh, veng::graph::DataHandle mvp)
{
	// The silhouette pass is just multi-draw: one draw per mesh, each pushing its own MVP (the
	// same per-object handle the scene pass uses for the same mesh — handles are reactive, so
	// when the object moves both passes see the change). Extending m_inputs/m_draws after the
	// node is in the graph is safe at construction time (no resolve yet); a runtime add would
	// also need to mark the node dirty so the planner re-runs it.
	m_silhouette->add_draw(mesh).push_constant<glm::mat4>(mvp, vk::ShaderStageFlagBits::eVertex);
}

OutlinePass create_outline_pass(veng::graph::Graph& graph, veng::ResourcePool& /*pool*/, vk::Format format,
								veng::graph::TypedHandle<vk::Extent2D> screen, veng::graph::DataHandle output)
{
	using namespace veng;
	using namespace veng::graph;
	using namespace veng::nodes;

	// 1/extent for the blur taps in UV space, so the glow width is resolution-independent. A
	// pure CPU transform: changes only when the screen-size source changes.
	const TypedHandle<glm::vec2> texel = graph.add_transform(
		[](const vk::Extent2D& size) -> glm::vec2 {
			return {1.0F / static_cast<float>(size.width), 1.0F / static_cast<float>(size.height)};
		},
		screen);

	// Intermediate handles for the silhouette mask + the horizontal-blur output. Internal to
	// this sub-graph; the caller never sees them.
	const DataHandle silhouette_image = graph.add(std::make_unique<ValueData<gpu::ImageRef>>(gpu::ImageRef{}));
	const DataHandle blurred_h_image  = graph.add(std::make_unique<ValueData<gpu::ImageRef>>(gpu::ImageRef{}));

	// Silhouette: a per-mesh white mask. Caller adds the meshes via `pass.add_mesh` after this
	// function returns; we keep a raw pointer to the live node for the extension. (The graph
	// owns the unique_ptr; the pointer is non-owning and stays valid while the graph lives.)
	auto			  silhouette	 = std::make_unique<GraphicsNode>("demo/mesh.vert", "demo/silhouette.frag", format,
																	  vk::Format::eUndefined, 0, screen, silhouette_image);
	GraphicsNode*	  silhouette_ptr = silhouette.get();
	const NodeHandle  silhouette_node = graph.add(std::move(silhouette));
	graph.set_producer(silhouette_image, silhouette_node);

	// Horizontal Gaussian of the silhouette mask (fullscreen sampling).
	auto blur_h = std::make_unique<GraphicsNode>("demo/fullscreen.vert", "demo/blur_h.frag", format,
												 vk::Format::eUndefined, 3, screen, blurred_h_image);
	blur_h->add_sampled_image(silhouette_image, "silhouette")
		.push_constant<glm::vec2>(texel, vk::ShaderStageFlagBits::eFragment);
	const NodeHandle blur_h_node = graph.add(std::move(blur_h));
	graph.set_producer(blurred_h_image, blur_h_node);

	// Vertical Gaussian + ring extraction (blurred − sharp silhouette, tinted): the glow itself.
	auto ring = std::make_unique<GraphicsNode>("demo/fullscreen.vert", "demo/ring.frag", format,
											   vk::Format::eUndefined, 3, screen, output);
	ring->add_sampled_image(blurred_h_image, "blurredH")
		.add_sampled_image(silhouette_image, "silhouette")
		.push_constant<glm::vec2>(texel, vk::ShaderStageFlagBits::eFragment);
	const NodeHandle ring_node = graph.add(std::move(ring));
	graph.set_producer(output, ring_node);

	return OutlinePass{silhouette_ptr, silhouette_node, blur_h_node, ring_node};
}
} // namespace demo
