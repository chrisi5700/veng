//
// See veng/passes/OutlinePass.hpp.
//

#include <memory>
#include <utility>
#include <veng/gpu/ImageRef.hpp>
#include <veng/passes/OutlinePass.hpp>
#include <veng/rendergraph/data/Data.hpp>

namespace veng::passes
{
OutlinePass::OutlinePass(graph::Graph& graph, vk::Format color_format, graph::TypedHandle<vk::Extent2D> screen,
						 graph::DataHandle output, const OutlineConfig& config)
	: m_graph(&graph)
{
	using graph::DataHandle;
	using graph::TypedHandle;
	using graph::ValueData;
	using nodes::GraphicsNode;

	// Reactive config sources. `width` feeds BOTH blur passes so the horizontal and vertical
	// kernels stay matched (a symmetric glow); `color` tints the final ring. Color is stored as
	// a vec4 because a push-constant vec3 still occupies a 16-byte slot — carrying the pad
	// explicitly keeps the C++ push size (16) and the shader's `float4` byte-identical.
	m_width_src = graph.add_source<float>(config.width);
	m_color_src = graph.add_source<glm::vec4>(glm::vec4(config.color, 1.0F));

	// 1 / extent, so the Gaussian taps are expressed in UV space and the glow is
	// resolution-independent. A pure CPU transform: recomputed only when the window resizes.
	const TypedHandle<glm::vec2> texel = graph.add_transform(
		[](const vk::Extent2D& size) -> glm::vec2
		{ return {1.0F / static_cast<float>(size.width), 1.0F / static_cast<float>(size.height)}; }, screen);

	// Internal edges the caller never sees: the white mask and the horizontal-blur result.
	const DataHandle silhouette_image = graph.add(std::make_unique<ValueData<gpu::ImageRef>>(gpu::ImageRef{}));
	const DataHandle blurred_image	  = graph.add(std::make_unique<ValueData<gpu::ImageRef>>(gpu::ImageRef{}));

	// Silhouette: a solid-white mask, no depth. Starts with zero draws; add_mesh appends one
	// draw per outlined mesh. We keep a non-owning pointer to the live node for that extension
	// (the graph owns the unique_ptr; the pointer stays valid while the graph lives).
	auto silhouette = std::make_unique<GraphicsNode>("passes/outline_silhouette.vert", "passes/outline_silhouette.frag",
													 color_format, vk::Format::eUndefined, 0, screen, silhouette_image);
	m_silhouette	= silhouette.get();
	m_silhouette_node = graph.add(std::move(silhouette));
	graph.set_producer(silhouette_image, m_silhouette_node);

	// Horizontal Gaussian of the mask. Fragment push layout: texel @0 (vec2), width @8 (float).
	auto blur = std::make_unique<GraphicsNode>("passes/outline_fullscreen.vert", "passes/outline_blur.frag",
											   color_format, vk::Format::eUndefined, 3, screen, blurred_image);
	blur->add_sampled_image(silhouette_image, "silhouette")
		.push_constant<glm::vec2>(texel, vk::ShaderStageFlagBits::eFragment, 0)
		.push_constant<float>(m_width_src, vk::ShaderStageFlagBits::eFragment, 8);
	m_blur_node = graph.add(std::move(blur));
	graph.set_producer(blurred_image, m_blur_node);

	// Vertical Gaussian + ring extraction + tint. Fragment push layout: color @0 (vec4),
	// texel @16 (vec2), width @24 (float) — color first so its 16-byte slot needs no interior
	// padding (matches the shader's `Push` exactly).
	auto ring = std::make_unique<GraphicsNode>("passes/outline_fullscreen.vert", "passes/outline_ring.frag",
											   color_format, vk::Format::eUndefined, 3, screen, output);
	ring->add_sampled_image(blurred_image, "blurredH")
		.add_sampled_image(silhouette_image, "silhouette")
		.push_constant<glm::vec4>(m_color_src, vk::ShaderStageFlagBits::eFragment, 0)
		.push_constant<glm::vec2>(texel, vk::ShaderStageFlagBits::eFragment, 16)
		.push_constant<float>(m_width_src, vk::ShaderStageFlagBits::eFragment, 24);
	m_ring_node = graph.add(std::move(ring));
	graph.set_producer(output, m_ring_node);
}

void OutlinePass::add_mesh(graph::DataHandle mesh, graph::DataHandle mvp)
{
	// One draw per mesh, each pushing its own MVP (vertex stage, offset 0). add_draw marks the
	// node dirty, so this is valid at runtime as well as during setup.
	m_silhouette->add_draw(mesh).push_constant<glm::mat4>(mvp, vk::ShaderStageFlagBits::eVertex, 0);
}

void OutlinePass::set_color(glm::vec3 color)
{
	m_graph->set(m_color_src, glm::vec4(color, 1.0F));
}

void OutlinePass::set_width(float width)
{
	m_graph->set(m_width_src, width);
}
} // namespace veng::passes
