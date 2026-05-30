//
// picking_outline — composing three passes. A handful of shaded shapes (some translucent);
// left-click a shape to add it to the outline (a glow highlight), left-click again to remove it.
// Shows how to wire veng::passes::PhongPass (lit + transparent scene), PickingPass (GPU id
// readback) and OutlinePass (screen-space glow) into one frame alongside a composite.
//
// Frame graph built here:
//   PhongPass   (Blinn-Phong + transparency, one object per add_object) -> scene image
//   OutlinePass (silhouettes of the *selected* objects)                 -> glow image
//   composite   (fullscreen: scene + glow, additive)                    -> app.scene_image()
//   PickingPass (ids of *all* objects)                                  -> done_token (a sink)
//
// Selection is reactive and needs no graph surgery: each object has a `selected` source (0/1),
// and its outline draw reads an `outline_mvp` transform that returns the real MVP when selected,
// or a degenerate matrix that collapses the mesh to a single point (no fragments) when not. So a
// deselected object contributes nothing to the silhouette — and because that transform's value
// is then a *constant*, it stays cached as the camera orbits, so the (wide) blur only runs while
// something is actually highlighted. A click runs PickingPass; its async callback toggles the
// picked object's `selected`, and the next frame the outline follows.
//

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <span>
#include <utility>
#include <vector>
#include <veng/gpu/MeshRef.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/nodes/MeshNode.hpp>
#include <veng/passes/OutlinePass.hpp>
#include <veng/passes/PhongPass.hpp>
#include <veng/passes/PickingPass.hpp>
#include <veng/rendergraph/data/Data.hpp>

#include "AppLoop.hpp"
#include "Geometry.hpp"

namespace
{
using veng::graph::DataHandle;
using veng::graph::Graph;
using veng::graph::TypedHandle;
using veng::graph::ValueData;

// MeshNode uploads the geometry once from these spans and produces a gpu::MeshRef on `slot`;
// only the ref is a graph edge (the node owns the vertex/index bytes).
DataHandle upload_mesh(Graph& graph, const example::Mesh& mesh)
{
	const DataHandle slot = graph.add(std::make_unique<ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));
	graph.set_producer(
		slot, graph.add(std::make_unique<veng::nodes::MeshNode>(std::span<const example::Vertex>(mesh.vertices),
																std::span<const std::uint32_t>(mesh.indices), slot)));
	return slot;
}

glm::mat4 model_at(glm::vec3 position, float scale)
{
	return glm::scale(glm::translate(glm::mat4(1.0F), position), glm::vec3(scale));
}

// A transform that maps every vertex to clip (0,0,0,1) — one point, so all triangles are
// zero-area and nothing rasterizes. Used as the "not selected" outline matrix; being constant,
// it also keeps the outline branch cached while deselected.
glm::mat4 hidden_transform()
{
	glm::mat4 collapse(0.0F);
	collapse[3][3] = 1.0F;
	return collapse;
}

// Cursor (screen px, top-left origin) -> framebuffer px, clamped in-bounds. Assumes a 1:1 display
// scale (the usual X11 / no-HiDPI case); on a fractional-scaled display you'd multiply by
// framebuffer-size / window-size.
std::pair<std::uint32_t, std::uint32_t> cursor_to_pixel(vk::Extent2D framebuffer, double cursor_x, double cursor_y)
{
	const auto px = static_cast<std::uint32_t>(std::clamp(cursor_x, 0.0, static_cast<double>(framebuffer.width) - 1.0));
	const auto py =
		static_cast<std::uint32_t>(std::clamp(cursor_y, 0.0, static_cast<double>(framebuffer.height) - 1.0));
	return {px, py};
}
} // namespace

int main()
{
	example::AppLoop app(example::AppConfig{.title = "veng — picking + outline", .camera_distance = 9.0F});
	auto&			 graph = app.graph();

	// White meshes: PhongPass multiplies the mesh's vertex color by the per-object push color, so
	// white geometry makes the color passed to add_object() authoritative (the user "sets the color
	// and mesh themselves").
	const glm::vec3	 white	= {1.0F, 1.0F, 1.0F};
	const DataHandle cube	= upload_mesh(graph, example::make_cube(white));
	const DataHandle sphere = upload_mesh(graph, example::make_sphere(0.5F, 24, 48, white));
	const DataHandle torus	= upload_mesh(graph, example::make_torus(0.5F, 0.18F, 48, 24, white));

	// Intermediate images: the lit scene (sampled by the composite) and the outline glow.
	const DataHandle scene = graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));
	const DataHandle glow  = graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));

	// The lit scene is a PhongPass: per-pixel Blinn-Phong with transparency. Each object's color
	// (set below via add_object) drives both the tint and, through its alpha, whether it is drawn
	// translucent. Specular highlights track the camera via app.camera().eye_pos().
	veng::passes::PhongPass phong(graph, app.scene_color_format(), app.depth_format(), app.screen(), scene,
								  app.view_proj(), app.camera().eye_pos(), {.shininess = 56.0F});

	veng::passes::OutlinePass outline(graph, app.scene_color_format(), app.screen(), glow,
									  {.color = {1.0F, 0.6F, 0.1F}, .width = 1.0F});
	veng::passes::PickingPass picking(graph, app.screen(), app.depth_format());
	app.add_sink(picking.done_token()); // so the driver pulls the readback in when a pick is pending
	// A pick resolves only on frame retirement (a few rendered frames later); keep the loop pumping
	// full frames until it does, so a highlight appears immediately instead of waiting for the next
	// camera move to wake the renderer.
	app.add_keep_alive([&picking] { return picking.pending(); });

	const glm::mat4 hidden = hidden_transform();

	struct Spec
	{
		DataHandle mesh;
		glm::vec3  position;
		glm::vec4  color;	  // rgb tint; a < 1 => translucent (drawn back-then-front, alpha-blended)
		float	   shininess; // per-object specular exponent — tighter highlight = glossier
	};
	// Two rows of the SAME three shapes so opaque and translucent can be compared side by side:
	// top row opaque (a = 1), bottom row the same shapes/colors at a = 0.45 (see-through). If
	// transparency is wired right, each bottom shape shows its near faces over its far faces; the
	// matching opaque shape directly above is the reference. Per-shape shininess varies the gloss:
	// soft spheres, sharp gold torus.
	const std::array<Spec, 6> specs{{
		{cube, {-2.2F, 1.3F, 0.0F}, {0.85F, 0.40F, 0.40F, 1.00F}, 48.0F},	// opaque red cube
		{sphere, {0.0F, 1.3F, 0.0F}, {0.40F, 0.80F, 0.50F, 1.00F}, 24.0F},	// opaque green sphere (soft)
		{torus, {2.2F, 1.3F, 0.0F}, {0.95F, 0.80F, 0.30F, 1.00F}, 96.0F},	// opaque gold torus (sharp)
		{cube, {-2.2F, -1.3F, 0.0F}, {0.85F, 0.40F, 0.40F, 0.45F}, 48.0F},	// glass red cube
		{sphere, {0.0F, -1.3F, 0.0F}, {0.40F, 0.80F, 0.50F, 0.45F}, 24.0F}, // glass green sphere (soft)
		{torus, {2.2F, -1.3F, 0.0F}, {0.95F, 0.80F, 0.30F, 0.45F}, 96.0F},	// glass gold torus (sharp)
	}};

	std::vector<TypedHandle<std::uint32_t>> selected_src;				// one 0/1 source per object
	std::vector<std::uint32_t>				selected(specs.size(), 0U); // CPU mirror, for toggling
	selected_src.reserve(specs.size());

	// Register one scene object across all three passes in a single call, so the phong / picking /
	// outline wiring can never drift: they share one `model` edge (PhongPass) and one `mvp`
	// transform (picking id + outline silhouette). It assigns the next picking id (1-based; id 0 is
	// "no hit") and adds a per-object `selected` toggle that swaps the outline transform between the
	// real mvp and the degenerate (invisible) one. This is the composition the whole example exists
	// to show, expressed once.
	const auto add_object = [&](DataHandle mesh, glm::vec3 position, glm::vec4 color, float shininess)
	{
		const glm::mat4					 model	   = model_at(position, 1.0F);
		const TypedHandle<glm::mat4>	 model_src = graph.add_source<glm::mat4>(model);
		const TypedHandle<std::uint32_t> sel	   = graph.add_source<std::uint32_t>(0U);
		const auto						 id		   = static_cast<std::uint32_t>(selected_src.size() + 1);
		selected_src.push_back(sel);

		// mvp = view_proj * model — shared by the picking pass and the outline's "selected" value.
		const TypedHandle<glm::mat4> mvp =
			graph.add_transform([model](const glm::mat4& view_proj) { return view_proj * model; }, app.view_proj());
		// Outline transform: the real mvp when selected, the degenerate (invisible) one when not.
		const TypedHandle<glm::mat4> outline_mvp =
			graph.add_transform([hidden](const glm::mat4& selected_mvp, const std::uint32_t& on)
								{ return on != 0U ? selected_mvp : hidden; }, mvp, sel);

		phong.add_object(mesh, model_src, color, shininess); // lit; color.a < 1 => translucent
		picking.add_object(mesh, mvp, id);					 // same mvp -> picks track the shaded geometry
		outline.add_mesh(mesh, outline_mvp);				 // glows when this object's `selected` is on
	};

	for (const Spec& spec : specs)
	{
		add_object(spec.mesh, spec.position, spec.color, spec.shininess);
	}

	// Composite: displayed image = scene + outline glow. The glow is premultiplied, so
	// demo/composite.frag (scene, ring, samp) simply adds it on top; with nothing selected the
	// glow is black and the scene passes through unchanged.
	auto composite = std::make_unique<veng::nodes::GraphicsNode>("demo/fullscreen.vert", "demo/composite.frag",
																 app.scene_color_format(), vk::Format::eUndefined, 3,
																 app.screen(), app.scene_image());
	composite->add_sampled_image(scene, "scene").add_sampled_image(glow, "ring");
	graph.set_producer(app.scene_image(), graph.add(std::move(composite)));

	// Input: a left *click* (press + release without dragging, so it doesn't fight orbit) picks
	// the object under the cursor; the async result toggles that object's selection.
	bool   button_was_down = false;
	double press_x		   = 0.0;
	double press_y		   = 0.0;

	app.run({},
			[&]
			{
				const bool down	  = app.window().mouse_down(GLFW_MOUSE_BUTTON_LEFT);
				const auto cursor = app.window().cursor_pos();
				if (down && !button_was_down)
				{
					press_x = cursor.x;
					press_y = cursor.y;
				}
				else if (!down && button_was_down)
				{
					const double drift = std::abs(cursor.x - press_x) + std::abs(cursor.y - press_y);
					if (drift < 5.0) // a click, not a drag-to-orbit
					{
						const auto [px, py] = cursor_to_pixel(app.window().framebuffer_extent(), cursor.x, cursor.y);
						picking.pick(px, py,
									 [&app, &selected, &selected_src](veng::passes::PickResult result)
									 {
										 if (!result.hit)
										 {
											 return;
										 }
										 const std::size_t idx = static_cast<std::size_t>(result.id) - 1;
										 if (idx >= selected.size())
										 {
											 return;
										 }
										 app.publish(
											 [idx, &selected, &selected_src](Graph& mutable_graph)
											 {
												 selected[idx] = selected[idx] != 0U ? 0U : 1U;
												 mutable_graph.set(selected_src[idx], selected[idx]);
											 });
									 });
					}
				}
				button_was_down = down;
			});
	return 0;
}
