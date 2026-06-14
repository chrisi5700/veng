//
// veng example — three static primitives (cube, sphere, torus) under a directional light,
// drawn in a single GraphicsNode with one `add_draw` per shape. No instancing, no per-frame
// CPU work: the only reactive surface is the orbit camera's `view_proj`. With OnDemand
// pacing (the default) the loop idles when the camera is still — the cached% line should
// quickly hit ~100% as soon as you stop dragging.
//
// Each draw carries its own (model, view_proj) push-constant block via per-draw push-
// constant edges at the right offsets (the shader's PushData puts model at 0, view_proj at
// 64). The model matrix is a one-shot source — set at startup, never mutated — so the
// shapes are pinned in place; the camera does all the work.
//

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <span>
#include <utility>
#include <veng/gpu/MeshRef.hpp>
#include <veng/gpu/UniformRef.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/nodes/MeshNode.hpp>
#include <veng/nodes/UniformNode.hpp>
#include <veng/rendergraph/data/Data.hpp>

#include "AppLoop.hpp"
#include "Geometry.hpp"

using namespace veng;
using namespace veng::graph;

namespace
{
// Add one mesh + one draw to `node` at a fixed world-space `position`. `mesh` is owned by
// `graph` (the caller will MeshNode-upload it) — this function just adds the draw and wires
// the per-draw model + the shared view_proj push constants.
void add_shape(Graph& graph, nodes::GraphicsNode& node, DataHandle mesh, TypedHandle<glm::mat4> view_proj,
			   glm::vec3 position)
{
	const auto model	 = graph.add_source<glm::mat4>(glm::translate(glm::mat4(1.0F), position));
	auto	   draw_cfg	 = node.add_draw(mesh);
	draw_cfg.push_constant<glm::mat4>(model.handle, veng::rhi::ShaderStage::VERTEX, 0);
	draw_cfg.push_constant<glm::mat4>(view_proj.handle, veng::rhi::ShaderStage::VERTEX, 64);
}
} // namespace

int main()
{
	example::AppLoop app(example::AppConfig{.title			 = "veng — static shapes",
											.width			 = 1280,
											.height			 = 720,
											.camera_distance = 5.0F});

	Graph& graph = app.graph();

	// --- Meshes (uploaded once) -----------------------------------------------------
	const example::Mesh cube_mesh	= example::make_cube({0.95F, 0.55F, 0.30F});
	const example::Mesh sphere_mesh = example::make_sphere(0.6F, 24, 48, {0.40F, 0.85F, 0.55F});
	const example::Mesh torus_mesh	= example::make_torus(0.55F, 0.18F, 48, 24, {0.50F, 0.65F, 1.0F});

	const auto upload_mesh = [&](const example::Mesh& mesh)
	{
		const DataHandle slot = graph.add(std::make_unique<ValueData<gpu::MeshRef>>(gpu::MeshRef{}));
		graph.set_producer(slot, graph.add(std::make_unique<nodes::MeshNode>(
										std::span<const example::Vertex>(mesh.vertices),
										std::span<const std::uint32_t>(mesh.indices), slot)));
		return slot;
	};
	const DataHandle cube_ref	= upload_mesh(cube_mesh);
	const DataHandle sphere_ref = upload_mesh(sphere_mesh);
	const DataHandle torus_ref	= upload_mesh(torus_mesh);

	// --- Directional light ---------------------------------------------------------
	auto			 light_dir_src = graph.add_source<glm::vec4>(glm::vec4(-0.4F, -0.8F, -0.45F, 0.0F));
	const DataHandle light_ref	   = graph.add(std::make_unique<ValueData<gpu::UniformRef>>(gpu::UniformRef{}));
	graph.set_producer(light_ref,
					   graph.add(std::make_unique<nodes::UniformNode>(light_dir_src, "light", light_ref)));

	// --- One GraphicsNode hosting three draws --------------------------------------
	auto scene = std::make_unique<nodes::GraphicsNode>("demo/lit_basic.vert", "demo/lit_directional.frag",
													   app.scene_color_format(), app.depth_format(), 0, app.screen(),
													   app.scene_image());
	scene->add_uniform(light_ref).clear_color({0.04F, 0.05F, 0.08F, 1.0F});

	auto* scene_ptr = scene.get();
	add_shape(graph, *scene_ptr, cube_ref, app.view_proj(), {-1.8F, 0.0F, 0.0F});
	add_shape(graph, *scene_ptr, sphere_ref, app.view_proj(), {0.0F, 0.0F, 0.0F});
	add_shape(graph, *scene_ptr, torus_ref, app.view_proj(), {1.8F, 0.0F, 0.0F});

	graph.set_producer(app.scene_image(), graph.add(std::move(scene)));

	app.run();
	return 0;
}
