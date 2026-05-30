//
// veng example — a small scene of `Object`s (mesh + transform + shared material reference)
// built up live and then animated. Demonstrates a real-world-shaped scene graph on top of
// the reactive renderer:
//
//   * A small Material library (4 entries: metallic, plastic, rubber, jewel) lives in a
//     shared SSBO. Multiple Objects reference the same material by index — that's the whole
//     point of separating Material from Transform.
//   * Three meshes (sphere, cube, torus) uploaded once via MeshNode each.
//   * Three Transform SSBOs (one per mesh kind). Each frame the sim thread partitions the
//     `Object` list by mesh kind and republishes the three arrays via `app.publish` — the
//     StorageBufferNodes pick up the change at the frame boundary, the instanced draws
//     pick up the new count via `set_instances_from`.
//   * One GraphicsNode hosts three draws (one per mesh kind), each binding its mesh and
//     pushing its `mesh_kind` push constant; the shader picks which transforms_<kind>[]
//     SSBO to read.
//   * Light + view are scene-wide UniformNodes (light position + color, eye position).
//   * view_proj rides as a push constant from the orbit camera.
//
// Two-phase sim thread:
//   * Buildup (0 .. N*1s): every second, one new Object is appended. Order is metallic
//     sphere, metallic cube, metallic torus, plastic sphere, … — i.e. by material, then
//     by mesh. The grid fills row by row.
//   * Animate: after all objects are placed, each one slowly rotates around its own axis
//     and bobs a little — every frame the per-object transforms are recomputed, the SSBOs
//     re-uploaded, the draws re-run. The materials buffer never changes after startup, so
//     it stays cached.
//

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <span>
#include <thread>
#include <utility>
#include <vector>
#include <veng/gpu/BufferRef.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/gpu/UniformRef.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/nodes/MeshNode.hpp>
#include <veng/nodes/StorageBufferNode.hpp>
#include <veng/nodes/UniformNode.hpp>
#include <veng/rendergraph/data/Data.hpp>

#include "AppLoop.hpp"
#include "Geometry.hpp"

using namespace veng;
using namespace veng::graph;

namespace
{
// --- Shader-mirror structs -------------------------------------------------------

// Mirrors `Transform` in shaders/demo/phong.vert.slang. std430 stride is 144: two mat4s
// (128) + uint material_index (4) padded out to the mat4's 16-byte alignment.
struct Transform
{
	glm::mat4	  model;
	glm::mat4	  normal_mat;
	std::uint32_t material_index;
	std::uint32_t _pad[3];

	friend bool operator==(const Transform&, const Transform&) noexcept = default;
};
static_assert(sizeof(Transform) == 144, "Transform stride must match the shader's std430 layout");

// Mirrors `Material` in shaders/demo/phong.frag.slang. Appearance only — no transform.
struct Material
{
	glm::vec4 ambient;	// .xyz = color, .w = Ka
	glm::vec4 diffuse;	// .xyz = color, .w = Kd
	glm::vec4 specular; // .xyz = color, .w = Ks
	glm::vec4 params;	// .x = shininess, .y = mode

	friend bool operator==(const Material&, const Material&) noexcept = default;
};

struct LightUniform
{
	glm::vec4 position;
	glm::vec4 color;

	friend bool operator==(const LightUniform&, const LightUniform&) noexcept = default;
};

// --- Scene model ---------------------------------------------------------------

// Each mesh kind has its own VkBuffer (uploaded once) AND its own Transform SSBO; the
// `mesh` enum is what the partition step keys off.
enum class MeshKind : std::uint8_t
{
	Sphere = 0,
	Cube   = 1,
	Torus  = 2,
	Count  = 3,
};

// Real-world-style scene object: owns *its* placement + a *reference* to a shared material.
// The orientation lives as (axis, angle) so the sim thread can drive `angle` from a clock
// without touching the rest of the object's state.
struct Object
{
	MeshKind	  mesh;
	std::uint32_t material_index;
	glm::vec3	  position;
	glm::vec3	  spin_axis;
	float		  spin_speed; // rad / s
	float		  bob_phase;  // radians
};

// --- Material library (referenced by index from Object::material_index) -------

const std::array<Material, 4> MATERIALS{
	// 0: metallic (gold-ish — high specular, high shininess)
	Material{.ambient  = glm::vec4(0.24725F, 0.1995F, 0.0745F, 1.0F),
			 .diffuse  = glm::vec4(0.75164F, 0.60648F, 0.22648F, 0.9F),
			 .specular = glm::vec4(0.628281F, 0.555802F, 0.366065F, 0.95F),
			 .params   = glm::vec4(51.2F, 1.0F, 0.0F, 0.0F)},
	// 1: plastic (red, medium specular, medium shininess — the "shiny red plastic" look)
	Material{.ambient  = glm::vec4(0.05F, 0.0F, 0.0F, 1.0F),
			 .diffuse  = glm::vec4(0.85F, 0.18F, 0.18F, 1.0F),
			 .specular = glm::vec4(0.70F, 0.60F, 0.60F, 0.50F),
			 .params   = glm::vec4(32.0F, 1.0F, 0.0F, 0.0F)},
	// 2: rubber (green, very low specular, low shininess — soft matte look)
	Material{.ambient  = glm::vec4(0.0F, 0.05F, 0.0F, 1.0F),
			 .diffuse  = glm::vec4(0.40F, 0.65F, 0.40F, 1.0F),
			 .specular = glm::vec4(0.04F, 0.10F, 0.04F, 0.10F),
			 .params   = glm::vec4(10.0F, 1.0F, 0.0F, 0.0F)},
	// 3: jewel (cyan, high specular, extreme shininess — sharp pinpoint highlights)
	Material{.ambient  = glm::vec4(0.0F, 0.10F, 0.10F, 1.0F),
			 .diffuse  = glm::vec4(0.05F, 0.55F, 0.55F, 1.0F),
			 .specular = glm::vec4(1.0F, 1.0F, 1.0F, 1.0F),
			 .params   = glm::vec4(96.0F, 1.0F, 0.0F, 0.0F)},
};
constexpr std::size_t MESH_KINDS = static_cast<std::size_t>(MeshKind::Count);
constexpr std::size_t TOTAL_OBJECTS = MATERIALS.size() * MESH_KINDS;

// --- Spawn ordering & placement ------------------------------------------------

// The (index)'th object is metallic-sphere, metallic-cube, metallic-torus, plastic-sphere,
// plastic-cube, … — material outer, mesh inner. Placement is a (materials × meshes) grid
// in the xz plane: mesh index along x, material index along z.
Object make_object(std::size_t index)
{
	constexpr float SPACING_X	  = 2.6F;
	constexpr float SPACING_Z	  = 2.6F;
	const std::size_t material	  = index / MESH_KINDS;
	const std::size_t mesh		  = index % MESH_KINDS;
	// Hash for per-object decorrelation (rotation axis + speed + bob phase).
	const std::uint32_t h		  = static_cast<std::uint32_t>(index) * 2654435761U;
	const float			h0		  = static_cast<float>(h & 0x3FFU) / 1024.0F;
	const float			h1		  = static_cast<float>((h >> 10) & 0x3FFU) / 1024.0F;
	const float			h2		  = static_cast<float>((h >> 20) & 0x3FFU) / 1024.0F;
	return Object{
		.mesh			= static_cast<MeshKind>(mesh),
		.material_index = static_cast<std::uint32_t>(material),
		.position		= glm::vec3{(static_cast<float>(mesh) - 1.0F) * SPACING_X, 0.0F,
									(static_cast<float>(material) - 1.5F) * SPACING_Z},
		.spin_axis		= glm::normalize(glm::vec3{h0 - 0.5F, h1 - 0.5F + 0.6F, h2 - 0.5F}),
		.spin_speed		= 0.4F + 1.4F * h0,
		.bob_phase		= 6.283F * h1,
	};
}

// Compose a Transform from an Object at time `t` (or static if `animate` is false). The
// rotation matrix is built from the spin axis × angle so the look is hand-tuned per object.
Transform build_transform(const Object& obj, float t, bool animate)
{
	glm::mat4 model = glm::translate(glm::mat4(1.0F), obj.position);
	if (animate)
	{
		const float bob = 0.18F * std::sin(t * 1.6F + obj.bob_phase);
		model			= glm::translate(model, glm::vec3{0.0F, bob, 0.0F});
		model			= glm::rotate(model, t * obj.spin_speed, obj.spin_axis);
	}
	return Transform{
		.model			= model,
		// Strict normal matrix: handles a future non-uniform scale or shear without breaking.
		.normal_mat		= glm::transpose(glm::inverse(model)),
		.material_index = obj.material_index,
		._pad			= {},
	};
}

// --- Mesh upload helper --------------------------------------------------------

DataHandle upload_mesh(Graph& graph, const example::Mesh& mesh)
{
	const DataHandle slot = graph.add(std::make_unique<ValueData<gpu::MeshRef>>(gpu::MeshRef{}));
	graph.set_producer(slot, graph.add(std::make_unique<nodes::MeshNode>(
									std::span<const example::Vertex>(mesh.vertices),
									std::span<const std::uint32_t>(mesh.indices), slot)));
	return slot;
}
} // namespace

int main()
{
	example::AppLoop app(example::AppConfig{.title			 = "veng — phong materials & objects",
											.width			 = 1280,
											.height			 = 720,
											.camera_distance = 11.0F,
											.camera_pitch	 = 0.55F});

	Graph& graph = app.graph();

	// --- Meshes (uploaded once) ---------------------------------------------------
	const example::Mesh sphere_mesh = example::make_sphere(0.7F, 28, 56, glm::vec3{1.0F});
	const example::Mesh cube_mesh	= example::make_cube({1.0F, 1.0F, 1.0F});
	const example::Mesh torus_mesh	= example::make_torus(0.6F, 0.22F, 48, 24, glm::vec3{1.0F});
	const DataHandle	mesh_refs[MESH_KINDS] {
		upload_mesh(graph, sphere_mesh),
		upload_mesh(graph, cube_mesh),
		upload_mesh(graph, torus_mesh),
	};

	// --- Material library (static — uploaded once, never republished) -------------
	std::vector<Material> initial_materials(MATERIALS.begin(), MATERIALS.end());
	auto			 materials_src = graph.add_source<std::vector<Material>>(std::move(initial_materials));
	const DataHandle materials_ref = graph.add(std::make_unique<ValueData<gpu::BufferRef>>(gpu::BufferRef{}));
	graph.set_producer(materials_ref,
					   graph.add(std::make_unique<nodes::StorageBufferNode>(materials_src, "materials",
																			materials_ref)));

	// --- Per-mesh Transform SSBOs (one per mesh kind) -----------------------------
	const char*					   transform_names[MESH_KINDS]{"transforms_sphere", "transforms_cube",
																"transforms_torus"};
	TypedHandle<std::vector<Transform>> transform_srcs[MESH_KINDS];
	DataHandle						   transform_refs[MESH_KINDS];
	for (std::size_t i = 0; i < MESH_KINDS; ++i)
	{
		transform_srcs[i] = graph.add_source<std::vector<Transform>>({});
		transform_refs[i] = graph.add(std::make_unique<ValueData<gpu::BufferRef>>(gpu::BufferRef{}));
		graph.set_producer(transform_refs[i], graph.add(std::make_unique<nodes::StorageBufferNode>(
													transform_srcs[i], transform_names[i], transform_refs[i])));
	}

	// --- Scene-wide uniforms ------------------------------------------------------
	const LightUniform light{
		.position = glm::vec4(3.5F, 5.5F, 4.5F, 1.0F),
		.color	  = glm::vec4(1.0F, 0.97F, 0.92F, 1.0F),
	};
	auto			 light_src = graph.add_source<LightUniform>(light);
	const DataHandle light_ref = graph.add(std::make_unique<ValueData<gpu::UniformRef>>(gpu::UniformRef{}));
	graph.set_producer(light_ref,
					   graph.add(std::make_unique<nodes::UniformNode>(light_src, "light", light_ref)));

	const DataHandle view_ref = graph.add(std::make_unique<ValueData<gpu::UniformRef>>(gpu::UniformRef{}));
	graph.set_producer(view_ref,
					   graph.add(std::make_unique<nodes::UniformNode>(app.camera().eye_pos(), "view", view_ref)));

	// --- Per-draw `mesh_kind` constants (one source per draw, never mutated) ------
	TypedHandle<std::uint32_t> mesh_kind_srcs[MESH_KINDS]{
		graph.add_source<std::uint32_t>(0U),
		graph.add_source<std::uint32_t>(1U),
		graph.add_source<std::uint32_t>(2U),
	};

	// --- One pass hosting three instanced draws -----------------------------------
	auto scene = std::make_unique<nodes::GraphicsNode>("demo/phong.vert", "demo/phong.frag",
													   app.scene_color_format(), app.depth_format(), 0, app.screen(),
													   app.scene_image());
	scene->add_storage_buffer(transform_refs[0])
		.add_storage_buffer(transform_refs[1])
		.add_storage_buffer(transform_refs[2])
		.add_storage_buffer(materials_ref)
		.add_uniform(light_ref)
		.add_uniform(view_ref)
		.clear_color({0.04F, 0.05F, 0.08F, 1.0F});

	for (std::size_t i = 0; i < MESH_KINDS; ++i)
	{
		auto draw = scene->add_draw(mesh_refs[i]);
		draw.push_constant<glm::mat4>(app.view_proj().handle, vk::ShaderStageFlagBits::eVertex, 0);
		draw.push_constant<std::uint32_t>(mesh_kind_srcs[i].handle, vk::ShaderStageFlagBits::eVertex, 64);
		draw.set_instances_from(transform_refs[i]);
	}
	graph.set_producer(app.scene_image(), graph.add(std::move(scene)));

	// --- Sim thread: buildup then animation --------------------------------------
	//
	// Two phases sharing one loop. During the buildup phase we add one new object per
	// second; during the animation phase we re-publish the partitioned transforms every
	// frame so the spin & bob update. The publish step is the same in both phases — the
	// reactive edges (the three transform SSBOs) automatically pick up either the
	// length-change (a new object) or the value-change (an animated frame).
	std::atomic<bool>	running{true};
	std::thread			sim(
		[&]
		{
			using namespace std::chrono;
			std::vector<Object> objects;
			objects.reserve(TOTAL_OBJECTS);
			const auto period	   = microseconds(16'667); // ~60 Hz
			const auto build_step  = seconds(1);
			const auto start	   = steady_clock::now();
			auto	   next		   = start;
			auto	   last_add	   = start - build_step; // add the first object immediately
			while (running.load(std::memory_order_relaxed))
			{
				const auto now		 = steady_clock::now();
				const float t		 = duration<float>(now - start).count();
				const bool	building = objects.size() < TOTAL_OBJECTS;

				if (building && (now - last_add) >= build_step)
				{
					objects.push_back(make_object(objects.size()));
					last_add = now;
				}

				// Animate only once all objects are placed — keeps the buildup phase visually
				// crisp (you see each new object snap into place against a still scene).
				const bool animate = !building;

				std::array<std::vector<Transform>, MESH_KINDS> partitioned;
				for (const Object& obj : objects)
				{
					partitioned[static_cast<std::size_t>(obj.mesh)].push_back(build_transform(obj, t, animate));
				}

				app.publish(
					[&](Graph& g)
					{
						for (std::size_t i = 0; i < MESH_KINDS; ++i)
						{
							g.set(transform_srcs[i], std::move(partitioned[i]));
						}
					});

				// During buildup nothing else is changing — sleep through the second; during
				// animation we want a 60Hz cadence.
				next = (building ? now + milliseconds(50) : next + period);
				std::this_thread::sleep_until(next);
			}
		});

	app.run();

	running.store(false);
	sim.join();
	return 0;
}
