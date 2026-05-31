//
// veng example — many instanced cubes orbiting the origin, lit by a directional light,
// viewed through the orbit camera. Demonstrates the rigid-body slice:
// one StorageBufferNode per-frame upload feeds an instanced GraphicsNode whose
// `instanceCount` is reactive on the SSBO's published count. With `example::AppLoop`
// owning all of the window / context / swapchain / camera / frame-closer / loop boilerplate,
// this file is just the graph wiring + the CPU "physics" thread.
//

#include <atomic>
#include <chrono>
#include <cstdint>
#include <glm/glm.hpp>
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
// One per cube — matches StructuredBuffer<Instance> in shaders/demo/lit_instanced.vert.slang.
struct Instance
{
	glm::mat4 model;

	// ValueData<std::vector<Instance>> frame-boundary equality gate forwards to vector ==
	// which forwards to Instance ==; defaulted op is enough for trivial struct equality.
	friend bool operator==(const Instance&, const Instance&) noexcept = default;
};

constexpr std::size_t BODY_COUNT = 125; // 5^3 grid

struct BodyState
{
	glm::vec3 grid_position;
	glm::vec3 spin_axis;
	float	  spin_speed;
	float	  scale;
};

std::vector<BodyState> initial_body_states()
{
	std::vector<BodyState> bodies;
	bodies.reserve(BODY_COUNT);
	constexpr int	N		= 5;
	constexpr float SPACING = 1.8F;
	for (int z = 0; z < N; ++z)
	{
		for (int y = 0; y < N; ++y)
		{
			for (int x = 0; x < N; ++x)
			{
				const glm::vec3 pos{(static_cast<float>(x) - (N - 1) * 0.5F) * SPACING,
									(static_cast<float>(y) - (N - 1) * 0.5F) * SPACING,
									(static_cast<float>(z) - (N - 1) * 0.5F) * SPACING};
				const std::uint32_t idx = static_cast<std::uint32_t>((z * N + y) * N + x);
				const float			h	= static_cast<float>((idx * 2654435761U) % 1024U) / 1024.0F;
				const float			h2	= static_cast<float>(((idx + 17) * 2654435761U) % 1024U) / 1024.0F;
				bodies.push_back(BodyState{
					.grid_position = pos,
					.spin_axis	   = glm::normalize(glm::vec3{h - 0.5F, h2 - 0.5F, (h + h2) * 0.5F - 0.5F + 0.001F}),
					.spin_speed	   = 0.4F + 1.6F * h,
					.scale		   = 0.55F + 0.2F * h2,
				});
			}
		}
	}
	return bodies;
}

std::vector<Instance> build_instances(const std::vector<BodyState>& bodies, float t)
{
	std::vector<Instance> out;
	out.reserve(bodies.size());
	for (const BodyState& body : bodies)
	{
		glm::mat4 m = glm::translate(glm::mat4(1.0F), body.grid_position);
		m			= glm::rotate(m, t * body.spin_speed, body.spin_axis);
		m			= glm::scale(m, glm::vec3(body.scale));
		out.push_back(Instance{.model = m});
	}
	return out;
}
} // namespace

int main()
{
	example::AppLoop app(example::AppConfig{
		.title = "veng — instanced rigid-body viewer", .width = 1280, .height = 720, .camera_distance = 14.0F});

	// --- Scene ----------------------------------------------------------------------
	//   cube_mesh   : MeshNode (static upload)
	//   instances   : StorageBufferNode (per-frame upload of model matrices)
	//   light_dir   : UniformNode (directional light)
	//   cube draw   : GraphicsNode (lit_instanced.vert + lit_directional.frag)
	//                 → produces app.scene_image(); count comes from instances ref
	Graph& graph = app.graph();

	auto bodies_src	   = graph.add_source<std::vector<Instance>>(std::vector<Instance>(BODY_COUNT));
	auto light_dir_src = graph.add_source<glm::vec4>(glm::vec4(-0.4F, -0.8F, -0.45F, 0.0F));

	const DataHandle cube_mesh	   = graph.add(std::make_unique<ValueData<gpu::MeshRef>>(gpu::MeshRef{}));
	const DataHandle instances_ref = graph.add(std::make_unique<ValueData<gpu::BufferRef>>(gpu::BufferRef{}));
	const DataHandle light_ref	   = graph.add(std::make_unique<ValueData<gpu::UniformRef>>(gpu::UniformRef{}));

	const example::Mesh mesh = example::make_cube_faceted();
	graph.set_producer(cube_mesh,
					   graph.add(std::make_unique<nodes::MeshNode>(std::span<const example::Vertex>(mesh.vertices),
																   std::span<const std::uint32_t>(mesh.indices),
																   cube_mesh)));

	graph.set_producer(instances_ref,
					   graph.add(std::make_unique<nodes::StorageBufferNode>(bodies_src, "instances", instances_ref)));
	graph.set_producer(light_ref,
					   graph.add(std::make_unique<nodes::UniformNode>(light_dir_src, "light", light_ref)));

	auto cube = std::make_unique<nodes::GraphicsNode>("demo/lit_instanced.vert", "demo/lit_directional.frag",
													  app.scene_color_format(), app.depth_format(), 0, app.screen(),
													  app.scene_image());
	cube->set_mesh(cube_mesh)
		.add_storage_buffer(instances_ref)
		.add_uniform(light_ref)
		.set_instances_from(instances_ref)
		.clear_color({0.04F, 0.05F, 0.08F, 1.0F})
		.push_constant<glm::mat4>(app.view_proj(), vk::ShaderStageFlagBits::eVertex);
	graph.set_producer(app.scene_image(), graph.add(std::move(cube)));

	// --- Sim thread: 60Hz model-matrix re-upload ------------------------------------
	std::atomic<bool>			 running{true};
	const std::vector<BodyState> body_states = initial_body_states();
	std::thread					 sim(
		[&]
		{
			using namespace std::chrono;
			const auto period = microseconds(16'667);
			auto	   next	  = steady_clock::now();
			const auto start  = next;
			while (running.load(std::memory_order_relaxed))
			{
				const float t		  = duration<float>(steady_clock::now() - start).count();
				auto		instances = build_instances(body_states, t);
				app.publish([&](Graph& g) { g.set(bodies_src, std::move(instances)); });
				next += period;
				std::this_thread::sleep_until(next);
			}
		});

	app.run();

	running.store(false);
	sim.join();
	return 0;
}
