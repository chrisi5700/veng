//
// AppLoop — the engine-side boilerplate of a windowed veng example, lifted out so the
// example's main.cpp is just graph wiring + sim code. AppLoop owns:
//   * a GLFW window + an OrbitCamera (the camera registers its own `view_proj` source);
//   * the Vulkan context, swapchain, resource pool, command manager, FrameExecutor;
//   * a Graph pre-populated with the well-known sources (`screen`, `swapchain_image`) and
//     the frame-closer tail (BlitNode + PresentNode) — the example only adds its scene
//     producer and points `scene_image()`'s producer at it.
//
// The example wires its scene like this:
//
//   example::AppLoop app(example::AppConfig{.title = "...", ...});
//   auto cube = std::make_unique<nodes::GraphicsNode>(
//       "demo/lit_instanced.vert", "demo/lit_directional.frag",
//       app.scene_color_format(), app.depth_format(), 0, app.screen(), app.scene_image());
//   cube->set_mesh(mesh_handle).push_constant<glm::mat4>(app.view_proj(), ...);
//   app.graph().set_producer(app.scene_image(), app.graph().add(std::move(cube)));
//
//   std::thread sim([&]{ while (running) { app.publish([&](auto& g){ g.set(bodies_src, ...); }); }});
//   app.run();           // returns when the window closes
//   running = false; sim.join();
//
// AppLoop's `run()` handles: input polling, SPACE = pacing toggle, OrbitCamera input,
// framebuffer resize, FrameExecutor dispatch + status handling, a 1Hz fps/cached% line.
//

#ifndef VENG_EXAMPLE_APP_LOOP_HPP
#define VENG_EXAMPLE_APP_LOOP_HPP

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <glm/glm.hpp>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include <veng/context/Context.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/managers/CommandManager.hpp>
#include <veng/managers/FrameExecutor.hpp>
#include <veng/managers/SwapchainManager.hpp>
#include <veng/nodes/PresentNode.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/rendergraph/nodes/Node.hpp>
#include <veng/resources/ResourcePool.hpp>
#include <veng/rhi/Enums.hpp>
#include <vulkan/vulkan.hpp>

#include "OrbitCamera.hpp"
#include "Window.hpp"

namespace example
{
struct AppConfig
{
	const char* title			 = "veng example";
	int			width			 = 1280;
	int			height			 = 720;
	glm::vec3	camera_target	 = {0.0F, 0.0F, 0.0F};
	float		camera_distance	 = 6.0F;
	float		camera_yaw		 = 0.6F;
	float		camera_pitch	 = 0.45F;
	veng::rhi::Format	depth_format	 = veng::rhi::Format::D32_SFLOAT;
	std::size_t frames_in_flight = 2;
	/// Render the scene into a linear-light HDR target (R16G16B16A16_SFLOAT) and resolve it to
	/// the swapchain through an ACES tonemap pass, instead of rendering straight into the
	/// swapchain's _SRGB format. Required for physically-correct PBR lighting (HDR headroom +
	/// a single linear->sRGB encode at the end). The flat-colour demos leave it off.
	bool  hdr	   = false;
	float exposure = 1.0F; // linear exposure applied before the tonemap curve (hdr only)
	/// Starting pacing for the loop; SPACE flips it at runtime.
	veng::FrameExecutor::Pacing pacing = veng::FrameExecutor::Pacing::OnDemand;
};

class AppLoop
{
	 public:
	explicit AppLoop(const AppConfig& config);
	~AppLoop();

	AppLoop(const AppLoop&)			   = delete;
	AppLoop& operator=(const AppLoop&) = delete;
	AppLoop(AppLoop&&)				   = delete;
	AppLoop& operator=(AppLoop&&)	   = delete;

	// --- The handles the example uses to build its scene producer --------------------

	[[nodiscard]] veng::graph::Graph& graph() noexcept { return m_graph; }
	[[nodiscard]] Window&			  window() noexcept { return m_window; }
	[[nodiscard]] OrbitCamera&		  camera() noexcept { return *m_camera; }
	[[nodiscard]] veng::Context&	  context() noexcept { return *m_ctx; }

	[[nodiscard]] veng::graph::TypedHandle<veng::rhi::Extent2D> screen() const noexcept { return m_screen; }
	[[nodiscard]] veng::graph::TypedHandle<glm::mat4>	 view_proj() const noexcept { return m_camera->view_proj(); }
	[[nodiscard]] veng::graph::DataHandle				 scene_image() const noexcept { return m_scene_image; }
	[[nodiscard]] veng::rhi::Format scene_color_format() const noexcept { return m_scene_color_format; }
	[[nodiscard]] veng::rhi::Format depth_format() const noexcept { return m_depth_format; }

	// --- Driving graph mutations from a sim / writer thread --------------------------

	/// Apply `mutator(graph)` under the graph mutex and wake the loop. The sim thread
	/// uses this to push new source values (the queue-at-frame-boundary semantics of
	/// `Graph::set` are preserved — the change is taken at the next planning pass).
	template <class F>
	void publish(F&& mutator)
	{
		{
			const std::scoped_lock lock(m_graph_mutex);
			std::forward<F>(mutator)(m_graph);
		}
		m_wake.notify_one();
	}

	// --- Main loop -------------------------------------------------------------------

	/// Run until the window closes. The optional `on_frame` hook fires after each
	/// successful render (status == Rendered) with the executed plan — examples use it for
	/// per-render telemetry (counting which optional subgraphs ran).
	/// Demand an extra sink edge each frame (alongside the frame-closer tail). A pass whose work
	/// lands outside the present chain — e.g. PickingPass's readback — registers its done-token
	/// here so the driver pulls that branch into the plan when it has work. Call at setup.
	void add_sink(veng::graph::DataHandle sink) { m_sinks.push_back(sink); }

	/// Register a predicate that forces full, presenting frames while it returns true, even in
	/// OnDemand pacing (which would otherwise idle on an unchanged graph). Use it for async work
	/// that completes only across several rendered frames and must not stall — e.g. a GPU readback
	/// whose result is delivered on frame retirement (`PickingPass::pending`). While any keep-alive
	/// is active the loop renders as if Continuous, so the pipeline keeps draining; when they all
	/// go false it returns to the configured pacing. Call at setup.
	void add_keep_alive(std::function<bool()> predicate) { m_keep_alive.push_back(std::move(predicate)); }

	/// `on_poll`, if set, runs once per loop iteration right after window polling (before the
	/// frame) — the seam for per-iteration input that must be caught regardless of pacing, e.g.
	/// click-to-pick. `on_frame` (above) runs only on rendered frames.
	void run(std::function<void(const veng::graph::FramePlan&)> on_frame = {}, std::function<void()> on_poll = {});

	 private:
	void rebuild_swapchain(vk::Extent2D extent);

	AppConfig									  m_config;
	Window										  m_window;
	std::unique_ptr<veng::Context>				  m_ctx;
	std::unique_ptr<veng::SwapchainManager>		  m_swap;
	veng::rhi::Format									  m_depth_format;
	veng::rhi::Format									  m_scene_color_format; // HDR when config.hdr, else the swapchain format
	std::unique_ptr<veng::ResourcePool>			  m_pool;
	std::unique_ptr<veng::CommandManager>		  m_commands;
	veng::graph::InlineScheduler				  m_scheduler;
	veng::graph::Graph							  m_graph;
	veng::graph::TypedHandle<veng::rhi::Extent2D> m_screen;
	veng::graph::TypedHandle<veng::gpu::ImageRef> m_swapchain_image;
	veng::graph::DataHandle						  m_scene_image;
	veng::graph::DataHandle						  m_tonemapped_image; // hdr only: ACES resolve of m_scene_image
	veng::graph::DataHandle						  m_presented_image;
	veng::graph::DataHandle						  m_frame_done;
	veng::nodes::PresentNode*					  m_present_ptr = nullptr;
	std::unique_ptr<veng::FrameExecutor>		  m_executor;
	std::unique_ptr<OrbitCamera>				  m_camera;
	std::mutex									  m_graph_mutex;
	std::condition_variable						  m_wake;
	std::vector<veng::graph::DataHandle>		  m_sinks; // frame-closer + pass-registered sinks
	std::vector<std::function<bool()>>			  m_keep_alive; // force presenting frames while any is true
};
} // namespace example

#endif // VENG_EXAMPLE_APP_LOOP_HPP
