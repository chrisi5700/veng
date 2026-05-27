//
// veng reactive renderer demo — a spinning cube on a real swapchain (design.md §1, §L4,
// §L5). Two threads exercise the demand-driven caching thesis on live GPU work:
//
//   * a writer thread spins the cube's rotation angle at a steady 60 Hz (a fresh angle
//     every tick), so the cube re-renders 60 times a second;
//   * the render thread (this, the main thread — GLFW requires it) is paced one of two
//     ways, toggled live with SPACE:
//       - Continuous: render every loop iteration (the swapchain image is fed as a
//         per-frame dirty pulse) — uncapped; present fps far exceeds the 60 Hz cube
//         re-render rate, the rest being cached re-blits of the last scene. Max power.
//       - OnDemand: render only when a data source actually changed — an empty FramePlan
//         means there is nothing to do, so the thread idles (no acquire/submit/present).
//         Here present fps tracks the 60 Hz data rate exactly: same picture, no waste.
//
// The graph tail is declarative: the cube renders into a scene ImageData; a BlitNode
// copies it into the acquired swapchain image; a PresentNode closes the frame (submit +
// present). The pacing flag only decides whether the swapchain image is fed as a dirty
// pulse (Continuous) or merely as an execute-time value via set_now (OnDemand).
//

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <GLFW/glfw3.h> // for the GLFW_KEY_* constant used by the live pacing toggle
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <print>
#include <span>
#include <thread>
#include <utility>
#include <vector>
#include <veng/context/Context.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/gpu/UniformRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/managers/CommandManager.hpp>
#include <veng/managers/QueueKind.hpp>
#include <veng/managers/SwapchainManager.hpp>
#include <veng/nodes/BlitNode.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/nodes/MeshNode.hpp>
#include <veng/nodes/PresentNode.hpp>
#include <veng/nodes/UniformNode.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/Image.hpp>

#include "Window.hpp"

using namespace veng;
using namespace veng::graph;

namespace
{
// How the render thread is paced — the "electricity" knob (toggle live with SPACE):
//   Continuous: render every loop iteration (the swapchain image is fed as a per-frame
//               dirty pulse) — uncapped, maximum throughput, maximum power.
//   OnDemand:   render only when a CPU-side data source actually changed — a static scene
//               costs nothing (empty FramePlan -> the thread idles, no acquire/submit).
enum class Pacing : std::uint8_t
{
	Continuous,
	OnDemand
};

bool plan_contains(const FramePlan& plan, NodeHandle node)
{
	for (const NodeHandle handle : plan.nodes())
	{
		if (handle == node)
		{
			return true;
		}
	}
	return false;
}

// A cube vertex — must match shaders/demo/mesh.vert.slang (`float3 position; float3 color;`,
// tightly packed, which is what Slang reflects: a 24-byte stride).
struct Vertex
{
	glm::vec3 position;
	glm::vec3 color;
};

// The cube as real geometry: 24 vertices (4 per face so each face carries its own flat
// color) + 36 indices. The corners and per-face colors match the old SV_VertexID cube.vert,
// so the picture is identical — only now the vertices live in a real VkBuffer.
struct CubeMesh
{
	std::vector<Vertex>		   vertices;
	std::vector<std::uint32_t> indices;
};

CubeMesh make_cube()
{
	constexpr std::array<glm::vec3, 8> corners{glm::vec3{-0.5F, -0.5F, -0.5F}, glm::vec3{0.5F, -0.5F, -0.5F},
											   glm::vec3{0.5F, 0.5F, -0.5F},   glm::vec3{-0.5F, 0.5F, -0.5F},
											   glm::vec3{-0.5F, -0.5F, 0.5F},  glm::vec3{0.5F, -0.5F, 0.5F},
											   glm::vec3{0.5F, 0.5F, 0.5F},	   glm::vec3{-0.5F, 0.5F, 0.5F}};

	// Per face: its four corners (wound for two triangles 0,1,2 / 0,2,3) and its flat color.
	struct Face
	{
		std::array<std::uint32_t, 4> corner;
		glm::vec3					 color;
	};
	constexpr std::array<Face, 6> faces{Face{{0, 1, 2, 3}, {1.0F, 0.25F, 0.25F}}, // front  (-z)
										Face{{5, 4, 7, 6}, {0.25F, 1.0F, 0.35F}}, // back   (+z)
										Face{{4, 0, 3, 7}, {0.30F, 0.45F, 1.0F}}, // left   (-x)
										Face{{1, 5, 6, 2}, {1.0F, 0.85F, 0.25F}}, // right  (+x)
										Face{{4, 5, 1, 0}, {1.0F, 0.35F, 1.0F}},  // bottom (-y)
										Face{{3, 2, 6, 7}, {0.30F, 1.0F, 1.0F}}}; // top    (+y)

	CubeMesh mesh;
	for (const Face& face : faces)
	{
		const auto base = static_cast<std::uint32_t>(mesh.vertices.size());
		for (const std::uint32_t corner : face.corner)
		{
			mesh.vertices.push_back(Vertex{.position = corners[corner], .color = face.color});
		}
		for (const std::uint32_t offset : {0U, 1U, 2U, 0U, 2U, 3U})
		{
			mesh.indices.push_back(base + offset);
		}
	}
	return mesh;
}

// A 1x1 opaque-black texture left in SHADER_READ_ONLY: the composite's "ring" input while the
// outline is OFF. Sampling black and adding it leaves the scene untouched, so disabling the
// outline takes no shader branch — just rebinding this one input (which, being off the
// outline branch, drops silhouette+blur out of the demanded plan entirely). Created with a
// one-shot clear + transition.
std::optional<Image> make_black_texture(const Context& ctx, vk::Format format)
{
	auto image = Image::create(ctx.allocator(), ctx.device(), vk::Extent2D{1, 1}, format,
							   vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst);
	if (!image.has_value())
	{
		return std::nullopt;
	}
	const vk::Device device = ctx.device();
	const auto		 pool =
		device.createCommandPool(vk::CommandPoolCreateInfo().setQueueFamilyIndex(ctx.queue_indices().graphics));
	const auto				cmds = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo()
																	 .setCommandPool(pool.value)
																	 .setLevel(vk::CommandBufferLevel::ePrimary)
																	 .setCommandBufferCount(1));
	const vk::CommandBuffer cmd	 = cmds.value.front();
	(void)cmd.begin(vk::CommandBufferBeginInfo().setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
	CommandManager::image_barrier(cmd, image->image(), vk::ImageLayout::eUndefined,
								  vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eTopOfPipe,
								  vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eTransfer,
								  vk::AccessFlagBits2::eTransferWrite);
	const auto range =
		vk::ImageSubresourceRange().setAspectMask(vk::ImageAspectFlagBits::eColor).setLevelCount(1).setLayerCount(1);
	cmd.clearColorImage(image->image(), vk::ImageLayout::eTransferDstOptimal,
						vk::ClearColorValue(std::array{0.0F, 0.0F, 0.0F, 1.0F}), range);
	CommandManager::image_barrier(cmd, image->image(), vk::ImageLayout::eTransferDstOptimal,
								  vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits2::eTransfer,
								  vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eFragmentShader,
								  vk::AccessFlagBits2::eShaderSampledRead);
	(void)cmd.end();
	const auto fence = device.createFence({});
	(void)ctx.graphics_queue().submit(vk::SubmitInfo().setCommandBuffers(cmd), fence.value);
	(void)device.waitForFences(fence.value, vk::True, UINT64_MAX);
	device.destroyFence(fence.value);
	device.destroyCommandPool(pool.value);
	return std::move(image.value());
}
} // namespace

int main()
{
	Logger::instance().set_level(spdlog::level::warn);

	demo::Window window("veng — reactive spinning cube", 1280, 720);

	auto ctx_result = Context::create("veng reactive renderer", window.required_extensions(),
									  [&window](VkInstance instance) { return window.create_surface(instance); });
	if (!ctx_result.has_value())
	{
		std::println("Failed to create Vulkan context");
		return 1;
	}
	Context ctx = std::move(ctx_result.value());

	auto swap_result = SwapchainManager::create(ctx, window.framebuffer_extent(), 1);
	if (!swap_result.has_value())
	{
		std::println("Failed to create swapchain: {}", vk::to_string(swap_result.error()));
		return 1;
	}
	SwapchainManager swap = std::move(swap_result.value());

	// The cube renders into a scene target of the swapchain's format, so the present blit
	// is a straight copy; the depth target is D32.
	const vk::Format	 scene_color  = swap.format();
	constexpr vk::Format depth_format = vk::Format::eD32Sfloat;

	// The composite's ring input falls back to this 1x1 black texture when the outline is off.
	auto black = make_black_texture(ctx, scene_color);
	if (!black.has_value())
	{
		std::println("Failed to create fallback texture");
		return 1;
	}
	const gpu::ImageRef black_ref{
		.image = black->image(), .view = black->view(), .extent = {1, 1}, .format = scene_color};

	// --- The reactive graph, built once -------------------------------------------------
	//   screen + angle -> cube -> scene_image ----------------------------------┐
	//   cube_mesh + mvp -> silhouette -> blur_h -> ring (the outline branch) ----┤
	//   scene_image + (ring | black) -> composite -> outlined_image             │
	//   outlined_image + swapchain_image (each frame) -> blit -> presented_image │
	//   presented_image -> present (submit + present) -> frame_done  (the sink)  ┘
	// Press O to rebind the composite's ring input between `ring_image` and `black`: with black
	// nothing demands silhouette/blur_h/ring, so they leave the plan — the outline disables by
	// design, not by a gate.
	Graph graph;
	auto  screen		  = graph.add_source<vk::Extent2D>(swap.extent());
	auto  angle			  = graph.add_source<float>(0.0F);
	auto  swapchain_image = graph.add_source<gpu::ImageRef>(gpu::ImageRef{});
	auto  black_source	  = graph.add_source<gpu::ImageRef>(black_ref);

	const DataHandle scene_image	  = graph.add(std::make_unique<ValueData<gpu::ImageRef>>(gpu::ImageRef{}));
	const DataHandle silhouette_image = graph.add(std::make_unique<ValueData<gpu::ImageRef>>(gpu::ImageRef{}));
	const DataHandle blurred_h_image  = graph.add(std::make_unique<ValueData<gpu::ImageRef>>(gpu::ImageRef{}));
	const DataHandle ring_image		  = graph.add(std::make_unique<ValueData<gpu::ImageRef>>(gpu::ImageRef{}));
	const DataHandle outlined_image	  = graph.add(std::make_unique<ValueData<gpu::ImageRef>>(gpu::ImageRef{}));
	const DataHandle presented_image  = graph.add(std::make_unique<ValueData<gpu::ImageRef>>(gpu::ImageRef{}));
	const DataHandle frame_done		  = graph.add(std::make_unique<ValueData<int>>(0));
	const DataHandle cube_mesh		  = graph.add(std::make_unique<ValueData<gpu::MeshRef>>(gpu::MeshRef{}));
	const DataHandle cube_tint		  = graph.add(std::make_unique<ValueData<gpu::UniformRef>>(gpu::UniformRef{}));

	// The cube's geometry is now real vertex data: a MeshNode uploads 24 vertices + 36 indices
	// into GPU buffers once (then caches forever), publishing a MeshRef the cube node draws.
	const CubeMesh mesh = make_cube();
	auto mesh_node		= std::make_unique<nodes::MeshNode>(std::span<const Vertex>(mesh.vertices),
															std::span<const std::uint32_t>(mesh.indices), cube_mesh);
	graph.set_producer(cube_mesh, graph.add(std::move(mesh_node)));

	// A global tint, uploaded into a uniform buffer by a UniformNode and bound by name into the
	// cube node's descriptor set: the lit fragment shader multiplies each face color by it. A
	// real descriptor-set uniform, fed declaratively as data — the user's `add_uniform`.
	auto tint	   = graph.add_source<glm::vec4>(glm::vec4{1.0F, 0.95F, 0.85F, 1.0F}); // warm tint
	auto tint_node = std::make_unique<nodes::UniformNode>(tint, "tint", cube_tint);
	graph.set_producer(cube_tint, graph.add(std::move(tint_node)));

	// The cube is no longer a bespoke node: a pure transform turns (angle, screen) into an
	// MVP matrix, and a generic GraphicsNode draws the cube shaders with that MVP as a
	// push-constant. Change the angle -> the transform recomputes -> the GraphicsNode
	// re-renders; hold it -> both are cached. "Cube-ness" lives entirely in the shaders +
	// this matrix edge.
	auto mvp = graph.add_transform(
		[](const float& spin, const vk::Extent2D& size) -> glm::mat4
		{
			const float aspect = static_cast<float>(size.width) / static_cast<float>(size.height);
			glm::mat4	proj   = glm::perspective(glm::radians(55.0F), aspect, 0.1F, 20.0F);
			proj[1][1] *= -1.0F; // Vulkan y-down clip space
			const glm::mat4 view =
				glm::lookAt(glm::vec3(2.4F, 1.7F, 2.4F), glm::vec3(0.0F), glm::vec3(0.0F, 1.0F, 0.0F));
			const glm::mat4 model = glm::rotate(glm::mat4(1.0F), spin, glm::normalize(glm::vec3(0.3F, 1.0F, 0.2F)));
			return proj * view * model;
		},
		angle, screen);

	// Buffer-backed geometry + a descriptor uniform: the mesh shader reads position + color
	// from the vertex buffer (vs the SV_VertexID cube.vert) and the lit fragment shader tints
	// it by the uniform; the draw count comes from the mesh, so vertex_count is 0. The scene is
	// now sampled by the composite (not blitted directly), so it rests in SHADER_READ_ONLY.
	auto cube = std::make_unique<nodes::GraphicsNode>("demo/mesh.vert", "demo/lit.frag", scene_color, depth_format, 0,
													  screen, scene_image);
	cube->set_mesh(cube_mesh)
		.add_uniform(cube_tint)
		.clear_color({0.02F, 0.02F, 0.05F, 1.0F})
		.final_layout(vk::ImageLayout::eShaderReadOnlyOptimal)
		.push_constant<glm::mat4>(mvp, vk::ShaderStageFlagBits::eVertex);
	const NodeHandle cube_node = graph.add(std::move(cube));
	graph.set_producer(scene_image, cube_node);

	// --- The outline branch: silhouette -> separable gaussian -> ring ----------------------
	// 1/extent for the blur taps (in UV space), so the glow width is resolution-independent.
	auto texel = graph.add_transform(
		[](const vk::Extent2D& size) -> glm::vec2
		{ return glm::vec2(1.0F / static_cast<float>(size.width), 1.0F / static_cast<float>(size.height)); }, screen);

	// Silhouette: the cube mesh as a solid white mask (same MVP as the cube), left readable.
	auto silhouette = std::make_unique<nodes::GraphicsNode>("demo/mesh.vert", "demo/silhouette.frag", scene_color,
															vk::Format::eUndefined, 0, screen, silhouette_image);
	silhouette->set_mesh(cube_mesh)
		.final_layout(vk::ImageLayout::eShaderReadOnlyOptimal)
		.push_constant<glm::mat4>(mvp, vk::ShaderStageFlagBits::eVertex);
	const NodeHandle silhouette_node = graph.add(std::move(silhouette));
	graph.set_producer(silhouette_image, silhouette_node);

	// Horizontal gaussian of the silhouette (fullscreen pass sampling it).
	auto blur_h = std::make_unique<nodes::GraphicsNode>("demo/fullscreen.vert", "demo/blur_h.frag", scene_color,
														vk::Format::eUndefined, 3, screen, blurred_h_image);
	blur_h->add_sampled_image(silhouette_image, "silhouette")
		.final_layout(vk::ImageLayout::eShaderReadOnlyOptimal)
		.push_constant<glm::vec2>(texel, vk::ShaderStageFlagBits::eFragment);
	const NodeHandle blur_h_node = graph.add(std::move(blur_h));
	graph.set_producer(blurred_h_image, blur_h_node);

	// Vertical gaussian + ring extraction (blurred minus sharp silhouette, tinted).
	auto ring = std::make_unique<nodes::GraphicsNode>("demo/fullscreen.vert", "demo/ring.frag", scene_color,
													  vk::Format::eUndefined, 3, screen, ring_image);
	ring->add_sampled_image(blurred_h_image, "blurredH")
		.add_sampled_image(silhouette_image, "silhouette")
		.final_layout(vk::ImageLayout::eShaderReadOnlyOptimal)
		.push_constant<glm::vec2>(texel, vk::ShaderStageFlagBits::eFragment);
	const NodeHandle ring_node = graph.add(std::move(ring));
	graph.set_producer(ring_image, ring_node);

	// Composite (fullscreen): scene + ring. The ring input starts on `black` (outline off) and
	// O rebinds it to `ring_image`. Output is TRANSFER_SRC for the present blit.
	auto composite = std::make_unique<nodes::GraphicsNode>("demo/fullscreen.vert", "demo/composite.frag", scene_color,
														   vk::Format::eUndefined, 3, screen, outlined_image);
	composite->add_sampled_image(scene_image, "scene").add_sampled_image(black_source, "ring");
	auto*			 composite_ptr	= composite.get();
	const NodeHandle composite_node = graph.add(std::move(composite));
	graph.set_producer(outlined_image, composite_node);

	auto blit = std::make_unique<nodes::BlitNode>(outlined_image, swapchain_image, presented_image,
												  vk::ImageLayout::ePresentSrcKHR);
	graph.set_producer(presented_image, graph.add(std::move(blit)));

	auto  present	  = std::make_unique<nodes::PresentNode>(swap, presented_image, frame_done);
	auto* present_ptr = present.get();
	graph.set_producer(frame_done, graph.add(std::move(present)));

	CommandManager	commands(ctx);
	InlineScheduler scheduler;

	std::mutex				graph_mutex;
	std::condition_variable wake; // OnDemand: the writer pokes this when it mutates a source
	std::atomic<bool>		running{true};
	Pacing					pacing = Pacing::OnDemand; // start power-frugal; SPACE toggles

	// Writer thread: a continuous 60 Hz spin — a fresh angle every tick. set() queues under
	// the lock; the render thread applies it in resolve() (also under the lock) — a clean
	// frame-boundary snapshot, with the expensive execute running outside the lock. The
	// notify wakes the render thread in OnDemand mode (it sleeps when nothing has changed).
	std::thread writer(
		[&]
		{
			using namespace std::chrono;
			const auto period	   = microseconds(16'667); // ~60 Hz
			auto	   next		   = steady_clock::now();
			float	   angle_value = 0.0F;
			while (running.load(std::memory_order_relaxed))
			{
				angle_value += 0.04F;
				{
					const std::scoped_lock lock(graph_mutex);
					graph.set(angle, angle_value);
				}
				wake.notify_one();
				next += period;
				std::this_thread::sleep_until(next);
			}
		});

	const vk::Device device			 = ctx.device();
	std::uint64_t	 presents		 = 0;
	std::uint64_t	 cube_renders	 = 0;
	std::uint64_t	 outline_renders = 0;
	auto			 stats_at		 = std::chrono::steady_clock::now();

	const auto rebuild_for_resize = [&](vk::Extent2D size)
	{
		(void)device.waitIdle();
		if (!swap.rebuild(size).has_value())
		{
			return false;
		}
		const std::scoped_lock lock(graph_mutex);
		graph.set(screen, swap.extent());
		return true;
	};

	bool space_was_down = false;
	bool o_was_down		= false;
	bool outline_on		= false;
	while (!window.should_close())
	{
		window.poll();

		// SPACE toggles pacing live, so the fps/power gap is visible on a single run.
		const bool space = window.key_down(GLFW_KEY_SPACE);
		if (space && !space_was_down)
		{
			pacing = (pacing == Pacing::OnDemand) ? Pacing::Continuous : Pacing::OnDemand;
			std::println("\n>> pacing: {}", pacing == Pacing::Continuous ? "Continuous (render every frame)"
																		 : "OnDemand (render only on data change)");
			std::fflush(stdout);
		}
		space_was_down = space;

		// O toggles the outline. We only rebind the composite's ring input between the live ring
		// image and the static black texture — the rest is the engine's doing: with black bound,
		// nothing demands the silhouette/blur/ring passes, so they drop out of the plan entirely
		// (and the additive composite emits the plain scene). Disabled by design, not by a gate.
		const bool o_key = window.key_down(GLFW_KEY_O);
		if (o_key && !o_was_down)
		{
			outline_on				  = !outline_on;
			const DataHandle ring_src = outline_on ? ring_image : static_cast<DataHandle>(black_source);
			{
				const std::scoped_lock lock(graph_mutex);
				composite_ptr->set_sampled_image("ring", ring_src);
			}
			wake.notify_one(); // nudge the OnDemand thread so the change shows immediately
			std::println("\n>> outline: {}", outline_on ? "ON" : "off");
			std::fflush(stdout);
		}
		o_was_down = o_key;

		vk::Extent2D fb = window.framebuffer_extent();
		if (fb.width == 0 || fb.height == 0)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(10)); // minimized
			continue;
		}

		// A resize dirties the screen source (a real data change), driving a render in either mode.
		if (fb.width != swap.extent().width || fb.height != swap.extent().height)
		{
			if (!rebuild_for_resize(fb))
			{
				break;
			}
		}

		// OnDemand: decide before spending anything. Resolve the demanded plan WITHOUT
		// touching the swapchain image — if nothing upstream changed it is empty, so we idle
		// (no acquire / submit / present) until the writer pokes us. This is the whole point:
		// a static scene costs a sleeping thread and nothing else.
		FramePlan plan;
		if (pacing == Pacing::OnDemand)
		{
			std::unique_lock lock(graph_mutex);
			auto			 resolved = graph.resolve(std::array{frame_done});
			if (!resolved.has_value())
			{
				std::println("resolve failed");
				break;
			}
			if (resolved->empty())
			{
				wake.wait_for(lock, std::chrono::milliseconds(16)); // sleep; timeout keeps events responsive
				continue;
			}
			plan = std::move(resolved.value());
		}

		// Acquire the next image; this also waits out (and resets) the slot's in-flight
		// fence, so the previous frame has retired and the command pool is safe to recycle.
		auto acquired = swap.acquire(0);
		if (!acquired.has_value())
		{
			std::println("acquire failed: {}", vk::to_string(acquired.error()));
			break;
		}
		if (!acquired.value().has_value())
		{
			if (!rebuild_for_resize(window.framebuffer_extent()))
			{
				break;
			}
			continue;
		}
		commands.reset_frame(0);
		const SwapchainManager::Frame frame = acquired.value().value();
		const gpu::ImageRef			  ref{.image		  = swap.image(frame.image_index),
										  .extent		  = swap.extent(),
										  .format		  = swap.format(),
										  .index		  = frame.image_index,
										  .acquire_wait	  = frame.image_available,
										  .present_signal = frame.render_finished,
										  .in_flight	  = frame.in_flight};

		if (pacing == Pacing::OnDemand)
		{
			// Value only: the plan above already committed to render off a real data change;
			// just hand the blit the freshly-acquired image. set_now never marks it dirty.
			graph.set_now(swapchain_image, ref);
		}
		else
		{
			// Continuous: feed the image as a per-frame dirty pulse so blit + present always
			// run, then resolve.
			const std::scoped_lock lock(graph_mutex);
			graph.set(swapchain_image, ref);
			auto resolved = graph.resolve(std::array{frame_done});
			if (!resolved.has_value())
			{
				std::println("resolve failed");
				break;
			}
			plan = std::move(resolved.value());
		}

		auto cmd = commands.begin(QueueKind::Graphics, 0);
		if (!cmd.has_value())
		{
			std::println("command buffer begin failed");
			break;
		}

		// Execute records the cube + blit and, as the sink, the PresentNode ends the
		// command buffer, submits it, and presents — all GPU/queue work lives in the graph.
		gpu::GpuExecContext gpu(graph, ctx, cmd.value(), 0);
		graph.execute(plan, scheduler, gpu);

		if (present_ptr->out_of_date() && !rebuild_for_resize(window.framebuffer_extent()))
		{
			break;
		}

		++presents;
		if (plan_contains(plan, cube_node))
		{
			++cube_renders;
		}
		if (plan_contains(plan, ring_node)) // the outline branch ran this frame
		{
			++outline_renders;
		}

		const auto now = std::chrono::steady_clock::now();
		if (now - stats_at >= std::chrono::seconds(1))
		{
			const double seconds = std::chrono::duration<double>(now - stats_at).count();
			std::println("[{}] present {:7.0f} fps | cube {:6.0f} fps | outline {:>3} {:6.0f} fps | cached {:5.1f}%",
						 pacing == Pacing::Continuous ? "continuous" : "on-demand ",
						 static_cast<double>(presents) / seconds, static_cast<double>(cube_renders) / seconds,
						 outline_on ? "ON" : "off", static_cast<double>(outline_renders) / seconds,
						 presents > 0
							 ? 100.0 * static_cast<double>(presents - cube_renders) / static_cast<double>(presents)
							 : 0.0);
			std::fflush(stdout); // a live monitor: flush each tick rather than at exit
			presents		= 0;
			cube_renders	= 0;
			outline_renders = 0;
			stats_at		= now;
		}
	}

	running.store(false);
	writer.join();
	(void)device.waitIdle();
	return 0;
}
