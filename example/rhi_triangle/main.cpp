//
// rhi_triangle — present a triangle through the veng RHI with NO Vulkan in sight.
//
// This is the whole point of a render hardware interface: you should be able to create a swapchain,
// record a pass, and present it to a window without ever naming a Vulkan type. grep this file for a
// `vk::` and you will not find one. It uses none of the reactive render-graph (no nodes, no
// GpuExecContext, no AppLoop) either — it is the RHI, by itself, driving a real present loop.
//
// The flow is a real frame, repeated until the window closes:
//   1. open a window + bootstrap a device   (Window + Context::create -> ctx.rhi())
//   2. create a swapchain over its surface   (rhi::Swapchain::create -> per-image TextureHandles)
//   3. build a pipeline                      (the reflection-driven builder; vk-free PipelineHandle)
//   4. per frame: acquire -> record -> present
//        rhi::Swapchain::acquire   -> the image to draw into, as a handle
//        rhi::CommandEncoder       -> transition / begin_rendering / draw / transition-to-present
//        rhi::Swapchain::present   -> submit + queue for display
//
// The only seam that still speaks the platform's tongue is creating the window surface: GLFW hands
// us a `VkSurfaceKHR` from a `VkInstance` (the C handles, not the `vk::` C++ wrappers). Window-system
// glue is inherently platform-coupled and lives behind Window/Context; the rendering above never is.
//

#include <array>
#include <veng/context/Context.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/pipelines/GraphicsPipeline.hpp>
#include <veng/rhi/CommandEncoder.hpp>
#include <veng/rhi/Device.hpp>
#include <veng/rhi/Swapchain.hpp>
#include <veng/shader/Shader.hpp>

#include "Window.hpp"

namespace
{
constexpr int WIDTH	 = 800;
constexpr int HEIGHT = 600;
} // namespace

int main()
{
	using namespace veng;
	Logger& log = Logger::instance();

	// 1. Open a window and bootstrap a device that can present to its surface. The surface factory is
	//    the one place the C Vulkan handle leaks through — pure window-system glue.
	example::Window window("rhi_triangle", WIDTH, HEIGHT);
	auto			ctx_result = Context::create("rhi_triangle", window.required_extensions(),
								 [&window](VkInstance instance) { return window.create_surface(instance); });
	if (!ctx_result.has_value())
	{
		Logger::instance().error("rhi_triangle: failed to create a windowed device");
		return 1;
	}
	Context		 ctx = std::move(ctx_result.value());
	rhi::Device& dev = ctx.rhi();

	// 2. A swapchain over the surface. From here on everything is rhi:: — no Vulkan.
	auto swap_result = rhi::Swapchain::create(ctx, {.width = WIDTH, .height = HEIGHT});
	if (!swap_result.has_value())
	{
		Logger::instance().error("rhi_triangle: failed to create the swapchain");
		return 1;
	}
	rhi::Swapchain swap = std::move(swap_result.value());

	// 3. Pipeline. The reflection-driven builder is the engine's pipeline layer — vk-free, and it
	//    registers the pipeline into the RHI, handing back an opaque PipelineHandle. Its one color
	//    target is the swapchain's own format, so the two always agree.
	auto vert = Shader::create_shader(ctx, "tests/slice/triangle.vert");
	auto frag = Shader::create_shader(ctx, "tests/slice/triangle.frag");
	if (!vert.has_value() || !frag.has_value())
	{
		Logger::instance().error("rhi_triangle: failed to load shaders");
		return 1;
	}
	const std::array<rhi::Format, 1> color_formats{swap.format()};
	auto							 pipeline =
		GraphicsPipelineBuilder(vert.value(), frag.value())
			.color_formats(color_formats)
			.rasterization(rhi::PolygonMode::FILL, rhi::CullMode::NONE, rhi::FrontFace::COUNTER_CLOCKWISE)
			.build(ctx);
	if (!pipeline.has_value())
	{
		Logger::instance().error("rhi_triangle: failed to build the pipeline");
		return 1;
	}

	log.info("rhi_triangle: presenting a triangle through the RHI — close the window to exit");

	// 4. The present loop. Every GPU command is RHI vocabulary; resize is handled by recreating the
	//    swapchain whenever acquire/present reports it out of date.
	while (!window.should_close())
	{
		window.poll();

		auto frame = swap.acquire();
		if (!frame.has_value())
		{
			log.error("rhi_triangle: acquire failed: {}", rhi::to_string(frame.error()));
			break;
		}
		if (!frame->has_value()) // out of date (e.g. a resize) — rebuild and try again next iteration
		{
			(void)swap.recreate();
			continue;
		}
		const rhi::Swapchain::Frame target = frame->value();

		rhi::CommandEncoder enc = dev.begin_commands();
		enc.transition(target.target, rhi::TextureUsage::UNDEFINED, rhi::TextureUsage::COLOR_ATTACHMENT);

		const std::array<rhi::ColorAttachment, 1> attachments{rhi::ColorAttachment{
			.texture = target.target, .load = rhi::LoadOp::CLEAR, .clear_color = {0.05F, 0.05F, 0.08F, 1.0F}}};
		enc.begin_rendering(rhi::RenderPassDesc{.area = target.extent, .color_attachments = attachments});
		enc.bind_pipeline(pipeline->handle());
		enc.set_viewport_scissor(target.extent);
		enc.draw(3, 1); // the slice shader fabricates a triangle from SV_VertexID
		enc.end_rendering();

		enc.transition(target.target, rhi::TextureUsage::COLOR_ATTACHMENT, rhi::TextureUsage::PRESENT);

		auto presented = swap.present(enc, target);
		if (!presented.has_value())
		{
			log.error("rhi_triangle: present failed: {}", rhi::to_string(presented.error()));
			break;
		}
		if (presented.value()) // out of date after present — rebuild for the next frame
		{
			(void)swap.recreate();
		}
	}

	log.info("rhi_triangle: window closed, exiting");
	return 0;
}
