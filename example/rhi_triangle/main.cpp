//
// rhi_triangle — a flat, node-free look at the veng RHI.
//
// Every other example drives the reactive render-graph (nodes, GpuExecContext, AppLoop). This one
// deliberately uses NONE of that. It records a single triangle straight through the RHI so the RHI
// surface itself is on display for review: the rhi::Device handle registry and the
// rhi::CommandEncoder recording API.
//
// What the RHI is — and is not. The RHI is veng's *containment + recording* layer over a single
// Vulkan backend. It does NOT create the device, allocate command buffers, or submit — those are
// the driver bottom (veng::Context, the command pool, the queue), shown here in plain `vk::` on
// purpose, so the boundary is honest. What the RHI owns is the part that used to leak Vulkan into
// "abstract" code: opaque resource handles (Image/Buffer/GraphicsPipeline register themselves into
// rhi::Device and hand back a TextureHandle/BufferHandle/PipelineHandle), and command recording in
// engine vocabulary (rhi::CommandEncoder: transition / bind_pipeline / set_viewport_scissor / draw /
// copy, all taking handles + rhi enums, never a vk::Image / vk::Pipeline / vk::ImageLayout).
//
// It renders headlessly and writes rhi_triangle.ppm (a red triangle on black).
//

#include <array>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/pipelines/GraphicsPipeline.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/resources/Image.hpp>
#include <veng/rhi/CommandEncoder.hpp>
#include <veng/rhi/Convert.hpp>
#include <veng/shader/Shader.hpp>

namespace
{
constexpr std::uint32_t		SIDE  = 64;
constexpr veng::rhi::Format COLOR = veng::rhi::Format::RGBA8_UNORM;
} // namespace

int main()
{
	// --- Driver setup (plain Vulkan — the layer the RHI sits on) ----------------------------------
	// The engine Context owns the device/queue/allocator AND the rhi::Device handle registry.
	// Headless: no window, no surface.
	auto ctx_result = veng::Context::create("rhi_triangle");
	if (!ctx_result.has_value())
	{
		std::fprintf(stderr, "rhi_triangle: failed to create a Vulkan context\n");
		return 1;
	}
	veng::Context	   ctx	  = std::move(ctx_result.value());
	const vk::Device   device = ctx.device();
	veng::rhi::Device& rhi	  = ctx.rhi(); // <-- the RHI: the handle registry every resource registers with

	// --- Resources: each registers itself into the RHI and hands back an opaque handle ------------
	// The color target. Image::create registers (image, view) in `rhi` and exposes handle(); from
	// here on the target is referred to ONLY by its opaque TextureHandle — the encoder resolves it.
	auto color = veng::Image::create(ctx.allocator(), device, rhi, vk::Extent2D{SIDE, SIDE}, veng::rhi::to_vk(COLOR),
									 vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc);
	if (!color.has_value())
	{
		std::fprintf(stderr, "rhi_triangle: failed to create the color target\n");
		return 1;
	}
	const veng::rhi::TextureHandle target = color->handle();

	// The pipeline. The reflection-driven builder compiles the shaders and, on build(), registers the
	// (pipeline, layout) pair into `rhi` — so a draw binds it by an opaque PipelineHandle, not vk::Pipeline.
	auto vert = veng::Shader::create_shader(device, "tests/slice/triangle.vert");
	auto frag = veng::Shader::create_shader(device, "tests/slice/triangle.frag");
	if (!vert.has_value() || !frag.has_value())
	{
		std::fprintf(stderr, "rhi_triangle: failed to load the triangle shaders\n");
		return 1;
	}
	const std::array<veng::rhi::Format, 1> color_formats{COLOR};
	auto pipeline = veng::GraphicsPipelineBuilder(vert.value(), frag.value())
						.color_formats(color_formats)
						.rasterization(veng::rhi::PolygonMode::FILL, veng::rhi::CullMode::NONE,
									   veng::rhi::FrontFace::COUNTER_CLOCKWISE)
						.build(ctx);
	if (!pipeline.has_value())
	{
		std::fprintf(stderr, "rhi_triangle: failed to build the pipeline\n");
		return 1;
	}
	const veng::rhi::PipelineHandle pso = pipeline->handle();

	// A host-visible buffer to read the result back into (also registers a BufferHandle).
	auto staging =
		veng::Buffer::create(ctx.allocator(), rhi, static_cast<vk::DeviceSize>(SIDE) * SIDE * 4,
							 vk::BufferUsageFlagBits::eTransferDst, vma::MemoryUsage::eAuto,
							 vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessRandom);
	if (!staging.has_value() || staging->mapped() == nullptr)
	{
		std::fprintf(stderr, "rhi_triangle: failed to create the staging buffer\n");
		return 1;
	}

	// A one-shot command buffer (command-buffer allocation is a driver concern, not the RHI's).
	const auto cmd_pool =
		device.createCommandPool(vk::CommandPoolCreateInfo().setQueueFamilyIndex(ctx.queue_indices().graphics));
	const auto				cmd_alloc = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo()
																		  .setCommandPool(cmd_pool.value)
																		  .setLevel(vk::CommandBufferLevel::ePrimary)
																		  .setCommandBufferCount(1));
	const vk::CommandBuffer cmd		  = cmd_alloc.value.front();
	(void)cmd.begin(vk::CommandBufferBeginInfo().setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

	// ============================ THE RHI ============================
	// Wrap the frame's command buffer; from here every recorded command goes through the encoder in
	// engine vocabulary, resolving handles via `rhi`.
	veng::rhi::CommandEncoder enc(cmd, rhi);

	// Move the target into a color-attachment state. The RHI maps the usage to the concrete layout +
	// barrier — no node ever names vk::ImageLayout::eColorAttachmentOptimal.
	enc.transition(target, veng::rhi::TextureUsage::UNDEFINED, veng::rhi::TextureUsage::COLOR_ATTACHMENT);

	// Begin dynamic rendering. NOTE — this is the ONE recording op the encoder does not yet wrap:
	// render-pass attachment setup lives in the engine's RenderTargetSet (which the encoder's begin
	// path defers to). The encoder exposes vk() as the documented escape hatch for exactly this case.
	// Everything after beginRendering is pure RHI again.
	const auto color_attach = vk::RenderingAttachmentInfo()
								  .setImageView(color->view())
								  .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
								  .setLoadOp(vk::AttachmentLoadOp::eClear)
								  .setStoreOp(vk::AttachmentStoreOp::eStore)
								  .setClearValue(vk::ClearValue().setColor(
									  vk::ClearColorValue(std::array<float, 4>{0.0F, 0.0F, 0.0F, 1.0F})));
	enc.vk().beginRendering(vk::RenderingInfo()
								.setRenderArea(vk::Rect2D({0, 0}, vk::Extent2D{SIDE, SIDE}))
								.setLayerCount(1)
								.setColorAttachments(color_attach));

	enc.bind_pipeline(pso); // PipelineHandle -> vk::Pipeline, resolved inside the RHI
	enc.set_viewport_scissor(veng::rhi::Extent2D{SIDE, SIDE});
	enc.draw(3, 1); // the slice shader fabricates a triangle from SV_VertexID
	enc.end_rendering();

	// Hand the rendered image to a transfer reader, then copy it to the host buffer — both by handle.
	enc.transition(target, veng::rhi::TextureUsage::COLOR_ATTACHMENT, veng::rhi::TextureUsage::TRANSFER_SRC);
	enc.copy_texture_to_host_buffer(target, staging->handle(), veng::rhi::Extent2D{SIDE, SIDE});
	// ========================== end RHI =============================

	(void)cmd.end();

	// Submit + wait (the queue is the driver bottom; the RHI does not own submission).
	const auto fence = device.createFence({});
	(void)ctx.graphics_queue().submit(vk::SubmitInfo().setCommandBuffers(cmd), fence.value);
	(void)device.waitForFences(fence.value, vk::True, UINT64_MAX);

	// --- Inspect: write a PPM and print the center pixel ------------------------------------------
	const auto* px = static_cast<const std::uint8_t*>(staging->mapped());
	if (std::FILE* file = std::fopen("rhi_triangle.ppm", "wb"); file != nullptr)
	{
		(void)std::fprintf(file, "P6\n%u %u\n255\n", SIDE, SIDE);
		for (std::uint32_t i = 0; i < SIDE * SIDE; ++i)
		{
			(void)std::fputc(static_cast<int>(px[(i * 4) + 0]), file);
			(void)std::fputc(static_cast<int>(px[(i * 4) + 1]), file);
			(void)std::fputc(static_cast<int>(px[(i * 4) + 2]), file);
		}
		(void)std::fclose(file);
	}
	const std::size_t center = ((static_cast<std::size_t>(SIDE) / 2 * SIDE) + (SIDE / 2)) * 4;
	std::printf("rhi_triangle: center pixel = (%u, %u, %u) — wrote rhi_triangle.ppm\n",
				static_cast<unsigned>(px[center + 0]), static_cast<unsigned>(px[center + 1]),
				static_cast<unsigned>(px[center + 2]));

	device.destroyFence(fence.value);
	device.destroyCommandPool(cmd_pool.value);
	return 0;
}
