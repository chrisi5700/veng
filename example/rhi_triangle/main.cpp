//
// rhi_triangle — render a triangle through the veng RHI with NO Vulkan in sight.
//
// This is the whole point of a render hardware interface: you should be able to create resources,
// record a pass, and submit it without ever naming a Vulkan type. This example does exactly that —
// grep this file for a Vulkan handle and you will not find one. It uses none of the reactive
// render-graph (no nodes, no GpuExecContext, no AppLoop) either; it is the RHI, by itself.
//
// The flow mirrors a real frame:
//   1. bootstrap a device          (Context::create -> ctx.rhi(), the only engine call)
//   2. create resources            (rhi::Device::create_texture / create_buffer -> opaque handles)
//   3. build a pipeline            (the reflection-driven builder; vk-free, yields a PipelineHandle)
//   4. record a pass               (rhi::CommandEncoder: transition / begin_rendering / draw / copy)
//   5. submit + read back          (rhi::Device::submit, then read the host-mapped buffer)
//
// It renders headlessly and writes rhi_triangle.ppm (a red triangle on black).
//

#include <array>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/pipelines/GraphicsPipeline.hpp>
#include <veng/rhi/CommandEncoder.hpp>
#include <veng/rhi/Device.hpp>
#include <veng/shader/Shader.hpp>

namespace
{
constexpr std::uint32_t		SIDE  = 64;
constexpr veng::rhi::Format COLOR = veng::rhi::Format::RGBA8_UNORM;
} // namespace

int main()
{
	using namespace veng;

	// 1. Bootstrap a device. Context is the engine's device bootstrap; ctx.rhi() is the RHI device.
	//    From here on, everything is rhi:: — no Vulkan.
	auto ctx_result = Context::create("rhi_triangle");
	if (!ctx_result.has_value())
	{
		std::fprintf(stderr, "rhi_triangle: failed to create a device\n");
		return 1;
	}
	Context		 ctx = std::move(ctx_result.value());
	rhi::Device& dev = ctx.rhi();

	// 2. Resources. The RHI owns them; we hold only opaque handles. The color target is rendered
	//    into and then read back; the staging buffer is host-visible so we can map and inspect it.
	auto target =
		dev.create_texture({.extent = {SIDE, SIDE},
							.format = COLOR,
							.usage	= rhi::TextureUsageFlags::COLOR_ATTACHMENT | rhi::TextureUsageFlags::TRANSFER_SRC});
	auto staging = dev.create_buffer({.size	  = static_cast<std::uint64_t>(SIDE) * SIDE * 4,
									  .usage  = rhi::BufferUsageFlags::TRANSFER_DST,
									  .memory = rhi::MemoryAccess::HOST_VISIBLE});
	if (!target.has_value() || !staging.has_value())
	{
		std::fprintf(stderr, "rhi_triangle: failed to create resources\n");
		return 1;
	}

	// 3. Pipeline. The reflection-driven builder is the engine's pipeline layer — vk-free, and it
	//    registers the pipeline into the RHI, handing back an opaque PipelineHandle.
	auto vert = Shader::create_shader(ctx, "tests/slice/triangle.vert");
	auto frag = Shader::create_shader(ctx, "tests/slice/triangle.frag");
	if (!vert.has_value() || !frag.has_value())
	{
		std::fprintf(stderr, "rhi_triangle: failed to load shaders\n");
		return 1;
	}
	const std::array<rhi::Format, 1> color_formats{COLOR};
	auto pipeline = GraphicsPipelineBuilder(vert.value(), frag.value())
						.color_formats(color_formats)
						.rasterization(rhi::PolygonMode::FILL, rhi::CullMode::NONE, rhi::FrontFace::COUNTER_CLOCKWISE)
						.build(ctx);
	if (!pipeline.has_value())
	{
		std::fprintf(stderr, "rhi_triangle: failed to build the pipeline\n");
		return 1;
	}

	// 4. Record the whole pass through the RHI — every command in engine vocabulary.
	rhi::CommandEncoder enc = dev.begin_commands();
	enc.transition(target.value(), rhi::TextureUsage::UNDEFINED, rhi::TextureUsage::COLOR_ATTACHMENT);

	const std::array<rhi::ColorAttachment, 1> attachments{rhi::ColorAttachment{
		.texture = target.value(), .load = rhi::LoadOp::CLEAR, .clear_color = {0.0F, 0.0F, 0.0F, 1.0F}}};
	enc.begin_rendering(rhi::RenderPassDesc{.area = {SIDE, SIDE}, .color_attachments = attachments});
	enc.bind_pipeline(pipeline->handle());
	enc.set_viewport_scissor({SIDE, SIDE});
	enc.draw(3, 1); // the slice shader fabricates a triangle from SV_VertexID
	enc.end_rendering();

	enc.transition(target.value(), rhi::TextureUsage::COLOR_ATTACHMENT, rhi::TextureUsage::TRANSFER_SRC);
	enc.copy_texture_to_host_buffer(target.value(), staging.value(), {SIDE, SIDE});

	// 5. Submit (blocks until the GPU is done) and read the result back from the mapped buffer.
	if (!dev.submit(enc).has_value())
	{
		std::fprintf(stderr, "rhi_triangle: submit failed\n");
		return 1;
	}

	const auto* px = static_cast<const std::uint8_t*>(dev.mapped(staging.value()));
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

	// The device frees its resources when the Context is destroyed; freeing early is just tidy.
	dev.destroy_buffer(staging.value());
	dev.destroy_texture(target.value());
	return 0;
}
