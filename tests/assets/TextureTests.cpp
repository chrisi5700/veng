//
// L2 asset test (review.md items 3, 4, 8): the texture loader, generated mip chain, configurable
// sampler, and sRGB-vs-linear decode. All analytic — synthesized pixels with known answers, read
// back through a fullscreen sample pass:
//   1. a uniform texture samples back its exact colour (load + upload + sample path),
//   2. forcing the sampler to the 1x1 mip returns the average of the source texels (mip
//      generation + a non-default sampler LOD clamp),
//   3. the same bytes loaded as sRGB sample back darker than as linear (hardware sRGB decode).
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>
#include <veng/assets/Texture.hpp>
#include <veng/context/Context.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/resources/ResourcePool.hpp>
#include <veng/rhi/SamplerDesc.hpp>

using namespace veng::graph;
using veng::assets::ColorSpace;
using veng::assets::Texture;

namespace
{
constexpr veng::rhi::Format COLOR = veng::rhi::Format::RGBA8_UNORM; // linear target: sampled values written as-is
constexpr std::uint32_t		SIDE  = 32;

veng::Context make_context()
{
	auto result = veng::Context::create("Texture Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

std::byte byte_of(int v)
{
	return static_cast<std::byte>(static_cast<std::uint8_t>(v));
}

// Render a fullscreen pass sampling `tex` with `sampler` into a SIDE x SIDE linear target and
// return the centre pixel's RGBA8. Mirrors the readback boilerplate in MeshNodeTests.
std::array<std::uint8_t, 4> sample_center(veng::Context& ctx, const Texture& tex,
										  const veng::rhi::SamplerDesc& sampler)
{
	const vk::Device device = ctx.device();

	Graph			 graph;
	auto			 screen	 = graph.add_source<veng::rhi::Extent2D>(veng::rhi::Extent2D{SIDE, SIDE});
	auto			 tex_src = graph.add_source<veng::gpu::ImageRef>(tex.ref());
	const DataHandle token	 = graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));

	auto node = std::make_unique<veng::nodes::GraphicsNode>("demo/fullscreen.vert", "tests/slice/sample.frag", COLOR,
															veng::rhi::Format::UNDEFINED, 3, screen, token);
	node->add_sampled_image(tex_src, "source");
	node->set_sampler(sampler);
	auto*			 node_ptr	 = node.get();
	const NodeHandle node_handle = graph.add(std::move(node));
	graph.set_producer(token, node_handle);

	auto staging =
		veng::Buffer::create(ctx.allocator(), ctx.rhi(), static_cast<vk::DeviceSize>(SIDE) * SIDE * 4,
							 vk::BufferUsageFlagBits::eTransferDst, vma::MemoryUsage::eAuto,
							 vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessRandom);
	REQUIRE(staging.has_value());

	const auto pool =
		device.createCommandPool(vk::CommandPoolCreateInfo().setQueueFamilyIndex(ctx.queue_indices().graphics));
	REQUIRE(pool.result == vk::Result::eSuccess);
	const auto cmds = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo()
														.setCommandPool(pool.value)
														.setLevel(vk::CommandBufferLevel::ePrimary)
														.setCommandBufferCount(1));
	REQUIRE(cmds.result == vk::Result::eSuccess);
	const vk::CommandBuffer cmd = cmds.value.front();
	REQUIRE(cmd.begin(vk::CommandBufferBeginInfo().setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)) ==
			vk::Result::eSuccess);

	veng::ResourcePool res_pool(device, ctx.rhi(), ctx.allocator(), 1);
	res_pool.begin_frame(0);
	veng::gpu::GpuExecContext gpu_ctx(graph, ctx, res_pool, cmd, 0);
	InlineScheduler			  scheduler;
	const auto				  plan = graph.resolve(std::array{token});
	REQUIRE(plan.has_value());
	REQUIRE(graph.execute(*plan, scheduler, gpu_ctx));
	REQUIRE(node_ptr->scene() != nullptr);

	const auto* readback_ref = dynamic_cast<ValueData<veng::gpu::ImageRef>*>(graph.get_data(token));
	res_pool.transition_image(readback_ref->value().pool_id, cmd, vk::ImageLayout::eTransferSrcOptimal,
							  vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
	const auto region =
		vk::BufferImageCopy()
			.setImageSubresource(
				vk::ImageSubresourceLayers().setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1))
			.setImageExtent(vk::Extent3D{SIDE, SIDE, 1});
	cmd.copyImageToBuffer(node_ptr->scene()->image(), vk::ImageLayout::eTransferSrcOptimal, staging->buffer(), region);
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {},
						vk::MemoryBarrier()
							.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
							.setDstAccessMask(vk::AccessFlagBits::eHostRead),
						{}, {});
	REQUIRE(cmd.end() == vk::Result::eSuccess);

	const auto fence = device.createFence({});
	REQUIRE(fence.result == vk::Result::eSuccess);
	REQUIRE(ctx.graphics_queue().submit(vk::SubmitInfo().setCommandBuffers(cmd), fence.value) == vk::Result::eSuccess);
	REQUIRE(device.waitForFences(fence.value, vk::True, UINT64_MAX) == vk::Result::eSuccess);

	const auto*					pixels = static_cast<const std::uint8_t*>(staging->mapped());
	const std::size_t			c	   = (static_cast<std::size_t>(SIDE / 2) * SIDE + (SIDE / 2)) * 4;
	std::array<std::uint8_t, 4> out{pixels[c + 0], pixels[c + 1], pixels[c + 2], pixels[c + 3]};

	device.destroyFence(fence.value);
	device.destroyCommandPool(pool.value);
	return out;
}

bool near(std::uint8_t value, int target, int tolerance = 3)
{
	return std::abs(static_cast<int>(value) - target) <= tolerance;
}
} // namespace

TEST_CASE("a loaded texture samples back its pixels", "[assets][texture]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	// 8x8 solid orange (255,128,0,255). Uniform, so any UV / filtering returns the same colour.
	std::vector<std::byte> pixels;
	for (int i = 0; i < 8 * 8; ++i)
	{
		pixels.insert(pixels.end(), {byte_of(255), byte_of(128), byte_of(0), byte_of(255)});
	}
	auto tex = Texture::from_pixels(ctx, pixels, 8, 8, ColorSpace::Linear);
	REQUIRE(tex.has_value());

	const auto rgba = sample_center(ctx, *tex, veng::rhi::SamplerDesc::render_target());
	REQUIRE(near(rgba[0], 255));
	REQUIRE(near(rgba[1], 128));
	REQUIRE(near(rgba[2], 0));
}

TEST_CASE("generated mips average the source texels", "[assets][texture][mips]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	// 2x2 with four primaries: red, green, blue, white. The 1x1 mip is their average:
	// R=(255+0+0+255)/4=128, G=(0+255+0+255)/4=128, B=(0+0+255+255)/4=128.
	const std::vector<std::byte> pixels{
		byte_of(255), byte_of(0),	byte_of(0),	  byte_of(255), // red
		byte_of(0),	  byte_of(255), byte_of(0),	  byte_of(255), // green
		byte_of(0),	  byte_of(0),	byte_of(255), byte_of(255), // blue
		byte_of(255), byte_of(255), byte_of(255), byte_of(255), // white
	};
	auto tex = Texture::from_pixels(ctx, pixels, 2, 2, ColorSpace::Linear);
	REQUIRE(tex.has_value());
	REQUIRE(tex->mip_levels() == 2); // 2x2 -> levels {2x2, 1x1}

	// Force the sampler to mip level 1 (the 1x1 average) regardless of screen-space derivatives.
	auto forced_lod	   = veng::rhi::SamplerDesc::texture();
	forced_lod.min_lod = 1.0F;
	forced_lod.max_lod = 1.0F;

	const auto rgba = sample_center(ctx, *tex, forced_lod);
	REQUIRE(near(rgba[0], 128, 4)); // averaged down the mip chain
	REQUIRE(near(rgba[1], 128, 4));
	REQUIRE(near(rgba[2], 128, 4));
}

TEST_CASE("an sRGB texture decodes to linear on sample", "[assets][texture][srgb]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	// Solid mid-grey byte 188 (~0.737 as UNORM). As sRGB it decodes to ~0.504 linear (~129 of
	// 255); as linear it stays 0.737 (~188). So the sRGB sample reads back markedly darker.
	std::vector<std::byte> pixels;
	for (int i = 0; i < 8 * 8; ++i)
	{
		pixels.insert(pixels.end(), {byte_of(188), byte_of(188), byte_of(188), byte_of(255)});
	}

	auto linear_tex = Texture::from_pixels(ctx, pixels, 8, 8, ColorSpace::Linear);
	auto srgb_tex	= Texture::from_pixels(ctx, pixels, 8, 8, ColorSpace::Srgb);
	REQUIRE(linear_tex.has_value());
	REQUIRE(srgb_tex.has_value());

	const auto linear = sample_center(ctx, *linear_tex, veng::rhi::SamplerDesc::render_target());
	const auto srgb	  = sample_center(ctx, *srgb_tex, veng::rhi::SamplerDesc::render_target());

	REQUIRE(near(linear[0], 188));	   // linear data passes straight through
	REQUIRE(near(srgb[0], 129, 4));	   // sRGB decoded to linear on read
	REQUIRE(srgb[0] < linear[0] - 40); // unmistakably darker — the decode happened
}
