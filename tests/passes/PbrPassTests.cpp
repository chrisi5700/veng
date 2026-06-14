//
// L5 pass test (review.md item 5): the metallic-roughness PbrPass. Analytic — a single quad facing
// the camera, lit head-on by one directional light, read back at the centre. We assert the BRDF's
// observable behaviour rather than exact radiance: a red material reads back red; emissive shows
// through with no light; and turning the light away collapses to the (dim) ambient term. Culling is
// disabled so the test exercises shading, not winding.
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <span>
#include <utility>
#include <vector>
#include <veng/assets/Texture.hpp>
#include <veng/context/Context.hpp>
#include <veng/culling/Clusters.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/gpu/Vertex.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/nodes/MeshNode.hpp>
#include <veng/passes/ClusteredLights.hpp>
#include <veng/passes/PbrPass.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/resources/ResourcePool.hpp>

using namespace veng::graph;
using veng::assets::ColorSpace;
using veng::assets::Texture;

namespace
{
constexpr veng::rhi::Format COLOR = veng::rhi::Format::RGBA8_UNORM; // linear target: read sampled radiance directly
constexpr veng::rhi::Format DEPTH = veng::rhi::Format::D32_SFLOAT;
constexpr std::uint32_t		SIDE  = 48;

veng::Context make_context()
{
	auto result = veng::Context::create("PbrPass Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

std::byte b(int v)
{
	return static_cast<std::byte>(static_cast<std::uint8_t>(v));
}

Texture solid(veng::Context& ctx, int r, int g, int bl, ColorSpace cs)
{
	const std::array<std::byte, 4> px{b(r), b(g), b(bl), b(255)};
	auto						   tex = Texture::from_pixels(ctx, px, 1, 1, cs);
	REQUIRE(tex.has_value());
	return std::move(tex.value());
}

// Everything a single-material PBR render needs, with the per-test knobs exposed.
struct PbrCase
{
	glm::vec4							 base_color_factor = glm::vec4(1.0F);
	float								 metallic		   = 0.0F;
	float								 roughness		   = 1.0F;
	glm::vec3							 emissive_factor   = glm::vec3(0.0F);
	glm::vec3							 light_direction = glm::vec3(0.0F, 0.0F, 1.0F); // toward the camera-facing quad
	veng::passes::AlphaMode				 alpha_mode		 = veng::passes::AlphaMode::Opaque;
	float								 alpha_cutoff = 0.5F;  // base_color_factor.a is the alpha the cutoff/blend uses
	bool								 light_off	  = false; // kill the directional light (isolate point lights)
	std::vector<veng::culling::GpuLight> point_lights{};	   // clustered point lights (wired when non-empty)
	veng::rhi::SampleCount				 samples = veng::rhi::SampleCount::X1; // MSAA level (clamped to device)
};

// Render the quad for `c` and return the centre pixel RGBA8.
std::array<std::uint8_t, 4> render_center(veng::Context& ctx, const PbrCase& c)
{
	const vk::Device device = ctx.device();

	// Default material textures: white base/MR/emissive/occlusion, flat (0,0,1) normal.
	Texture white		= solid(ctx, 255, 255, 255, ColorSpace::Srgb);	 // baseColor (sRGB)
	Texture white_lin	= solid(ctx, 255, 255, 255, ColorSpace::Linear); // MR + occlusion (data)
	Texture white_em	= solid(ctx, 255, 255, 255, ColorSpace::Srgb);	 // emissive (sRGB)
	Texture flat_normal = solid(ctx, 128, 128, 255, ColorSpace::Linear); // (0,0,1) after unpack

	// A quad in the XY plane facing +Z, with a +X tangent and corner UVs.
	const std::vector<veng::gpu::PbrVertex> verts{
		{.position = {-1, -1, 0}, .normal = {0, 0, 1}, .tangent = {1, 0, 0, 1}, .uv = {0, 1}},
		{.position = {1, -1, 0}, .normal = {0, 0, 1}, .tangent = {1, 0, 0, 1}, .uv = {1, 1}},
		{.position = {1, 1, 0}, .normal = {0, 0, 1}, .tangent = {1, 0, 0, 1}, .uv = {1, 0}},
		{.position = {-1, 1, 0}, .normal = {0, 0, 1}, .tangent = {1, 0, 0, 1}, .uv = {0, 0}},
	};
	const std::vector<std::uint32_t> indices{0, 1, 2, 0, 2, 3};

	Graph graph;
	auto  screen = graph.add_source<veng::rhi::Extent2D>(veng::rhi::Extent2D{SIDE, SIDE});

	// Camera at +Z looking at the origin; Vulkan Y-flip on the projection (engine convention).
	const glm::mat4 view = glm::lookAt(glm::vec3(0, 0, 2.2F), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
	glm::mat4		proj = glm::perspective(glm::radians(60.0F), 1.0F, 0.1F, 10.0F);
	proj[1][1] *= -1.0F;
	auto view_proj = graph.add_source<glm::mat4>(proj * view);
	auto view_src  = graph.add_source<glm::mat4>(view);
	auto proj_src  = graph.add_source<glm::mat4>(proj);
	auto eye	   = graph.add_source<glm::vec4>(glm::vec4(0, 0, 2.2F, 0));

	auto base_src	= graph.add_source<veng::gpu::ImageRef>(white.ref());
	auto normal_src = graph.add_source<veng::gpu::ImageRef>(flat_normal.ref());
	auto mr_src		= graph.add_source<veng::gpu::ImageRef>(white_lin.ref());
	auto em_src		= graph.add_source<veng::gpu::ImageRef>(white_em.ref());
	auto occ_src	= graph.add_source<veng::gpu::ImageRef>(white_lin.ref());

	const DataHandle mesh  = graph.add(std::make_unique<ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));
	auto			 model = graph.add_source<glm::mat4>(glm::mat4(1.0F));
	const DataHandle token = graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));

	const NodeHandle mesh_node = graph.add(std::make_unique<veng::nodes::MeshNode>(
		std::span<const veng::gpu::PbrVertex>(verts), std::span<const std::uint32_t>(indices), mesh));
	graph.set_producer(mesh, mesh_node);

	veng::passes::PbrConfig config;
	config.light_direction = c.light_direction;
	config.cull_mode	   = veng::rhi::CullMode::NONE; // shading test, not a winding test
	config.samples		   = c.samples;
	if (c.light_off)
	{
		config.light_intensity = 0.0F; // isolate the point lights: the directional term contributes nothing
	}
	veng::passes::PbrPass pass(graph, COLOR, DEPTH, screen, token, view_proj, eye, config);

	veng::passes::PbrMaterial material{.base_color		  = base_src,
									   .normal			  = normal_src,
									   .metal_rough		  = mr_src,
									   .emissive		  = em_src,
									   .occlusion		  = occ_src,
									   .base_color_factor = c.base_color_factor,
									   .metallic_factor	  = c.metallic,
									   .roughness_factor  = c.roughness,
									   .emissive_factor	  = c.emissive_factor,
									   .alpha_mode		  = c.alpha_mode,
									   .alpha_cutoff	  = c.alpha_cutoff};
	const std::uint32_t		  mat = pass.add_material(material);
	pass.add_object(mesh, model, mat);

	if (!c.point_lights.empty())
	{
		auto lights_src = graph.add_source<std::vector<veng::culling::GpuLight>>(c.point_lights);
		const veng::culling::ClusterGrid grid{.dims = {8, 8, 8}, .z_near = 0.1F, .z_far = 10.0F};
		const auto edges = veng::passes::wire_clustered_lights(graph, lights_src, view_src, proj_src, grid);
		pass.set_clustered_lights(view_src, edges.lights, edges.light_grid, edges.light_index, grid);
	}

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

	const auto* out_ref = dynamic_cast<ValueData<veng::gpu::ImageRef>*>(graph.get_data(token));
	REQUIRE(out_ref != nullptr);
	const veng::gpu::ImageRef out = out_ref->value();
	res_pool.transition_image(out.pool_id, cmd, vk::ImageLayout::eTransferSrcOptimal,
							  vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
	const auto region =
		vk::BufferImageCopy()
			.setImageSubresource(
				vk::ImageSubresourceLayers().setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1))
			.setImageExtent(vk::Extent3D{SIDE, SIDE, 1});
	cmd.copyImageToBuffer(ctx.rhi().image(out.texture), vk::ImageLayout::eTransferSrcOptimal, staging->buffer(),
						  region);
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
	const std::size_t			cc	   = (static_cast<std::size_t>(SIDE / 2) * SIDE + (SIDE / 2)) * 4;
	std::array<std::uint8_t, 4> out_px{pixels[cc + 0], pixels[cc + 1], pixels[cc + 2], pixels[cc + 3]};

	device.destroyFence(fence.value);
	device.destroyCommandPool(pool.value);
	return out_px;
}
} // namespace

TEST_CASE("PbrPass lights a diffuse material in its base colour", "[passes][pbr]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	const auto rgba = render_center(ctx, PbrCase{.base_color_factor = glm::vec4(1.0F, 0.0F, 0.0F, 1.0F)});
	REQUIRE(rgba[0] > 180);			  // red material, lit head-on -> strong red
	REQUIRE(rgba[1] < 40);			  // little green
	REQUIRE(rgba[2] < 40);			  // little blue
	REQUIRE(rgba[0] > rgba[1] + 120); // unmistakably red-dominant
}

TEST_CASE("PbrPass renders with MSAA and resolves to a single-sample image", "[passes][pbr][msaa]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	// 4x MSAA exercises the multisampled attachment + end-of-pass resolve through the real pass under
	// the validation layer (the gate would flag any attachment/resolve/sample-count mismatch). The
	// quad centre is interior, so the resolved colour matches the single-sample render: red reads red.
	const auto rgba = render_center(
		ctx, PbrCase{.base_color_factor = glm::vec4(1.0F, 0.0F, 0.0F, 1.0F), .samples = veng::rhi::SampleCount::X4});
	REQUIRE(rgba[0] > 180);
	REQUIRE(rgba[1] < 40);
	REQUIRE(rgba[2] < 40);
	REQUIRE(rgba[0] > rgba[1] + 120);
}

TEST_CASE("PbrPass emits emissive even with no incident light", "[passes][pbr]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	// Black base colour, green emissive, light pointing AWAY: only the emissive term survives.
	const auto rgba = render_center(ctx, PbrCase{.base_color_factor = glm::vec4(0.0F, 0.0F, 0.0F, 1.0F),
												 .emissive_factor	= glm::vec3(0.0F, 1.0F, 0.0F),
												 .light_direction	= glm::vec3(0.0F, 0.0F, -1.0F)});
	REQUIRE(rgba[1] > 180); // green emissive shows through
	REQUIRE(rgba[0] < 40);
	REQUIRE(rgba[2] < 40);
}

TEST_CASE("PbrPass darkens when the light turns away", "[passes][pbr]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	const auto lit	= render_center(ctx, PbrCase{.base_color_factor = glm::vec4(1.0F, 1.0F, 1.0F, 1.0F),
												 .light_direction	= glm::vec3(0.0F, 0.0F, 1.0F)});
	const auto dark = render_center(ctx, PbrCase{.base_color_factor = glm::vec4(1.0F, 1.0F, 1.0F, 1.0F),
												 .light_direction	= glm::vec3(0.0F, 0.0F, -1.0F)});
	REQUIRE(lit[0] > 150);			// lit face is bright
	REQUIRE(dark[0] < 40);			// back-lit collapses to the dim ambient term
	REQUIRE(lit[0] > dark[0] + 80); // N.L responds to light direction
}

TEST_CASE("PbrPass MASK discards fragments below the alpha cutoff", "[passes][pbr][alpha]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	// A red MASK material whose alpha (base_color_factor.a = 0.2) is below the 0.5 cutoff: every
	// fragment is discarded, so the centre reads back the clear colour (~5,8,13), not red.
	const auto culled = render_center(ctx, PbrCase{.base_color_factor = glm::vec4(1.0F, 0.0F, 0.0F, 0.2F),
												   .alpha_mode		  = veng::passes::AlphaMode::Mask,
												   .alpha_cutoff	  = 0.5F});
	REQUIRE(culled[0] < 30); // discarded -> clear, no red
	REQUIRE(culled[2] < 30);

	// The same material above the cutoff (alpha 0.9): the quad survives and reads back red.
	const auto kept = render_center(ctx, PbrCase{.base_color_factor = glm::vec4(1.0F, 0.0F, 0.0F, 0.9F),
												 .alpha_mode		= veng::passes::AlphaMode::Mask,
												 .alpha_cutoff		= 0.5F});
	REQUIRE(kept[0] > 150);				// kept -> red shows through
	REQUIRE(kept[0] > culled[0] + 100); // the cutoff visibly gates the fragment
}

TEST_CASE("PbrPass BLEND alpha-blends over the background", "[passes][pbr][alpha]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	// An opaque white quad lit head-on: the full shaded radiance at the centre.
	const auto opaque = render_center(ctx, PbrCase{.base_color_factor = glm::vec4(1.0F, 1.0F, 1.0F, 1.0F)});
	REQUIRE(opaque[0] > 150); // sanity: opaque is bright

	// The same quad as BLEND at 50% alpha blends with the dark clear behind it, so the centre is a
	// mix: clearly dimmer than opaque, yet clearly brighter than the bare clear (~5).
	const auto blended = render_center(ctx, PbrCase{.base_color_factor = glm::vec4(1.0F, 1.0F, 1.0F, 0.5F),
													.alpha_mode		   = veng::passes::AlphaMode::Blend});
	REQUIRE(blended[0] < opaque[0] - 40); // a 50% blend is visibly dimmer than opaque
	REQUIRE(blended[0] > 30);			  // but a real blend, well above the bare clear colour
}

TEST_CASE("PbrPass clustered point light illuminates an otherwise-dark fragment", "[passes][pbr][clustered]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	// Directional light off + tiny ambient: the quad is dark unless a point light reaches it. This
	// exercises the whole clustered path — froxel build, light->cluster cull, the SSBO uploads, and the
	// shader reconstructing its froxel to loop the light.
	PbrCase base{.base_color_factor = glm::vec4(1.0F, 1.0F, 1.0F, 1.0F), .light_off = true};

	const auto dark = render_center(ctx, base); // no point lights: only the dim ambient term
	REQUIRE(dark[0] < 25);

	// A bright white point light just in front of the quad (world +Z, between it and the camera).
	base.point_lights = {veng::culling::GpuLight{.position = glm::vec4(0.0F, 0.0F, 1.0F, 5.0F),
												 .color	   = glm::vec4(1.0F, 1.0F, 1.0F, 20.0F)}};
	const auto lit	  = render_center(ctx, base);
	REQUIRE(lit[0] > 90);			// the clustered point light clearly illuminates the centre
	REQUIRE(lit[0] > dark[0] + 60); // and it is the light doing it, not ambient

	// A point light pushed far outside its radius from the quad contributes ~nothing: the cull still
	// assigns it near its own position, but the shader's distance falloff kills it at the surface.
	base.point_lights = {veng::culling::GpuLight{.position = glm::vec4(0.0F, 8.0F, 1.0F, 1.0F),
												 .color	   = glm::vec4(1.0F, 1.0F, 1.0F, 20.0F)}};
	const auto away	  = render_center(ctx, base);
	REQUIRE(away[0] < dark[0] + 25); // out-of-range light leaves the fragment dark
}

TEST_CASE("PbrPass caches a static multi-material scene", "[passes][pbr][caching]")
{
	// The reactive thesis for PBR: a static scene (no input changes) re-plans to nothing after the
	// cold frame, so an OnDemand driver idles. Exercised with several materials sharing one mesh —
	// the shape the pbr_materials example uses — to catch a per-material always-dirty regression.
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			 ctx	= make_context();
	const vk::Device device = ctx.device();

	Texture white		= solid(ctx, 255, 255, 255, ColorSpace::Srgb);
	Texture white_lin	= solid(ctx, 255, 255, 255, ColorSpace::Linear);
	Texture flat_normal = solid(ctx, 128, 128, 255, ColorSpace::Linear);

	const std::vector<veng::gpu::PbrVertex> verts{
		{.position = {-1, -1, 0}, .normal = {0, 0, 1}, .tangent = {1, 0, 0, 1}, .uv = {0, 1}},
		{.position = {1, -1, 0}, .normal = {0, 0, 1}, .tangent = {1, 0, 0, 1}, .uv = {1, 1}},
		{.position = {1, 1, 0}, .normal = {0, 0, 1}, .tangent = {1, 0, 0, 1}, .uv = {1, 0}},
	};
	const std::vector<std::uint32_t> indices{0, 1, 2};

	Graph			 graph;
	auto			 screen	   = graph.add_source<veng::rhi::Extent2D>(veng::rhi::Extent2D{SIDE, SIDE});
	auto			 view_proj = graph.add_source<glm::mat4>(glm::mat4(1.0F));
	auto			 eye	   = graph.add_source<glm::vec4>(glm::vec4(0, 0, 2, 0));
	auto			 base	   = graph.add_source<veng::gpu::ImageRef>(white.ref());
	auto			 normal	   = graph.add_source<veng::gpu::ImageRef>(flat_normal.ref());
	auto			 mr		   = graph.add_source<veng::gpu::ImageRef>(white_lin.ref());
	const DataHandle mesh	   = graph.add(std::make_unique<ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));
	const DataHandle token	   = graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));
	graph.set_producer(
		mesh, graph.add(std::make_unique<veng::nodes::MeshNode>(std::span<const veng::gpu::PbrVertex>(verts),
																std::span<const std::uint32_t>(indices), mesh)));

	veng::passes::PbrConfig config;
	config.cull_mode = veng::rhi::CullMode::NONE;
	veng::passes::PbrPass pass(graph, COLOR, DEPTH, screen, token, view_proj, eye, config);
	for (int i = 0; i < 4; ++i)
	{
		const std::uint32_t material =
			pass.add_material(veng::passes::PbrMaterial{.base_color		 = base,
														.normal			 = normal,
														.metal_rough	 = mr,
														.emissive		 = base,
														.occlusion		 = mr,
														.metallic_factor = static_cast<float>(i) / 3.0F});
		auto model = graph.add_source<glm::mat4>(glm::mat4(1.0F));
		pass.add_object(mesh, model, material);
	}

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

	// Cold frame: the pass and the mesh upload are planned and run.
	const auto cold = graph.resolve(std::array{token});
	REQUIRE(cold.has_value());
	REQUIRE_FALSE(cold->empty());
	REQUIRE(graph.execute(*cold, scheduler, gpu_ctx));
	(void)cmd.end();

	// Second frame, nothing changed: the plan must be empty — the static scene caches.
	const auto held = graph.resolve(std::array{token});
	REQUIRE(held.has_value());
	REQUIRE(held->empty());

	device.destroyCommandPool(pool.value);
}
