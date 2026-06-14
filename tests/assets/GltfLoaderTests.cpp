//
// Integration test (review.md item 6): load real Khronos glTF Sample Models through the fastgltf
// loader and render them with PbrPass. The .glb files are fetched at configure time (gitignored,
// SHA256-pinned via tests/assets/models.cmake); any model that is absent (offline build) makes its
// section SKIP rather than fail. Assertions are structural ("parsed N primitives / M materials,
// loaded textures") plus, for the acceptance models, "renders a non-empty frame with zero
// validation errors" — exact pixels are not pinned for real assets.
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <veng/assets/GltfLoader.hpp>
#include <veng/context/Context.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/passes/PbrPass.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/resources/ResourcePool.hpp>

using namespace veng::graph;

namespace
{
constexpr veng::rhi::Format COLOR = veng::rhi::Format::RGBA8_UNORM;
constexpr veng::rhi::Format DEPTH = veng::rhi::Format::D32_SFLOAT;
constexpr std::uint32_t		SIDE  = 64;

veng::Context make_context()
{
	auto result = veng::Context::create("GltfLoader Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

std::string model_path(const std::string& name)
{
	return std::string(VENG_MODELS_DIR) + "/" + name + ".glb";
}

bool model_present(const std::string& name)
{
	return std::filesystem::exists(model_path(name));
}

// Wire a loaded model into a PbrPass (materials + objects index-aligned), frame the camera on its
// bounds, render once, and return how many pixels differ from the clear colour (i.e. were drawn).
std::size_t render_and_count(veng::Context& ctx, Graph& graph, const veng::assets::GltfModel& model)
{
	const vk::Device device = ctx.device();

	auto			 screen = graph.add_source<vk::Extent2D>(vk::Extent2D{SIDE, SIDE});
	const DataHandle token	= graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));

	// Frame the camera on the model's bounding sphere from a 3/4 angle.
	const glm::vec3 center	= model.center();
	const float		radius	= std::max(model.radius(), 0.05F);
	const float		fov		= glm::radians(45.0F);
	const float		dist	= (radius / std::tan(fov * 0.5F)) * 1.5F;
	const glm::vec3 eye_pos = center + glm::normalize(glm::vec3(0.5F, 0.4F, 1.0F)) * dist;
	const glm::mat4 view	= glm::lookAt(eye_pos, center, glm::vec3(0, 1, 0));
	glm::mat4		proj	= glm::perspective(fov, 1.0F, dist * 0.01F, dist * 4.0F + radius);
	proj[1][1] *= -1.0F;
	auto view_proj = graph.add_source<glm::mat4>(proj * view);
	auto eye	   = graph.add_source<glm::vec4>(glm::vec4(eye_pos, 0.0F));

	veng::passes::PbrPass pass(graph, COLOR, DEPTH, screen, token, view_proj, eye);
	// add_material returns its index; we add in model order so pass index == model index, hence the
	// discard. add_object below uses the model's own material index directly.
	for (const veng::assets::GltfMaterialDesc& m : model.materials)
	{
		static_cast<void>(pass.add_material(
			veng::passes::PbrMaterial{.base_color		  = m.base_color,
									  .normal			  = m.normal,
									  .metal_rough		  = m.metal_rough,
									  .emissive			  = m.emissive,
									  .occlusion		  = m.occlusion,
									  .base_color_factor  = m.base_color_factor,
									  .metallic_factor	  = m.metallic_factor,
									  .roughness_factor	  = m.roughness_factor,
									  .normal_scale		  = m.normal_scale,
									  .occlusion_strength = m.occlusion_strength,
									  .emissive_factor	  = m.emissive_factor,
									  .alpha_mode		  = static_cast<veng::passes::AlphaMode>(m.alpha_mode),
									  .alpha_cutoff		  = m.alpha_cutoff}));
	}
	for (const veng::assets::GltfPrimitive& p : model.primitives)
	{
		pass.add_object(p.mesh, p.model, p.material);
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

	const auto* pixels = static_cast<const std::uint8_t*>(staging->mapped());
	std::size_t drawn  = 0;
	for (std::uint32_t i = 0; i < SIDE * SIDE; ++i)
	{
		// Default clear colour (0.02,0.03,0.05) is ~ (5,8,13) in this UNORM target. Count any pixel
		// that differs from it — geometry was drawn there, whether brightly lit or dark metallic.
		const int dr = std::abs(static_cast<int>(pixels[(i * 4) + 0]) - 5);
		const int dg = std::abs(static_cast<int>(pixels[(i * 4) + 1]) - 8);
		const int db = std::abs(static_cast<int>(pixels[(i * 4) + 2]) - 13);
		if (dr + dg + db > 8)
		{
			++drawn;
		}
	}

	device.destroyFence(fence.value);
	device.destroyCommandPool(pool.value);
	return drawn;
}
} // namespace

TEST_CASE("glTF loader parses Box geometry and renders it", "[assets][gltf][integration]")
{
	if (!model_present("Box"))
	{
		SKIP("Box.glb not fetched (offline build)");
	}
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto  ctx = make_context();
	Graph graph;
	auto  model = veng::assets::load_gltf(ctx, graph, model_path("Box"));
	REQUIRE(model.has_value());
	REQUIRE(model->primitives.size() == 1); // a single cube primitive
	REQUIRE(model->materials.size() >= 1);
	REQUIRE(render_and_count(ctx, graph, *model) > 100); // the cube fills a good part of the frame
}

TEST_CASE("glTF loader uploads BoxTextured's texture and renders it", "[assets][gltf][integration]")
{
	if (!model_present("BoxTextured"))
	{
		SKIP("BoxTextured.glb not fetched (offline build)");
	}
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto  ctx = make_context();
	Graph graph;
	auto  model = veng::assets::load_gltf(ctx, graph, model_path("BoxTextured"));
	REQUIRE(model.has_value());
	REQUIRE(model->primitives.size() == 1);
	REQUIRE(model->textures.size() > 3); // 3 shared defaults + at least the baseColor texture
	REQUIRE(render_and_count(ctx, graph, *model) > 100);
}

TEST_CASE("glTF loader parses NormalTangentTest with tangents", "[assets][gltf][integration]")
{
	if (!model_present("NormalTangentTest"))
	{
		SKIP("NormalTangentTest.glb not fetched (offline build)");
	}
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto  ctx = make_context();
	Graph graph;
	auto  model = veng::assets::load_gltf(ctx, graph, model_path("NormalTangentTest"));
	REQUIRE(model.has_value());
	REQUIRE(model->primitives.size() >= 1);
	REQUIRE(model->textures.size() > 3); // normal map present
}

TEST_CASE("glTF loader flattens MetalRoughSpheres' multi-mesh node tree", "[assets][gltf][integration]")
{
	if (!model_present("MetalRoughSpheres"))
	{
		SKIP("MetalRoughSpheres.glb not fetched (offline build)");
	}
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto  ctx = make_context();
	Graph graph;
	auto  model = veng::assets::load_gltf(ctx, graph, model_path("MetalRoughSpheres"));
	REQUIRE(model.has_value());
	// This asset is several meshes under distinct nodes (the metallic/roughness sweep lives in a
	// texture, not separate materials) — a real exercise of the node-hierarchy flatten.
	REQUIRE(model->primitives.size() >= 2);
	REQUIRE(render_and_count(ctx, graph, *model) > 200);
}

TEST_CASE("glTF loader renders DamagedHelmet (acceptance)", "[assets][gltf][integration][acceptance]")
{
	if (!model_present("DamagedHelmet"))
	{
		SKIP("DamagedHelmet.glb not fetched (offline build)");
	}
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto  ctx = make_context();
	Graph graph;
	auto  model = veng::assets::load_gltf(ctx, graph, model_path("DamagedHelmet"));
	REQUIRE(model.has_value());
	REQUIRE(model->primitives.size() == 1);
	REQUIRE(model->textures.size() >= 5);				 // baseColor, normal, MR, emissive, occlusion (+ defaults)
	REQUIRE(render_and_count(ctx, graph, *model) > 200); // the full PBR helmet renders, no validation errors
}

TEST_CASE("load_gltf_lods builds a LOD chain per primitive", "[assets][gltf][integration]")
{
	if (!model_present("DamagedHelmet"))
	{
		SKIP("DamagedHelmet.glb not fetched (offline build)");
	}
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto  ctx = make_context();
	Graph graph;

	const std::array<veng::assets::LodLevel, 3> levels{{{.target_ratio = 1.0F},
														{.target_ratio = 0.5F, .max_error = 0.05F},
														{.target_ratio = 0.2F, .max_error = 0.2F, .aggressive = true}}};
	auto model = veng::assets::load_gltf_lods(ctx, graph, model_path("DamagedHelmet"), levels);
	REQUIRE(model.has_value());
	REQUIRE(model->primitives.size() == 1);
	for (const veng::assets::GltfLodPrimitive& prim : model->primitives)
	{
		CHECK(prim.lods.size() == 3); // one eagerly-uploaded mesh edge per level
	}
	CHECK(model->materials.size() >= 1);
	CHECK(model->radius() > 0.0F);

	// No levels ⇒ one full-detail mesh per primitive.
	auto full = veng::assets::load_gltf_lods(ctx, graph, model_path("DamagedHelmet"), {});
	REQUIRE(full.has_value());
	for (const veng::assets::GltfLodPrimitive& prim : full->primitives)
	{
		CHECK(prim.lods.size() == 1);
	}
}
