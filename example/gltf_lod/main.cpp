//
// glTF LOD viewer — loads a glTF (the DamagedHelmet sample by default) as a decimated LOD chain via
// veng::assets::load_gltf_lods and selects the level at runtime from the model's screen coverage: one
// CoverageLodNode on the model bounds drives a MeshSelectorNode per primitive (so every part of the
// model switches together), all rendered with the model's authored materials through PbrPass.
//
// Unlike the STL viewer, glTF carries authored UVs — decimation preserves them (the simplifier scores
// normals + UVs and surviving vertices keep their exact attributes), so the texturing is identical
// across levels. Pass a model path as argv[1]; with none it loads the fetched DamagedHelmet. Scroll to
// zoom out and watch the LOD drop; drag to orbit.
//

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <veng/assets/GltfLoader.hpp>
#include <veng/assets/Lod.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/nodes/CoverageLodNode.hpp>
#include <veng/nodes/MeshSelectorNode.hpp>
#include <veng/passes/PbrPass.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/Graph.hpp>

#include "AppLoop.hpp"

// The loader keeps its own assets::AlphaMode; bridge it to passes::AlphaMode with a static_cast, so
// lock the two enums' values together here (same as gltf_viewer).
static_assert(static_cast<int>(veng::assets::AlphaMode::Opaque) == static_cast<int>(veng::passes::AlphaMode::Opaque));
static_assert(static_cast<int>(veng::assets::AlphaMode::Mask) == static_cast<int>(veng::passes::AlphaMode::Mask));
static_assert(static_cast<int>(veng::assets::AlphaMode::Blend) == static_cast<int>(veng::passes::AlphaMode::Blend));

int main(int argc, char** argv)
{
	const std::string path = argc > 1 ? std::string(argv[1]) : std::string(VENG_MODELS_DIR) + "/DamagedHelmet.glb";

	example::AppLoop app(example::AppConfig{.title = "veng — glTF LOD viewer", .camera_distance = 4.0F, .hdr = true});
	veng::graph::Graph& graph = app.graph();

	// Decimate each primitive into a LOD chain (finest first); the coarsest uses the sloppy simplifier.
	constexpr std::array<veng::assets::LodLevel, 4> levels{
		{{.target_ratio = 1.00F},
		 {.target_ratio = 0.50F, .max_error = 0.02F},
		 {.target_ratio = 0.25F, .max_error = 0.06F},
		 {.target_ratio = 0.10F, .max_error = 0.30F, .aggressive = true}}};
	auto loaded = veng::assets::load_gltf_lods(app.context(), graph, path, levels);
	if (!loaded.has_value())
	{
		veng::Logger::instance().error("could not load '{}': {}", path, veng::assets::to_string(loaded.error()));
		veng::Logger::instance().error("usage: gltf_lod <path/to/model.glb>");
		return 1;
	}
	const veng::assets::GltfLodModel model = std::move(loaded.value());

	veng::passes::PbrPass pass(graph, app.scene_color_format(), app.depth_format(), app.screen(), app.scene_image(),
							   app.view_proj(), app.camera().eye_pos());
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

	// One coverage metric on the whole model's bounding sphere drives a level the selectors share, so
	// every primitive switches LOD coherently. Thresholds are the model's projected diameter as a
	// fraction of viewport height (descending), with a hysteresis margin.
	const auto					  sphere = graph.add_source<glm::vec4>(glm::vec4(model.center(), model.radius()));
	const veng::graph::DataHandle level	 = graph.add(std::make_unique<veng::graph::ValueData<std::uint32_t>>(0U));
	const veng::graph::NodeHandle metric = graph.add(std::make_unique<veng::nodes::CoverageLodNode>(
		app.camera().view(), app.camera().proj(), sphere, level, std::vector<float>{0.50F, 0.25F, 0.12F}, 0.1F));
	graph.set_producer(level, metric);

	for (const veng::assets::GltfLodPrimitive& prim : model.primitives)
	{
		const veng::graph::DataHandle selected =
			graph.add(std::make_unique<veng::graph::ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));
		const veng::graph::NodeHandle selector =
			graph.add(std::make_unique<veng::nodes::MeshSelectorNode>(prim.lods, level, selected));
		graph.set_producer(selected, selector);
		pass.add_object(selected, prim.model, prim.material);
	}

	app.camera().frame(model.center(), model.radius(), vk::Extent2D{1280, 720});

	veng::Logger::instance().info("glTF LOD: '{}' — {} primitives, {} materials, {} LOD levels; scroll to zoom "
								  "out and watch the LOD drop",
								  path, model.primitives.size(), model.materials.size(), levels.size());
	app.run();
	return 0;
}
