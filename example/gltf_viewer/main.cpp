//
// glTF scene viewer — loads a .glb/.gltf with veng::assets::load_gltf and renders it through
// veng::passes::PbrPass with an orbit camera. Pass a model path as argv[1]; with no argument it
// loads the DamagedHelmet sample model fetched for the integration tests (if present). Drag to
// orbit, scroll to zoom. The camera frames itself on the model's bounds after loading.
//

#include <cstdint>
#include <string>

#include <glm/glm.hpp>

#include <veng/assets/GltfLoader.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/passes/PbrPass.hpp>
#include <veng/rendergraph/Graph.hpp>

#include "AppLoop.hpp"

// The loader keeps its own assets::AlphaMode (decoupled from the passes lib); we bridge it to
// passes::AlphaMode with a static_cast, so lock the two enums' values together here.
static_assert(static_cast<int>(veng::assets::AlphaMode::Opaque) == static_cast<int>(veng::passes::AlphaMode::Opaque));
static_assert(static_cast<int>(veng::assets::AlphaMode::Mask) == static_cast<int>(veng::passes::AlphaMode::Mask));
static_assert(static_cast<int>(veng::assets::AlphaMode::Blend) == static_cast<int>(veng::passes::AlphaMode::Blend));

int main(int argc, char** argv)
{
	veng::Logger::instance().set_level(spdlog::level::info);

	const std::string path = argc > 1 ? std::string(argv[1]) : std::string(VENG_MODELS_DIR) + "/DamagedHelmet.glb";

	example::AppLoop app(example::AppConfig{.title = "veng — glTF viewer", .camera_distance = 4.0F, .hdr = true});

	auto loaded = veng::assets::load_gltf(app.context(), app.graph(), path);
	if (!loaded.has_value())
	{
		veng::Logger::instance().error("could not load '{}': {}", path,
									   veng::assets::to_string(loaded.error()));
		veng::Logger::instance().error("usage: gltf_viewer <path/to/model.glb>");
		return 1;
	}
	const veng::assets::GltfModel model = std::move(loaded.value());

	veng::passes::PbrPass pass(app.graph(), app.scene_color_format(), app.depth_format(), app.screen(),
							   app.scene_image(), app.view_proj(), app.camera().eye_pos());
	for (const veng::assets::GltfMaterialDesc& m : model.materials)
	{
		static_cast<void>(pass.add_material(veng::passes::PbrMaterial{
			.base_color			= m.base_color,
			.normal				= m.normal,
			.metal_rough		= m.metal_rough,
			.emissive			= m.emissive,
			.occlusion			= m.occlusion,
			.base_color_factor	= m.base_color_factor,
			.metallic_factor	= m.metallic_factor,
			.roughness_factor	= m.roughness_factor,
			.normal_scale		= m.normal_scale,
			.occlusion_strength = m.occlusion_strength,
			.emissive_factor	= m.emissive_factor,
			.alpha_mode			= static_cast<veng::passes::AlphaMode>(m.alpha_mode),
			.alpha_cutoff		= m.alpha_cutoff}));
	}
	for (const veng::assets::GltfPrimitive& p : model.primitives)
	{
		pass.add_object(p.mesh, p.model, p.material);
	}

	// Frame the camera on the loaded model's bounds (the AppConfig defaults are arbitrary).
	app.camera().frame(model.center(), model.radius(), vk::Extent2D{1280, 720});

	veng::Logger::instance().info("loaded '{}': {} primitives, {} materials, {} textures", path,
								  model.primitives.size(), model.materials.size(), model.textures.size());
	app.run();
	return 0;
}
