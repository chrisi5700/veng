//
// STL viewer — loads an STL mesh (positions only: no UVs, no usable normals) via
// veng::assets::load_stl, which welds the facet soup, derives crease-smoothed normals, projects box
// UVs and a tangent frame, then renders it with a CC0 bare-steel PBR material through
// veng::passes::PbrPass. The point: an unwrapped CAD/print format like STL can still take a real,
// fully textured PBR material — the loader generates the texture coordinates the file never had.
//
// Pass an .stl path as argv[1]; with no argument it loads example/screw.stl. The steel texture set
// is fetched at configure (steel.cmake) into a gitignored assets/ dir; if it's absent (offline /
// no Pillow) the viewer falls back to a flat factor-only steel so it still runs.
//
// It builds a discrete LOD chain of the screw (load_stl_lods) and selects between the levels at
// runtime from the object's screen coverage: a CoverageLodNode emits a level index that a
// MeshSelectorNode uses to forward the matching (eagerly-uploaded) mesh to PbrPass. Scroll to zoom —
// as the screw shrinks on screen the LOD drops; the texturing is identical across levels. Drag to orbit.
//

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <veng/assets/StlLoader.hpp>
#include <veng/assets/Texture.hpp>
#include <veng/culling/Clusters.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/nodes/CoverageLodNode.hpp>
#include <veng/nodes/MeshSelectorNode.hpp>
#include <veng/passes/ClusteredLights.hpp>
#include <veng/passes/PbrPass.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/Graph.hpp>

#include "AppLoop.hpp"

namespace
{
// A 1×1 solid texture for the channels the steel set doesn't provide (occlusion/emissive) and for
// the flat fallback. Returns false into `ok` on GPU failure so callers can degrade.
veng::graph::DataHandle solid_source(veng::Context& ctx, veng::graph::Graph& graph,
									 std::vector<veng::assets::Texture>& keep, std::array<std::byte, 4> rgba,
									 veng::assets::ColorSpace space, bool& ok)
{
	auto tex = veng::assets::Texture::from_pixels(ctx, rgba, 1, 1, space);
	if (!tex.has_value())
	{
		ok = false;
		return {};
	}
	keep.push_back(std::move(tex.value()));
	return graph.add_source<veng::gpu::ImageRef>(keep.back().ref());
}
} // namespace

int main(int argc, char** argv)
{
	example::AppLoop app(
		example::AppConfig{.title = "veng — STL viewer (galvanized steel)", .camera_distance = 6.0F, .hdr = true});
	veng::graph::Graph& graph = app.graph();
	veng::Context&		ctx	  = app.context();

	const std::string mesh_path = argc > 1 ? std::string(argv[1]) : std::string(VENG_STL_MESH);

	// Build a discrete LOD chain of the screw — each level decimated harder (the coarsest via the
	// sloppy simplifier), all uploaded eagerly so switching is a free edge swap. A widening error
	// budget protects the high-curvature threads on the finer levels. UVs are synthesised *after*
	// decimation, so the steel texturing is identical across levels; only the geometry slims.
	constexpr std::array<veng::assets::StlLodLevel, 4> levels{
		{{.target_ratio = 1.00F},
		 {.target_ratio = 0.50F, .max_error = 0.02F},
		 {.target_ratio = 0.25F, .max_error = 0.06F},
		 {.target_ratio = 0.10F, .max_error = 0.30F, .aggressive = true}}};
	auto loaded = veng::assets::load_stl_lods(graph, mesh_path, levels);
	if (!loaded.has_value())
	{
		veng::Logger::instance().error("could not load '{}': {}", mesh_path, veng::assets::to_string(loaded.error()));
		veng::Logger::instance().error("usage: stl_viewer <path/to/mesh.stl>");
		return 1;
	}
	const veng::assets::StlLodSet lods = std::move(loaded.value());

	// Normalise to a ~2-unit radius centred at the origin so the studio lights + LOD thresholds sit in
	// a known regime. p' = scale · (p − centre).
	constexpr float target_radius = 2.0F;
	const float		scale		  = lods.radius() > 1e-6F ? target_radius / lods.radius() : 1.0F;
	const glm::vec3 center		  = lods.center();
	const auto		model_src	  = graph.add_source<glm::mat4>(glm::scale(glm::mat4(1.0F), glm::vec3(scale)) *
																glm::translate(glm::mat4(1.0F), -center));

	// --- Steel material -----------------------------------------------------------------------
	// Channels we always need a 1×1 for: occlusion (white = none) and emissive (white × factor 0).
	std::vector<veng::assets::Texture> textures;
	textures.reserve(6);
	constexpr std::array<std::byte, 4> white{std::byte{255}, std::byte{255}, std::byte{255}, std::byte{255}};
	bool							   ok = true;
	const auto white_linear = solid_source(ctx, graph, textures, white, veng::assets::ColorSpace::Linear, ok);

	const std::filesystem::path dir		  = VENG_STL_ASSETS_DIR;
	const std::string			base_path = (dir / "steel_basecolor.png").string();
	const std::string			norm_path = (dir / "steel_normal.png").string();
	const std::string			mr_path	  = (dir / "steel_mr.png").string();
	const bool textured = ok && std::filesystem::exists(base_path) && std::filesystem::exists(norm_path) &&
						  std::filesystem::exists(mr_path);

	veng::passes::PbrMaterial material{};
	material.occlusion = white_linear;
	material.emissive  = white_linear; // emissive_factor stays 0 → no emission
	// A faint cool tint pushes the bare steel toward the bluish cast of galvanized zinc.
	material.base_color_factor = glm::vec4(0.92F, 0.95F, 1.0F, 1.0F);

	bool textured_applied = false;
	if (textured)
	{
		auto base = veng::assets::Texture::from_file(ctx, base_path, veng::assets::ColorSpace::Srgb);
		auto norm = veng::assets::Texture::from_file(ctx, norm_path, veng::assets::ColorSpace::Linear);
		auto mr	  = veng::assets::Texture::from_file(ctx, mr_path, veng::assets::ColorSpace::Linear);
		if (base.has_value() && norm.has_value() && mr.has_value())
		{
			textures.push_back(std::move(base.value()));
			material.base_color = graph.add_source<veng::gpu::ImageRef>(textures.back().ref());
			textures.push_back(std::move(norm.value()));
			material.normal = graph.add_source<veng::gpu::ImageRef>(textures.back().ref());
			textures.push_back(std::move(mr.value()));
			material.metal_rough	  = graph.add_source<veng::gpu::ImageRef>(textures.back().ref());
			material.metallic_factor  = 1.0F; // metalness comes from the texture's B channel
			material.roughness_factor = 1.0F; // roughness comes from the texture's G channel
			textured_applied		  = true;
			veng::Logger::instance().info("loaded steel PBR texture set from {}", dir.string());
		}
		else
		{
			veng::Logger::instance().warn("steel textures failed to upload — using flat steel");
		}
	}

	if (!textured_applied)
	{
		// Flat fallback: 1×1 defaults + factors. Reads as plain brushed steel, no surface detail.
		veng::Logger::instance().warn("steel texture set not found at {} — using flat factor steel "
									  "(run cmake configure online with Pillow to fetch it)",
									  dir.string());
		constexpr std::array<std::byte, 4> flat{std::byte{128}, std::byte{128}, std::byte{255}, std::byte{255}};
		material.base_color		   = solid_source(ctx, graph, textures, white, veng::assets::ColorSpace::Srgb, ok);
		material.normal			   = solid_source(ctx, graph, textures, flat, veng::assets::ColorSpace::Linear, ok);
		material.metal_rough	   = white_linear;
		material.base_color_factor = glm::vec4(0.62F, 0.64F, 0.68F, 1.0F);
		material.metallic_factor   = 1.0F;
		material.roughness_factor  = 0.4F;
	}

	if (!ok)
	{
		veng::Logger::instance().error("failed to create material textures");
		return 1;
	}

	// --- Pass + object ------------------------------------------------------------------------
	veng::passes::PbrConfig config;
	// load_stl's repair pass unifies winding and orients the shell outward, so backface culling is
	// safe (and the normals are trustworthy) — no need to fall back to eNone for unknown STL winding.
	config.cull_mode	   = veng::rhi::CullMode::BACK;
	config.light_intensity = 2.0F;
	config.ambient		   = glm::vec3(0.06F, 0.06F, 0.07F);
	// The screw's hard machined edges alias badly without MSAA; 4x is clamped to the device max on
	// first record, and the pass resolves to a single-sample image so the rest of the graph is unchanged.
	config.samples = veng::rhi::SampleCount::X4;
	veng::passes::PbrPass pass(graph, app.scene_color_format(), app.depth_format(), app.screen(), app.scene_image(),
							   app.view_proj(), app.camera().eye_pos(), config);
	const std::uint32_t	  mat_index = pass.add_material(material);

	// --- Dynamic LOD wiring -------------------------------------------------------------------
	// The object's world-space bounding sphere (centre origin, radius target_radius) feeds the
	// coverage metric → a uint level → the mesh selector, which forwards the chosen LOD. Thresholds
	// are the screw's projected diameter as a fraction of viewport height (descending); the hysteresis
	// margin keeps it from flickering at a boundary. All of this is pipeline-agnostic — PbrPass just
	// consumes the selector's MeshRef edge like any other mesh.
	const auto					  sphere = graph.add_source<glm::vec4>(glm::vec4(0.0F, 0.0F, 0.0F, target_radius));
	const veng::graph::DataHandle level	 = graph.add(std::make_unique<veng::graph::ValueData<std::uint32_t>>(0U));
	const veng::graph::NodeHandle metric = graph.add(std::make_unique<veng::nodes::CoverageLodNode>(
		app.camera().view(), app.camera().proj(), sphere, level, std::vector<float>{0.55F, 0.30F, 0.15F}, 0.1F));
	graph.set_producer(level, metric);

	const veng::graph::DataHandle selected =
		graph.add(std::make_unique<veng::graph::ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));
	const veng::graph::NodeHandle selector =
		graph.add(std::make_unique<veng::nodes::MeshSelectorNode>(lods.meshes, level, selected));
	graph.set_producer(selected, selector);

	pass.add_object(selected, model_src, mat_index);

	// --- Studio point lights ------------------------------------------------------------------
	// Key / fill / rim plus a warm kicker — metals read mostly through their speculars, and with no
	// IBL yet these moving highlights are what give the steel life as you orbit.
	const std::vector<veng::culling::GpuLight> lights{
		{.position = glm::vec4(4.0F, 5.0F, 4.0F, 14.0F), .color = glm::vec4(1.0F, 1.0F, 1.0F, 45.0F)},	  // key
		{.position = glm::vec4(-5.0F, 1.5F, 3.0F, 14.0F), .color = glm::vec4(0.7F, 0.8F, 1.0F, 22.0F)},	  // cool fill
		{.position = glm::vec4(0.0F, 3.0F, -5.0F, 14.0F), .color = glm::vec4(1.0F, 1.0F, 1.0F, 35.0F)},	  // rim
		{.position = glm::vec4(2.0F, -4.0F, 2.0F, 14.0F), .color = glm::vec4(1.0F, 0.85F, 0.6F, 18.0F)}}; // warm kicker
	const auto						 lights_src = graph.add_source<std::vector<veng::culling::GpuLight>>(lights);
	const veng::culling::ClusterGrid grid{.dims = {12, 8, 24}, .z_near = 0.5F, .z_far = 30.0F};
	const auto						 edges =
		veng::passes::wire_clustered_lights(graph, lights_src, app.camera().view(), app.camera().proj(), grid);
	pass.set_clustered_lights(app.camera().view(), edges.lights, edges.light_grid, edges.light_index, grid);

	app.camera().frame(glm::vec3(0.0F), target_radius, vk::Extent2D{1280, 720});

	veng::Logger::instance().info("STL viewer: '{}' — {} LOD levels ({}), coverage-driven; scroll to zoom "
								  "out and watch the LOD drop; {} point lights",
								  mesh_path, lods.meshes.size(), textured ? "textured steel" : "flat steel",
								  lights.size());
	app.run();
	return 0;
}
