//
// PBR materials showcase — a grid of spheres sweeping metallic (left→right) against roughness
// (top→bottom), the classic way to read a metallic-roughness BRDF at a glance. It wires one
// sphere mesh, one veng::passes::PbrPass, and N×N materials/objects: each cell reuses the mesh
// with its own model matrix + material factors, all sharing 1×1 default textures (the sweep lives
// entirely in the per-material metallic/roughness factors). Drag to orbit, scroll to zoom.
//

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <veng/assets/Texture.hpp>
#include <veng/culling/Clusters.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/gpu/Vertex.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/nodes/MeshNode.hpp>
#include <veng/passes/ClusteredLights.hpp>
#include <veng/passes/PbrPass.hpp>
#include <veng/rendergraph/Graph.hpp>

#include "AppLoop.hpp"

namespace
{
struct SphereMesh
{
	std::vector<veng::gpu::PbrVertex> vertices;
	std::vector<std::uint32_t>		  indices;
};

// A UV sphere as engine PbrVertices: smooth normals (= position), longitude tangents, and
// lat/long UVs — everything pbr.vert needs, including a proper tangent frame for normal mapping.
SphereMesh make_sphere(float radius, std::uint32_t lat, std::uint32_t lon)
{
	constexpr float pi = std::numbers::pi_v<float>;
	SphereMesh		mesh;
	for (std::uint32_t y = 0; y <= lat; ++y)
	{
		const float v	  = static_cast<float>(y) / static_cast<float>(lat);
		const float theta = v * pi;
		for (std::uint32_t x = 0; x <= lon; ++x)
		{
			const float		u	= static_cast<float>(x) / static_cast<float>(lon);
			const float		phi = u * 2.0F * pi;
			const glm::vec3 n{std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)};
			const glm::vec3 tangent{-std::sin(phi), 0.0F, std::cos(phi)};
			mesh.vertices.push_back(veng::gpu::PbrVertex{
				.position = n * radius, .normal = n, .tangent = glm::vec4(tangent, 1.0F), .uv = {u, v}});
		}
	}
	for (std::uint32_t y = 0; y < lat; ++y)
	{
		for (std::uint32_t x = 0; x < lon; ++x)
		{
			const std::uint32_t i0 = (y * (lon + 1)) + x;
			const std::uint32_t i1 = i0 + 1;
			const std::uint32_t i2 = i0 + (lon + 1);
			const std::uint32_t i3 = i2 + 1;
			mesh.indices.insert(mesh.indices.end(), {i0, i2, i1, i1, i2, i3});
		}
	}
	return mesh;
}
} // namespace

int main()
{
	veng::Logger::instance().set_level(spdlog::level::info);

	example::AppLoop app(example::AppConfig{.title			= "veng — PBR materials (metallic × roughness)",
										   .camera_target	= {0.0F, 0.0F, 0.0F},
										   .camera_distance = 12.0F,
										   .camera_pitch	= 0.15F,
										   .hdr				= true});
	veng::graph::Graph& graph = app.graph();
	veng::Context&		ctx	  = app.context();

	// 1×1 default textures — the sweep is in the factors, so every material samples these.
	const std::array<std::byte, 4> white{std::byte{255}, std::byte{255}, std::byte{255}, std::byte{255}};
	const std::array<std::byte, 4> flat{std::byte{128}, std::byte{128}, std::byte{255}, std::byte{255}};
	auto						   white_srgb_r	  = veng::assets::Texture::from_pixels(ctx, white, 1, 1,
																					   veng::assets::ColorSpace::Srgb);
	auto						   white_linear_r = veng::assets::Texture::from_pixels(ctx, white, 1, 1,
																					  veng::assets::ColorSpace::Linear);
	auto						   normal_r		  = veng::assets::Texture::from_pixels(ctx, flat, 1, 1,
																				   veng::assets::ColorSpace::Linear);
	if (!white_srgb_r.has_value() || !white_linear_r.has_value() || !normal_r.has_value())
	{
		veng::Logger::instance().error("failed to create default textures");
		return 1;
	}
	// Hold the textures for the app's lifetime — the material edges reference their image handles.
	const veng::assets::Texture white_srgb	 = std::move(white_srgb_r.value());
	const veng::assets::Texture white_linear = std::move(white_linear_r.value());
	const veng::assets::Texture normal_tex	 = std::move(normal_r.value());

	const auto base_src	  = graph.add_source<veng::gpu::ImageRef>(white_srgb.ref());
	const auto mr_src	  = graph.add_source<veng::gpu::ImageRef>(white_linear.ref());
	const auto normal_src = graph.add_source<veng::gpu::ImageRef>(normal_tex.ref());

	// One sphere mesh, reused by every grid cell.
	const SphereMesh		  sphere = make_sphere(0.5F, 48, 96);
	const veng::graph::DataHandle mesh =
		graph.add(std::make_unique<veng::graph::ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));
	const veng::graph::NodeHandle mesh_node = graph.add(std::make_unique<veng::nodes::MeshNode>(
		std::span<const veng::gpu::PbrVertex>(sphere.vertices), std::span<const std::uint32_t>(sphere.indices), mesh));
	graph.set_producer(mesh, mesh_node);

	veng::passes::PbrConfig config;
	config.light_intensity = 1.5F; // dimmed so the clustered point lights read as coloured highlights
	config.cull_mode	   = veng::rhi::CullMode::NONE; // closed spheres: depth handles occlusion, no winding worry
	veng::passes::PbrPass pass(graph, app.scene_color_format(), app.depth_format(), app.screen(), app.scene_image(),
							   app.view_proj(), app.camera().eye_pos(), config);

	constexpr int	N		= 6;
	constexpr float spacing = 1.5F;
	for (int row = 0; row < N; ++row)
	{
		for (int col = 0; col < N; ++col)
		{
			const float metallic  = static_cast<float>(col) / static_cast<float>(N - 1);
			const float roughness = glm::clamp(static_cast<float>(row) / static_cast<float>(N - 1), 0.05F, 1.0F);
			const std::uint32_t material =
				pass.add_material(veng::passes::PbrMaterial{.base_color		  = base_src,
															.normal			  = normal_src,
															.metal_rough	  = mr_src,
															.emissive		  = base_src,
															.occlusion		  = mr_src,
															.base_color_factor = glm::vec4(1.0F, 0.84F, 0.40F, 1.0F),
															.metallic_factor  = metallic,
															.roughness_factor = roughness});
			const glm::vec3 pos{(static_cast<float>(col) - (N - 1) * 0.5F) * spacing,
								((N - 1) * 0.5F - static_cast<float>(row)) * spacing, 0.0F};
			const auto model = graph.add_source<glm::mat4>(glm::translate(glm::mat4(1.0F), pos));
			pass.add_object(mesh, model, material);
		}
	}

	// A row of translucent glass spheres floating in front of the opaque grid — the glTF BLEND path:
	// low-roughness dielectrics whose base-colour alpha < 1, so the metallic/roughness sweep behind
	// shows through, tinted by the glass. They are spaced apart so each overlays only the grid (not
	// another glass sphere): PbrPass blends transparent objects in submission order without a per-object
	// depth sort yet (findings.md), so non-overlapping placement keeps inter-object ordering correct.
	struct Glass
	{
		glm::vec3 pos;
		glm::vec4 color; // rgb tint, a = opacity
	};
	const std::array<Glass, 4> glass{
		Glass{.pos = {-4.0F, 0.0F, 4.0F}, .color = glm::vec4(0.95F, 0.30F, 0.30F, 0.40F)}, // red
		Glass{.pos = {-1.3F, 0.0F, 4.0F}, .color = glm::vec4(0.30F, 0.85F, 0.45F, 0.45F)}, // green
		Glass{.pos = {1.3F, 0.0F, 4.0F}, .color = glm::vec4(0.30F, 0.50F, 0.95F, 0.50F)},  // blue
		Glass{.pos = {4.0F, 0.0F, 4.0F}, .color = glm::vec4(0.95F, 0.80F, 0.30F, 0.55F)},  // amber
	};
	for (const Glass& g : glass)
	{
		const std::uint32_t material =
			pass.add_material(veng::passes::PbrMaterial{.base_color		   = base_src,
														.normal			   = normal_src,
														.metal_rough	   = mr_src,
														.emissive		   = base_src,
														.occlusion		   = mr_src,
														.base_color_factor = g.color,
														.metallic_factor   = 0.0F,
														.roughness_factor  = 0.12F,
														.alpha_mode		   = veng::passes::AlphaMode::Blend});
		const auto model =
			graph.add_source<glm::mat4>(glm::scale(glm::translate(glm::mat4(1.0F), g.pos), glm::vec3(2.2F)));
		pass.add_object(mesh, model, material);
	}

	// Clustered point lights: four coloured lamps in front of the grid, culled per froxel by the CPU
	// cull and looped per fragment by pbr.frag. They sweep coloured speculars across the sweep —
	// brightest and tightest on the metallic, low-roughness spheres (top-right).
	const std::vector<veng::culling::GpuLight> point_lights{
		{.position = glm::vec4(-3.5F, 2.5F, 4.0F, 9.0F), .color = glm::vec4(1.0F, 0.25F, 0.20F, 30.0F)}, // red
		{.position = glm::vec4(3.5F, -2.5F, 4.0F, 9.0F), .color = glm::vec4(0.25F, 0.45F, 1.0F, 30.0F)}, // blue
		{.position = glm::vec4(0.0F, 3.5F, 4.0F, 9.0F), .color = glm::vec4(0.35F, 1.0F, 0.40F, 24.0F)},	// green
		{.position = glm::vec4(0.0F, -3.5F, 4.0F, 9.0F), .color = glm::vec4(1.0F, 0.75F, 0.20F, 24.0F)}, // amber
	};
	const auto lights_src = graph.add_source<std::vector<veng::culling::GpuLight>>(point_lights);
	const veng::culling::ClusterGrid grid{.dims = {12, 8, 24}, .z_near = 0.5F, .z_far = 30.0F};
	const auto edges =
		veng::passes::wire_clustered_lights(graph, lights_src, app.camera().view(), app.camera().proj(), grid);
	pass.set_clustered_lights(app.camera().view(), edges.lights, edges.light_grid, edges.light_index, grid);

	veng::Logger::instance().info("PBR materials: {0}x{0} spheres — metallic left→right, roughness top→bottom; "
								  "{1} clustered point lights; {2} translucent (BLEND) spheres in front",
								  N, point_lights.size(), glass.size());
	app.run();
	return 0;
}
