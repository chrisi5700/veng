/**
 * @file
 * @author chris
 * @brief Implementation of @ref veng::passes::PbrPass — the @ref veng::passes::PbrRenderNode private class,
 * Cook-Torrance BRDF recording, descriptor management, and the @ref veng::passes::PbrPass public facade.
 * @ingroup render_passes
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>
#include <veng/culling/Clusters.hpp>
#include <veng/descriptors/DescriptorAllocator.hpp>
#include <veng/gpu/BufferRef.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/gpu/VersionedOutput.hpp>
#include <veng/gpu/Vertex.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/passes/PbrPass.hpp>
#include <veng/pipelines/GraphicsPipeline.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/resources/Image.hpp>
#include <veng/resources/ResourcePool.hpp>
#include <veng/resources/SamplerConfig.hpp>
#include <veng/shader/Shader.hpp>

namespace veng::passes
{
namespace
{
// The shared per-frame camera/light block — must match pbr.vert/.frag's `Frame` (set 0, binding 0).
struct PbrFrame
{
	glm::mat4  view_proj;	   // 0
	glm::mat4  view;		   // 64   world->view, for the fragment's clustered depth slice
	glm::vec4  camera_pos;	   // 128
	glm::vec4  light_dir;	   // 144
	glm::vec4  light_color;	   // 160
	glm::vec4  ambient;		   // 176
	glm::uvec4 cluster_dims;   // 192  xyz = froxel grid dims, w = cluster count (0 => no clustered lights)
	glm::vec4  cluster_params; // 208  x z_near, y z_far, zw = screen extent (px)
};
static_assert(sizeof(PbrFrame) == 224, "PbrFrame must match the shader's Frame uniform layout");

// The per-object push block — must match pbr.vert/.frag's `Push`. Vertex reads `model`, fragment
// reads the factors; a single pushConstants(Vertex|Fragment, 0, 112) covers both stages' ranges.
struct PbrPush
{
	glm::mat4 model;	  // 0
	glm::vec4 base_color; // 64
	glm::vec4 mr_factors; // 80  x=metallic, y=roughness, z=normal_scale, w=occlusion_strength
	glm::vec4 emissive;	  // 96
};
static_assert(sizeof(PbrPush) == 112, "PbrPush must match the shader's 112-byte push block");

constexpr std::uint32_t FRAME_BINDING		= 0;
constexpr std::uint32_t TEXTURE_BINDING		= 1; // base_color, normal, metal_rough, emissive, occlusion -> 1..5
constexpr std::uint32_t SAMPLER_BINDING		= 6;
constexpr std::uint32_t LIGHTS_BINDING		= 7; // StructuredBuffer<Light> (clustered point lights)
constexpr std::uint32_t LIGHT_GRID_BINDING	= 8; // StructuredBuffer<uint2> per-cluster (offset, count)
constexpr std::uint32_t LIGHT_INDEX_BINDING = 9; // StructuredBuffer<uint> flat light indices
constexpr std::size_t	TEXTURE_COUNT		= 5;
} // namespace

class PbrRenderNode final : public gpu::GpuNode
{
	 public:
	PbrRenderNode(vk::Format color_format, vk::Format depth_format, graph::DataHandle screen, graph::DataHandle output,
				  graph::DataHandle view_proj, graph::DataHandle eye, PbrConfig config) noexcept
		: m_color_format(color_format)
		, m_depth_format(depth_format)
		, m_output(output)
		, m_view_proj(view_proj)
		, m_eye(eye)
		, m_config(config)
		, m_inputs{screen, view_proj, eye}
	{
	}

	std::uint32_t add_material(const PbrMaterial& material)
	{
		const auto index = static_cast<std::uint32_t>(m_materials.size());
		// emissive.w carries the alpha-MASK cutoff (shader discards below it); 0 for OPAQUE/BLEND so
		// the shader's discard never triggers. BLEND additionally routes the object to the blended batch.
		const float cutoff = material.alpha_mode == AlphaMode::Mask ? material.alpha_cutoff : 0.0F;
		m_materials.push_back(Material{.textures	= {material.base_color, material.normal, material.metal_rough,
													   material.emissive, material.occlusion},
									   .base_color	= material.base_color_factor,
									   .mr_factors	= glm::vec4(material.metallic_factor, material.roughness_factor,
																material.normal_scale, material.occlusion_strength),
									   .emissive	= glm::vec4(material.emissive_factor, cutoff),
									   .transparent = material.alpha_mode == AlphaMode::Blend});
		for (const graph::DataHandle texture : m_materials.back().textures)
		{
			m_inputs.push_back(texture); // textures are reactive sampled inputs -> demand them
		}
		mark_dirty();
		return index;
	}

	void add_object(graph::DataHandle mesh, graph::DataHandle model, std::uint32_t material)
	{
		m_objects.push_back(Object{.mesh = mesh, .model = model, .material = material});
		m_inputs.push_back(mesh);
		m_inputs.push_back(model);
		mark_dirty();
	}

	// Bind the clustered point-light edges: the `view` matrix (world->view, for the depth slice), the
	// world-space lights SSBO, and the cull's per-cluster grid + flat index SSBOs (all gpu::BufferRef
	// edges, typically from StorageBufferNodes downstream of LightCullCpu). `grid` must match the one
	// the cull used. Without this call the pass shades the directional light + ambient only.
	void set_clustered_lights(graph::DataHandle view, graph::DataHandle lights, graph::DataHandle light_grid,
							  graph::DataHandle light_index, const culling::ClusterGrid& grid)
	{
		m_cluster_view = view;
		m_lights	   = lights;
		m_light_grid   = light_grid;
		m_light_index  = light_index;
		m_cluster_grid = grid;
		m_clustered	   = true;
		m_inputs.push_back(view);
		m_inputs.push_back(lights);
		m_inputs.push_back(light_grid);
		m_inputs.push_back(light_index);
		mark_dirty();
	}

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return m_inputs; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	 protected:
	std::expected<bool, graph::ExecError> record(gpu::GpuExecContext& ctx) override
	{
		if (auto built = ensure_pipelines(ctx); !built.has_value())
		{
			return std::unexpected(built.error());
		}

		const auto* size = dynamic_cast<graph::ValueData<vk::Extent2D>*>(ctx.data(m_inputs[0]));
		if (size == nullptr)
		{
			return std::unexpected(graph::ExecError::MISSING_INPUT);
		}
		const vk::Extent2D extent = size->value();
		if (extent.width == 0 || extent.height == 0)
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}

		const auto* view_proj_d = dynamic_cast<graph::ValueData<glm::mat4>*>(ctx.data(m_view_proj));
		const auto* eye_d		= dynamic_cast<graph::ValueData<glm::vec4>*>(ctx.data(m_eye));
		if (view_proj_d == nullptr || eye_d == nullptr)
		{
			return std::unexpected(graph::ExecError::MISSING_INPUT);
		}

		// Declare pool-backed targets + the per-frame uniform once; acquire this frame's copies.
		if (!m_declared)
		{
			m_color_id = ctx.pool().declare_image(m_color_format, vk::ImageUsageFlagBits::eColorAttachment |
																	  vk::ImageUsageFlagBits::eTransferSrc |
																	  vk::ImageUsageFlagBits::eSampled);
			m_depth_id = ctx.pool().declare_image(m_depth_format, vk::ImageUsageFlagBits::eDepthStencilAttachment,
												  vk::ImageAspectFlagBits::eDepth);
			m_frame_id = ctx.pool().declare_buffer(vk::BufferUsageFlagBits::eUniformBuffer);
			m_default_ssbo_id = ctx.pool().declare_buffer(vk::BufferUsageFlagBits::eStorageBuffer);
			m_declared		  = true;
		}
		auto color = ctx.pool().acquire_image(m_color_id, extent);
		auto depth = ctx.pool().acquire_image(m_depth_id, extent);
		auto frame = ctx.pool().acquire_buffer(m_frame_id, sizeof(PbrFrame));
		if (!color.has_value() || !depth.has_value() || !frame.has_value())
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		Image* const color_image = color.value();
		Image* const depth_image = depth.value();

		// Clustered point-light state: the view matrix (for the fragment's depth slice) + the grid
		// dimensions/extent. Off by default — cluster_dims.w (the cluster count) stays 0, so the
		// shader's point-light loop is skipped and only the directional light + ambient apply.
		glm::mat4  view			  = glm::mat4(1.0F);
		glm::uvec4 cluster_dims	  = glm::uvec4(0);
		glm::vec4  cluster_params = glm::vec4(0.0F);
		if (m_clustered)
		{
			const auto* view_d = dynamic_cast<graph::ValueData<glm::mat4>*>(ctx.data(m_cluster_view));
			if (view_d == nullptr)
			{
				return std::unexpected(graph::ExecError::MISSING_INPUT);
			}
			view		   = view_d->value();
			cluster_dims   = glm::uvec4(m_cluster_grid.dims, m_cluster_grid.count());
			cluster_params = glm::vec4(m_cluster_grid.z_near, m_cluster_grid.z_far, static_cast<float>(extent.width),
									   static_cast<float>(extent.height));
		}

		const PbrFrame frame_data{.view_proj	  = view_proj_d->value(),
								  .view			  = view,
								  .camera_pos	  = glm::vec4(glm::vec3(eye_d->value()), 1.0F),
								  .light_dir	  = glm::vec4(glm::normalize(m_config.light_direction), 0.0F),
								  .light_color	  = glm::vec4(m_config.light_color * m_config.light_intensity, 0.0F),
								  .ambient		  = glm::vec4(m_config.ambient, 0.0F),
								  .cluster_dims	  = cluster_dims,
								  .cluster_params = cluster_params};
		std::memcpy(frame.value()->mapped(), &frame_data, sizeof(PbrFrame));

		// The light SSBOs bound this frame: the caller's clustered buffers, or a small default buffer
		// (cluster count 0 means the shader never reads them, but the descriptor set must be complete).
		LightBuffers light_buffers;
		if (m_clustered)
		{
			if (auto resolved = resolve_light_buffers(ctx); resolved.has_value())
			{
				light_buffers = resolved.value();
			}
			else
			{
				return std::unexpected(resolved.error());
			}
		}
		else
		{
			auto def = ctx.pool().acquire_buffer(m_default_ssbo_id, 16);
			if (!def.has_value())
			{
				return std::unexpected(graph::ExecError::NODE_FAILED);
			}
			light_buffers = LightBuffers{.lights = def.value()->buffer(),
										 .grid	 = def.value()->buffer(),
										 .index	 = def.value()->buffer(),
										 .sizes	 = {16, 16, 16}};
		}

		if (!ensure_sampler(ctx))
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		if (auto sets = write_descriptor_sets(ctx, frame.value()->buffer(), light_buffers); !sets.has_value())
		{
			return std::unexpected(sets.error());
		}

		const vk::CommandBuffer cmd = ctx.command_buffer();
		ctx.pool().transition_image(m_color_id, cmd, vk::ImageLayout::eColorAttachmentOptimal,
									vk::PipelineStageFlagBits2::eColorAttachmentOutput,
									vk::AccessFlagBits2::eColorAttachmentWrite);
		ctx.pool().transition_image(m_depth_id, cmd, vk::ImageLayout::eDepthStencilAttachmentOptimal,
									vk::PipelineStageFlagBits2::eEarlyFragmentTests,
									vk::AccessFlagBits2::eDepthStencilAttachmentWrite);

		const auto color_clear		= vk::ClearValue().setColor(vk::ClearColorValue(m_config.clear_color));
		const auto color_attachment = vk::RenderingAttachmentInfo()
										  .setImageView(color_image->view())
										  .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
										  .setLoadOp(vk::AttachmentLoadOp::eClear)
										  .setStoreOp(vk::AttachmentStoreOp::eStore)
										  .setClearValue(color_clear);
		const auto depth_clear		= vk::ClearValue().setDepthStencil(vk::ClearDepthStencilValue(1.0F, 0));
		const auto depth_attachment = vk::RenderingAttachmentInfo()
										  .setImageView(depth_image->view())
										  .setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
										  .setLoadOp(vk::AttachmentLoadOp::eClear)
										  .setStoreOp(vk::AttachmentStoreOp::eDontCare)
										  .setClearValue(depth_clear);
		const auto rendering		= vk::RenderingInfo()
										  .setRenderArea(vk::Rect2D({0, 0}, extent))
										  .setLayerCount(1)
										  .setColorAttachments(color_attachment)
										  .setPDepthAttachment(&depth_attachment);
		cmd.beginRendering(rendering);
		cmd.setViewport(0, vk::Viewport(0.0F, 0.0F, static_cast<float>(extent.width), static_cast<float>(extent.height),
										0.0F, 1.0F));
		cmd.setScissor(0, vk::Rect2D({0, 0}, extent));
		const std::size_t					  slot		  = ctx.frame_slot();
		std::expected<void, graph::ExecError> draw_result = {};

		// Record one object against the already-bound `pipe`: bind its material's descriptor set, push
		// its per-object block, bind the mesh, draw. An un-uploaded mesh is skipped (success); a typed
		// failure aborts the frame. All three pipelines share a layout (same shaders), so the bound
		// pipeline's layout drives descriptors + push constants.
		const auto draw_object = [&](const Object&			 obj,
									 const GraphicsPipeline& pipe) -> std::expected<void, graph::ExecError>
		{
			const auto* model_d = dynamic_cast<graph::ValueData<glm::mat4>*>(ctx.data(obj.model));
			const auto* mesh_d	= dynamic_cast<graph::ValueData<gpu::MeshRef>*>(ctx.data(obj.mesh));
			if (model_d == nullptr || mesh_d == nullptr)
			{
				return std::unexpected(graph::ExecError::MISSING_INPUT);
			}
			const gpu::MeshRef mesh = mesh_d->value();
			if (!mesh.vertex_buffer)
			{
				return {}; // mesh not uploaded yet — skip until its MeshNode has produced
			}
			if (m_expected_vertex_stride.has_value() && mesh.vertex_stride != 0 &&
				mesh.vertex_stride != *m_expected_vertex_stride)
			{
				Logger::instance().error("PbrPass vertex stride mismatch: mesh has {} bytes/vertex, shader expects {}",
										 mesh.vertex_stride, *m_expected_vertex_stride);
				return std::unexpected(graph::ExecError::NODE_FAILED);
			}
			const Material& material = m_materials[obj.material];
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipe.layout(), 0, m_sets[obj.material][slot], {});
			const PbrPush push{.model	   = model_d->value(),
							   .base_color = material.base_color,
							   .mr_factors = material.mr_factors,
							   .emissive   = material.emissive};
			cmd.pushConstants<std::byte>(
				pipe.layout(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
				vk::ArrayProxy<const std::byte>(static_cast<std::uint32_t>(sizeof(PbrPush)),
												static_cast<const std::byte*>(static_cast<const void*>(&push))));
			const vk::DeviceSize offset = 0;
			cmd.bindVertexBuffers(0, mesh.vertex_buffer, offset);
			if (mesh.index_buffer)
			{
				cmd.bindIndexBuffer(mesh.index_buffer, 0, mesh.index_type);
				cmd.drawIndexed(mesh.index_count, 1, 0, 0, 0);
			}
			else
			{
				cmd.draw(mesh.vertex_count, 1, 0, 0);
			}
			return {};
		};

		// Draw the objects whose material's `transparent` flag matches, with `pipe` bound once.
		const auto draw_batch = [&](const GraphicsPipeline& pipe, bool transparent)
		{
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipe.pipeline());
			for (const Object& obj : m_objects)
			{
				if (m_materials[obj.material].transparent != transparent)
				{
					continue;
				}
				draw_result = draw_object(obj, pipe);
				if (!draw_result.has_value())
				{
					return;
				}
			}
		};

		// Opaque + MASK first (writes depth), then BLEND far faces, then BLEND near faces — the
		// transparent batches depth-test against the opaque depth but don't write it, alpha-blending
		// each convex object's far surface then its near surface (PhongPass's transparency trick).
		draw_batch(*m_pipeline, false);
		if (draw_result.has_value())
		{
			draw_batch(*m_blend_far, true);
		}
		if (draw_result.has_value())
		{
			draw_batch(*m_blend_near, true);
		}
		cmd.endRendering();
		if (!draw_result.has_value())
		{
			return std::unexpected(draw_result.error());
		}

		m_versioned.publish(ctx, m_output,
							gpu::ImageRef{.image   = color_image->image(),
										  .view	   = color_image->view(),
										  .extent  = extent,
										  .format  = m_color_format,
										  .pool_id = m_color_id});
		return true;
	}

	 private:
	struct Material
	{
		std::array<graph::DataHandle, TEXTURE_COUNT> textures; // base_color, normal, metal_rough, emissive, occlusion
		glm::vec4									 base_color;
		glm::vec4									 mr_factors;
		glm::vec4									 emissive;	  // rgb = emissive; w = MASK cutoff (0 otherwise)
		bool										 transparent; // glTF BLEND -> the alpha-blended batch
	};
	struct Object
	{
		graph::DataHandle mesh;
		graph::DataHandle model;
		std::uint32_t	  material;
	};

	// The buffers bound to the three light-SSBO bindings (7,8,9) this frame, with their byte ranges.
	struct LightBuffers
	{
		vk::Buffer					  lights;
		vk::Buffer					  grid;
		vk::Buffer					  index;
		std::array<vk::DeviceSize, 3> sizes; // lights, grid, index
	};

	// Resolve the clustered-light BufferRef edges to their buffers + ranges (set_clustered_lights path).
	std::expected<LightBuffers, graph::ExecError> resolve_light_buffers(gpu::GpuExecContext& ctx)
	{
		const std::array<graph::DataHandle, 3> handles{m_lights, m_light_grid, m_light_index};
		LightBuffers						   out{};
		std::array<vk::Buffer*, 3>			   targets{&out.lights, &out.grid, &out.index};
		for (std::size_t i = 0; i < handles.size(); ++i)
		{
			const auto* ref = dynamic_cast<graph::ValueData<gpu::BufferRef>*>(ctx.data(handles[i]));
			if (ref == nullptr || !ref->value().buffer)
			{
				return std::unexpected(graph::ExecError::MISSING_INPUT);
			}
			// Retain the pooled copy while this frame is in flight — the StorageBufferNode producing it
			// is cached when the lights are static, so without this the pool could recycle the buffer
			// out from under our descriptor set (VUID-vkDestroyBuffer-buffer-00922).
			ctx.pool().consume(ref->value());
			*targets[i]	 = ref->value().buffer;
			out.sizes[i] = ref->value().size;
		}
		return out;
	}

	std::expected<void, graph::ExecError> ensure_pipelines(gpu::GpuExecContext& ctx)
	{
		if (m_pipeline.has_value())
		{
			return {};
		}
		auto vert = Shader::create_shader(ctx.device(), "passes/pbr.vert");
		auto frag = Shader::create_shader(ctx.device(), "passes/pbr.frag");
		if (!vert.has_value() || !frag.has_value())
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		const std::array color_formats{m_color_format};
		struct PipeState
		{
			vk::CullModeFlags cull;
			bool			  depth_write;
			bool			  blend;
		};
		const auto make = [&](const PipeState& state) -> std::expected<GraphicsPipeline, PipelineError>
		{
			GraphicsPipelineBuilder builder(vert.value(), frag.value());
			// eCounterClockwise — NOTE this is the opposite of PhongPass. glTF defines front faces as
			// CCW-wound in model space (spec 3.7.2.1), whereas the engine's own hand-authored meshes
			// (example/common/Geometry.cpp) wind their *outward* faces CW. The Y-flipped projection
			// (proj[1][1] *= -1) negates winding in framebuffer space for both, so the model-space
			// difference carries through: glTF's CCW-front geometry reads as front under
			// eCounterClockwise here, and eBack then culls the true far faces. Using eClockwise (as
			// PhongPass does for the engine's CW-front geometry) would cull a glTF model's outer shell
			// and leave only its backfaces visible.
			builder.color_formats(color_formats)
				.depth_format(m_depth_format)
				.depth_write(state.depth_write)
				.rasterization(vk::PolygonMode::eFill, state.cull, vk::FrontFace::eCounterClockwise)
				.blend(state.blend);
			return builder.build(ctx.context());
		};
		// Opaque + MASK ride the depth-writing pipeline (cull per config; the test passes eNone). BLEND
		// draws far faces then near faces, depth-tested but not writing, alpha-blended — the same
		// convex-transparency trick as PhongPass. eBack keeps near faces here (it shows the outer shell
		// in the opaque pass), so eFront selects the far faces drawn first.
		auto opaque = make(PipeState{.cull = m_config.cull_mode, .depth_write = true, .blend = false});
		auto far	= make(PipeState{.cull = vk::CullModeFlagBits::eFront, .depth_write = false, .blend = true});
		auto near	= make(PipeState{.cull = vk::CullModeFlagBits::eBack, .depth_write = false, .blend = true});
		if (!opaque.has_value() || !far.has_value() || !near.has_value())
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		m_pipeline	 = std::move(opaque.value());
		m_blend_far	 = std::move(far.value());
		m_blend_near = std::move(near.value());

		if (const auto* details = std::get_if<VertexDetails>(&vert.value().get_details()))
		{
			for (const VertexBinding& binding : details->bindings)
			{
				if (binding.binding == 0)
				{
					m_expected_vertex_stride = binding.stride;
					break;
				}
			}
		}
		return {};
	}

	bool ensure_sampler(gpu::GpuExecContext& ctx)
	{
		if (m_sampler)
		{
			return true;
		}
		m_device		   = ctx.device();
		const auto sampler = ctx.device().createSampler(gpu::SamplerConfig::texture().to_create_info());
		if (sampler.result != vk::Result::eSuccess)
		{
			return false;
		}
		m_sampler = sampler.value;
		return true;
	}

	// Allocate (lazily) and rewrite one descriptor set per (material, frame slot): the per-frame
	// uniform at binding 0, the material's five textures at 1..5, the node sampler at 6, and the three
	// clustered-light SSBOs at 7..9. Per-slot sets are safe to rewrite — the slot's set was last used
	// by a now-retired frame.
	std::expected<void, graph::ExecError> write_descriptor_sets(gpu::GpuExecContext& ctx, vk::Buffer frame_buffer,
																const LightBuffers& light_buffers)
	{
		if (!m_descriptors.has_value())
		{
			m_descriptors.emplace(ctx.device());
		}
		const std::size_t slot = ctx.frame_slot();
		m_sets.resize(m_materials.size());

		for (std::size_t m = 0; m < m_materials.size(); ++m)
		{
			if (slot >= m_sets[m].size())
			{
				m_sets[m].resize(slot + 1, vk::DescriptorSet{});
			}
			if (!m_sets[m][slot])
			{
				auto set = m_descriptors->allocate(m_pipeline->descriptor_set_layout());
				if (!set.has_value())
				{
					return std::unexpected(graph::ExecError::NODE_FAILED);
				}
				m_sets[m][slot] = set.value();
			}
			const vk::DescriptorSet set = m_sets[m][slot];

			const auto buffer_info =
				vk::DescriptorBufferInfo().setBuffer(frame_buffer).setOffset(0).setRange(sizeof(PbrFrame));
			std::array<vk::DescriptorImageInfo, TEXTURE_COUNT> image_infos{};
			std::vector<vk::WriteDescriptorSet>				   writes;
			writes.reserve(TEXTURE_COUNT + 2);
			writes.push_back(vk::WriteDescriptorSet()
								 .setDstSet(set)
								 .setDstBinding(FRAME_BINDING)
								 .setDescriptorType(vk::DescriptorType::eUniformBuffer)
								 .setBufferInfo(buffer_info));

			for (std::size_t t = 0; t < TEXTURE_COUNT; ++t)
			{
				const auto* slot_data =
					dynamic_cast<graph::ValueData<gpu::ImageRef>*>(ctx.data(m_materials[m].textures[t]));
				if (slot_data == nullptr || !slot_data->value().view)
				{
					return std::unexpected(graph::ExecError::MISSING_INPUT); // every material slot needs a texture
				}
				const gpu::ImageRef ref = slot_data->value();
				ctx.pool().consume(ref); // no-op for a non-pooled (loaded) texture
				image_infos[t] = vk::DescriptorImageInfo().setImageView(ref.view).setImageLayout(
					vk::ImageLayout::eShaderReadOnlyOptimal);
				writes.push_back(vk::WriteDescriptorSet()
									 .setDstSet(set)
									 .setDstBinding(TEXTURE_BINDING + static_cast<std::uint32_t>(t))
									 .setDescriptorType(vk::DescriptorType::eSampledImage)
									 .setImageInfo(image_infos[t]));
			}

			const auto sampler_info = vk::DescriptorImageInfo().setSampler(m_sampler);
			writes.push_back(vk::WriteDescriptorSet()
								 .setDstSet(set)
								 .setDstBinding(SAMPLER_BINDING)
								 .setDescriptorType(vk::DescriptorType::eSampler)
								 .setImageInfo(sampler_info));

			// Clustered-light SSBOs (7,8,9). Real buffers when clustered, else the shared default
			// buffer (never read by the shader since cluster_dims.w is 0, but the set must be complete).
			const std::array<vk::DescriptorBufferInfo, 3> ssbo_infos{
				vk::DescriptorBufferInfo()
					.setBuffer(light_buffers.lights)
					.setOffset(0)
					.setRange(light_buffers.sizes[0]),
				vk::DescriptorBufferInfo().setBuffer(light_buffers.grid).setOffset(0).setRange(light_buffers.sizes[1]),
				vk::DescriptorBufferInfo()
					.setBuffer(light_buffers.index)
					.setOffset(0)
					.setRange(light_buffers.sizes[2])};
			const std::array<std::uint32_t, 3> ssbo_bindings{LIGHTS_BINDING, LIGHT_GRID_BINDING, LIGHT_INDEX_BINDING};
			for (std::size_t i = 0; i < ssbo_infos.size(); ++i)
			{
				writes.push_back(vk::WriteDescriptorSet()
									 .setDstSet(set)
									 .setDstBinding(ssbo_bindings[i])
									 .setDescriptorType(vk::DescriptorType::eStorageBuffer)
									 .setBufferInfo(ssbo_infos[i]));
			}

			ctx.device().updateDescriptorSets(writes, {});
		}
		return {};
	}

	vk::Format					   m_color_format;
	vk::Format					   m_depth_format;
	graph::DataHandle			   m_output;
	graph::DataHandle			   m_view_proj;
	graph::DataHandle			   m_eye;
	PbrConfig					   m_config;
	std::vector<graph::DataHandle> m_inputs; // [screen, view_proj, eye, then textures + mesh/model per object]
	std::vector<Material>		   m_materials;
	std::vector<Object>			   m_objects;

	// Clustered point lights (optional; see set_clustered_lights). When m_clustered is false the
	// shader's cluster count is 0 and the point-light loop is skipped.
	bool				 m_clustered = false;
	graph::DataHandle	 m_cluster_view;
	graph::DataHandle	 m_lights;
	graph::DataHandle	 m_light_grid;
	graph::DataHandle	 m_light_index;
	culling::ClusterGrid m_cluster_grid;

	std::optional<GraphicsPipeline>				m_pipeline;	  // opaque + MASK (depth write)
	std::optional<GraphicsPipeline>				m_blend_far;  // BLEND far faces  (eFront cull, no depth write)
	std::optional<GraphicsPipeline>				m_blend_near; // BLEND near faces (eBack cull,  no depth write)
	std::optional<std::uint32_t>				m_expected_vertex_stride;
	std::optional<DescriptorAllocator>			m_descriptors;
	std::vector<std::vector<vk::DescriptorSet>> m_sets; // [material][frame slot]
	vk::Sampler									m_sampler;
	vk::Device									m_device;

	bool				 m_declared		   = false;
	ImageId				 m_color_id		   = 0;
	ImageId				 m_depth_id		   = 0;
	BufferId			 m_frame_id		   = 0;
	BufferId			 m_default_ssbo_id = 0; // bound to the light SSBOs when no clustered lights are set
	gpu::VersionedOutput m_versioned;

	 public:
	~PbrRenderNode() override
	{
		if (m_device && m_sampler)
		{
			m_device.destroySampler(m_sampler);
		}
	}
	PbrRenderNode(const PbrRenderNode&)			   = delete;
	PbrRenderNode& operator=(const PbrRenderNode&) = delete;
	PbrRenderNode(PbrRenderNode&&)				   = delete;
	PbrRenderNode& operator=(PbrRenderNode&&)	   = delete;
};

PbrPass::PbrPass(graph::Graph& graph, vk::Format color_format, vk::Format depth_format,
				 graph::TypedHandle<vk::Extent2D> screen, graph::DataHandle output,
				 graph::TypedHandle<glm::mat4> view_proj, graph::TypedHandle<glm::vec4> eye, const PbrConfig& config)
	: m_graph(&graph)
	, m_output(output)
{
	auto render = std::make_unique<PbrRenderNode>(color_format, depth_format, screen, output, view_proj, eye, config);
	m_render	= render.get();
	m_node		= graph.add(std::move(render));
	graph.set_producer(output, m_node);
}

std::uint32_t PbrPass::add_material(const PbrMaterial& material)
{
	return m_render->add_material(material);
}

void PbrPass::add_object(graph::DataHandle mesh, graph::DataHandle model, std::uint32_t material)
{
	m_render->add_object(mesh, model, material);
}

void PbrPass::set_clustered_lights(graph::DataHandle view, graph::DataHandle lights, graph::DataHandle light_grid,
								   graph::DataHandle light_index, const culling::ClusterGrid& grid)
{
	m_render->set_clustered_lights(view, lights, light_grid, light_index, grid);
}
} // namespace veng::passes
