//
// See veng/passes/PbrPass.hpp and review.md item 5.
//

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
#include <veng/descriptors/DescriptorAllocator.hpp>
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
	glm::mat4 view_proj;   // 0
	glm::vec4 camera_pos;  // 64
	glm::vec4 light_dir;   // 80
	glm::vec4 light_color; // 96
	glm::vec4 ambient;	   // 112
};
static_assert(sizeof(PbrFrame) == 128, "PbrFrame must match the shader's Frame uniform layout");

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

constexpr std::uint32_t FRAME_BINDING	= 0;
constexpr std::uint32_t TEXTURE_BINDING = 1; // base_color, normal, metal_rough, emissive, occlusion -> 1..5
constexpr std::uint32_t SAMPLER_BINDING = 6;
constexpr std::size_t	TEXTURE_COUNT	= 5;
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
		m_materials.push_back(Material{.textures   = {material.base_color, material.normal, material.metal_rough,
													  material.emissive, material.occlusion},
									   .base_color = material.base_color_factor,
									   .mr_factors = glm::vec4(material.metallic_factor, material.roughness_factor,
															   material.normal_scale, material.occlusion_strength),
									   .emissive   = glm::vec4(material.emissive_factor, 0.0F)});
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

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return m_inputs; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	 protected:
	std::expected<bool, graph::ExecError> record(gpu::GpuExecContext& ctx) override
	{
		if (auto built = ensure_pipeline(ctx); !built.has_value())
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
			m_declared = true;
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

		const PbrFrame frame_data{.view_proj   = view_proj_d->value(),
								  .camera_pos  = glm::vec4(glm::vec3(eye_d->value()), 1.0F),
								  .light_dir   = glm::vec4(glm::normalize(m_config.light_direction), 0.0F),
								  .light_color = glm::vec4(m_config.light_color * m_config.light_intensity, 0.0F),
								  .ambient	   = glm::vec4(m_config.ambient, 0.0F)};
		std::memcpy(frame.value()->mapped(), &frame_data, sizeof(PbrFrame));

		if (!ensure_sampler(ctx))
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		if (auto sets = write_descriptor_sets(ctx, frame.value()->buffer()); !sets.has_value())
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
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline->pipeline());

		const std::size_t slot = ctx.frame_slot();
		for (const Object& obj : m_objects)
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
				continue; // mesh not uploaded yet — skip until its MeshNode has produced
			}
			if (m_expected_vertex_stride.has_value() && mesh.vertex_stride != 0 &&
				mesh.vertex_stride != *m_expected_vertex_stride)
			{
				Logger::instance().error("PbrPass vertex stride mismatch: mesh has {} bytes/vertex, shader expects {}",
										 mesh.vertex_stride, *m_expected_vertex_stride);
				cmd.endRendering();
				return std::unexpected(graph::ExecError::NODE_FAILED);
			}

			const Material& material = m_materials[obj.material];
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipeline->layout(), 0,
								   m_sets[obj.material][slot], {});
			const PbrPush push{.model	   = model_d->value(),
							   .base_color = material.base_color,
							   .mr_factors = material.mr_factors,
							   .emissive   = material.emissive};
			cmd.pushConstants<std::byte>(
				m_pipeline->layout(), vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
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
		}
		cmd.endRendering();

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
		glm::vec4									 emissive;
	};
	struct Object
	{
		graph::DataHandle mesh;
		graph::DataHandle model;
		std::uint32_t	  material;
	};

	std::expected<void, graph::ExecError> ensure_pipeline(gpu::GpuExecContext& ctx)
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
		const std::array		color_formats{m_color_format};
		GraphicsPipelineBuilder builder(vert.value(), frag.value());
		// eCounterClockwise — NOTE this is the opposite of PhongPass. glTF defines front faces as
		// CCW-wound in model space (spec 3.7.2.1), whereas the engine's own hand-authored meshes
		// (example/common/Geometry.cpp) wind their *outward* faces CW. The Y-flipped projection
		// (proj[1][1] *= -1) negates winding in framebuffer space for both, so the model-space
		// difference carries through: glTF's CCW-front geometry reads as front under
		// eCounterClockwise here, and eBack then culls the true far faces. Using eClockwise (as
		// PhongPass correctly does for the engine's CW-front geometry) would cull a glTF model's
		// outer shell and leave only its backfaces visible. The integration test passes eNone.
		builder.color_formats(color_formats)
			.depth_format(m_depth_format)
			.depth_write(true)
			.rasterization(vk::PolygonMode::eFill, m_config.cull_mode, vk::FrontFace::eCounterClockwise);
		auto pipeline = builder.build(ctx.context());
		if (!pipeline.has_value())
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		m_pipeline = std::move(pipeline.value());

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
	// uniform at binding 0, the material's five textures at 1..5, the node sampler at 6. Per-slot
	// sets are safe to rewrite — the slot's set was last used by a now-retired frame.
	std::expected<void, graph::ExecError> write_descriptor_sets(gpu::GpuExecContext& ctx, vk::Buffer frame_buffer)
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

	std::optional<GraphicsPipeline>				m_pipeline;
	std::optional<std::uint32_t>				m_expected_vertex_stride;
	std::optional<DescriptorAllocator>			m_descriptors;
	std::vector<std::vector<vk::DescriptorSet>> m_sets; // [material][frame slot]
	vk::Sampler									m_sampler;
	vk::Device									m_device;

	bool				 m_declared = false;
	ImageId				 m_color_id = 0;
	ImageId				 m_depth_id = 0;
	BufferId			 m_frame_id = 0;
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
} // namespace veng::passes
