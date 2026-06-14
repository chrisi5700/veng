/**
 * @file
 * @author chris
 * @brief Implementation of @ref veng::passes::PhongPass — the @ref veng::passes::PhongRenderNode private class and the
 *        three-pipeline opaque/transparent draw recording.
 * @ingroup render_passes
 */

#include <array>
#include <cstddef>
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/gpu/VersionedOutput.hpp>
#include <veng/passes/PhongPass.hpp>
#include <veng/pipelines/GraphicsPipeline.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/resources/Image.hpp>
#include <veng/resources/RenderTargetSet.hpp>
#include <veng/resources/ResourcePool.hpp>
#include <veng/rhi/Convert.hpp>
#include <veng/shader/Shader.hpp>

namespace veng::passes
{
namespace
{
// The single 160-byte vertex push block — must match phong.vert's PushData exactly.
struct PhongPush
{
	glm::mat4 model;	 // 0
	glm::mat4 view_proj; // 64
	glm::vec4 color;	 // 128 (rgb tint, a opacity)
	glm::vec4 eye;		 // 144 (xyz eye, w shininess)
};
static_assert(sizeof(PhongPush) == 160, "PhongPush must match the shader's 160-byte push block");
} // namespace

// The rendering half of the pass: a custom GpuNode owning one color + depth target, drawing every
// object opaque-first then transparent-back-then-front with three pipelines built from the one
// shader pair. See PhongPass.hpp for the rationale (a GraphicsNode can't share a depth buffer).
class PhongRenderNode final : public gpu::GpuNode
{
	 public:
	PhongRenderNode(rhi::Format color_format, rhi::Format depth_format, graph::DataHandle screen,
					graph::DataHandle output, graph::DataHandle view_proj, graph::DataHandle eye,
					PhongConfig config) noexcept
		: m_color_format(color_format)
		, m_depth_format(depth_format)
		, m_output(output)
		, m_view_proj(view_proj)
		, m_eye(eye)
		, m_config(config)
		, m_inputs{screen, view_proj, eye}
	{
	}

	void add_object(graph::DataHandle mesh, graph::DataHandle model, glm::vec4 color, float shininess)
	{
		m_objects.push_back(Object{
			.mesh = mesh, .model = model, .color = color, .shininess = shininess, .transparent = color.a < 1.0F});
		m_inputs.push_back(mesh);
		m_inputs.push_back(model);
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
		const glm::mat4 view_proj = view_proj_d->value();
		const glm::vec3 eye_pos	  = glm::vec3(eye_d->value());

		// The RenderTargetSet (configured in ensure_pipelines) declares the color + depth targets
		// once, acquires this frame's copies, and resolves the MSAA attachment when enabled.
		if (auto acquired = m_targets.acquire(ctx.pool(), extent); !acquired.has_value())
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		Image* const color_image = m_targets.color();

		rhi::CommandEncoder& enc = ctx.encoder();
		m_targets.begin(ctx.pool(), enc, extent, m_config.clear_color);
		enc.set_viewport_scissor(rhi::to_rhi(extent));

		// Opaque first (writes depth), then each translucent object's far faces, then its near
		// faces — both depth-tested against the opaque pass but not writing depth, alpha-blended.
		struct Batch
		{
			const GraphicsPipeline* pipe;
			bool					transparent;
		};
		const auto draw_batch = [&](const Batch& batch) -> std::expected<void, graph::ExecError>
		{
			enc.bind_pipeline(batch.pipe->handle());
			for (const Object& obj : m_objects)
			{
				if (obj.transparent != batch.transparent)
				{
					continue;
				}
				const auto* model_d = dynamic_cast<graph::ValueData<glm::mat4>*>(ctx.data(obj.model));
				const auto* mesh_d	= dynamic_cast<graph::ValueData<gpu::MeshRef>*>(ctx.data(obj.mesh));
				if (model_d == nullptr || mesh_d == nullptr)
				{
					return std::unexpected(graph::ExecError::MISSING_INPUT);
				}
				const gpu::MeshRef mesh = mesh_d->value();
				if (!mesh.vertex_buffer.valid())
				{
					continue; // mesh not uploaded yet — skip until its MeshNode has produced
				}
				// eye.w carries the specular exponent: per-object when set (>= 0), else the config default.
				const float		shininess = obj.shininess >= 0.0F ? obj.shininess : m_config.shininess;
				const PhongPush push{.model		= model_d->value(),
									 .view_proj = view_proj,
									 .color		= obj.color,
									 .eye		= glm::vec4(eye_pos, shininess)};
				enc.push_constants(
					batch.pipe->handle(), rhi::ShaderStage::VERTEX, 0,
					std::span<const std::byte>(static_cast<const std::byte*>(static_cast<const void*>(&push)),
											   sizeof(PhongPush)));
				enc.bind_vertex_buffer(mesh.vertex_buffer);
				if (mesh.index_buffer.valid())
				{
					enc.bind_index_buffer(mesh.index_buffer, mesh.index_type);
					enc.draw_indexed(mesh.index_count, 1);
				}
				else
				{
					enc.draw(mesh.vertex_count, 1);
				}
			}
			return {};
		};

		std::expected<void, graph::ExecError> drawn = draw_batch(Batch{.pipe = &*m_opaque, .transparent = false});
		if (drawn.has_value())
		{
			drawn = draw_batch(Batch{.pipe = &*m_transparent_back, .transparent = true});
		}
		if (drawn.has_value())
		{
			drawn = draw_batch(Batch{.pipe = &*m_transparent_front, .transparent = true});
		}
		enc.end_rendering();
		if (!drawn.has_value())
		{
			return std::unexpected(drawn.error());
		}

		// Publish the lit-scene ref (version-bumped so consumers see the change). Left in
		// eColorAttachmentOptimal; the consumer's image_usages drives the transition it needs.
		m_versioned.publish(ctx, m_output,
							gpu::ImageRef{.texture		= color_image->handle(),
										  .extent		= rhi::to_rhi(extent),
										  .format		= m_color_format,
										  .sample_count = rhi::to_rhi(color_image->sample_count()),
										  .pool_id		= m_targets.color_id()});
		return true;
	}

	 private:
	struct Object
	{
		graph::DataHandle mesh;
		graph::DataHandle model;
		glm::vec4		  color;
		float			  shininess; // per-object specular exponent; < 0 => use PhongConfig::shininess
		bool			  transparent;
	};

	std::expected<void, graph::ExecError> ensure_pipelines(gpu::GpuExecContext& ctx)
	{
		if (m_opaque.has_value())
		{
			return {};
		}
		auto vert = Shader::create_shader(ctx.device(), "passes/phong.vert");
		auto frag = Shader::create_shader(ctx.device(), "passes/phong.frag");
		if (!vert.has_value() || !frag.has_value())
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		// Clamp the requested MSAA to the device and configure the targets + every pipeline with the
		// same sample count (Vulkan requires the pipeline match its multisampled attachments).
		const vk::SampleCountFlagBits samples = clamp_sample_count(ctx.context(), rhi::to_vk(m_config.samples));
		m_targets.configure(rhi::to_vk(m_color_format), rhi::to_vk(m_depth_format), samples);

		const std::array color_formats{m_color_format};
		// All three pipelines share the shader pair; only cull / depth-write / blend differ.
		struct PipeState
		{
			rhi::CullMode cull;
			bool		  depth_write;
			bool		  blend;
		};
		const auto make = [&](const PipeState& state) -> std::expected<GraphicsPipeline, PipelineError>
		{
			GraphicsPipelineBuilder builder(vert.value(), frag.value());
			// FrontFace is eClockwise, NOT eCounterClockwise: the engine geometry is CCW-front in
			// world space, but the camera's projection negates Y for Vulkan clip space
			// (proj[1][1] *= -1), which flips triangle winding in *framebuffer* space — where Vulkan
			// decides facing. So world-front triangles read as clockwise here, and eBack then culls
			// the true far faces (eFront the near). Get this wrong and back-then-front inverts:
			// the "front" batch draws far faces last and you see only backfaces.
			builder.color_formats(color_formats)
				.depth_format(m_depth_format)
				.depth_write(state.depth_write)
				.sample_count(rhi::to_rhi(samples))
				.rasterization(rhi::PolygonMode::FILL, state.cull, rhi::FrontFace::CLOCKWISE)
				.blend(state.blend);
			return builder.build(ctx.context());
		};
		// front faces, write depth | far faces of glass | near faces of glass
		auto opaque = make(PipeState{.cull = rhi::CullMode::BACK, .depth_write = true, .blend = false});
		auto back	= make(PipeState{.cull = rhi::CullMode::FRONT, .depth_write = false, .blend = true});
		auto front	= make(PipeState{.cull = rhi::CullMode::BACK, .depth_write = false, .blend = true});
		if (!opaque.has_value() || !back.has_value() || !front.has_value())
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		m_opaque			= std::move(opaque.value());
		m_transparent_back	= std::move(back.value());
		m_transparent_front = std::move(front.value());
		return {};
	}

	rhi::Format					   m_color_format;
	rhi::Format					   m_depth_format;
	graph::DataHandle			   m_output;
	graph::DataHandle			   m_view_proj;
	graph::DataHandle			   m_eye;
	PhongConfig					   m_config;
	std::vector<graph::DataHandle> m_inputs; // [screen, view_proj, eye, then mesh+model per object]
	std::vector<Object>			   m_objects;

	std::optional<GraphicsPipeline> m_opaque;
	std::optional<GraphicsPipeline> m_transparent_back;
	std::optional<GraphicsPipeline> m_transparent_front;

	RenderTargetSet		 m_targets; // color + depth render target, with MSAA resolve when configured
	gpu::VersionedOutput m_versioned;
};

PhongPass::PhongPass(graph::Graph& graph, rhi::Format color_format, rhi::Format depth_format,
					 graph::TypedHandle<vk::Extent2D> screen, graph::DataHandle output,
					 graph::TypedHandle<glm::mat4> view_proj, graph::TypedHandle<glm::vec4> eye,
					 const PhongConfig& config)
	: m_output(output)
{
	auto render = std::make_unique<PhongRenderNode>(color_format, depth_format, screen, output, view_proj, eye, config);
	m_render	= render.get();
	m_node		= graph.add(std::move(render));
	graph.set_producer(output, m_node);
}

void PhongPass::add_object(graph::DataHandle mesh, graph::DataHandle model, glm::vec4 color, float shininess)
{
	m_render->add_object(mesh, model, color, shininess);
}
} // namespace veng::passes
