//
// Created by chris on 5/26/26.
//
// See GraphicsNode.hpp and design.md §L4.
//

#include <array>
#include <cstddef>
#include <utility>
#include <vector>
#include <veng/gpu/ImageRef.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/gpu/UniformRef.hpp>
#include <veng/managers/CommandManager.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/shader/Shader.hpp>

namespace veng::nodes
{
GraphicsNode::GraphicsNode(std::string vertex_shader, std::string fragment_shader, vk::Format color_format,
						   vk::Format depth_format, std::uint32_t vertex_count, graph::DataHandle screen_size,
						   graph::DataHandle output) noexcept
	: m_vertex_shader(std::move(vertex_shader))
	, m_fragment_shader(std::move(fragment_shader))
	, m_color_format(color_format)
	, m_depth_format(depth_format)
	, m_vertex_count(vertex_count)
	, m_output(output)
	, m_inputs{screen_size}
{
}

GraphicsNode::~GraphicsNode()
{
	if (m_device && m_sampler)
	{
		m_device.destroySampler(m_sampler);
	}
}

std::vector<gpu::ImageUsage> GraphicsNode::image_usages(graph::ExecContext& ctx)
{
	// Declare the read-side dependencies for the executor's auto-barrier pass: each pool-backed
	// sampled input must be in SHADER_READ_ONLY before record begins. Producer-side transitions
	// (this node's own color/depth targets) stay in record() — they need a freshly-acquired copy
	// and an extent that prepare_for cannot supply.
	std::vector<gpu::ImageUsage> usages;
	usages.reserve(m_sampled_images.size());
	for (const SampledBinding& binding : m_sampled_images)
	{
		const auto* slot = dynamic_cast<graph::ValueData<gpu::ImageRef>*>(ctx.data(m_inputs[binding.input_index]));
		if (slot == nullptr)
		{
			continue;
		}
		const gpu::ImageRef ref = slot->value();
		if (ref.pool_id == gpu::ImageRef::INVALID_POOL_ID)
		{
			continue;
		}
		usages.push_back(gpu::ImageUsage{.id	 = ref.pool_id,
										 .layout = vk::ImageLayout::eShaderReadOnlyOptimal,
										 .stage	 = vk::PipelineStageFlagBits2::eFragmentShader,
										 .access = vk::AccessFlagBits2::eShaderSampledRead});
	}
	return usages;
}

std::expected<bool, graph::ExecError> GraphicsNode::record(gpu::GpuExecContext& ctx)
{
	const bool has_depth = m_depth_format != vk::Format::eUndefined;

	// Build the pipeline the first time we have a device. The node owns this entirely —
	// graph construction never sees the L2 pipeline layer. A shader/build failure surfaces
	// as a node failure (the frame is dropped, the node stays dirty and retries).
	if (!m_pipeline.has_value())
	{
		auto vert = Shader::create_shader(ctx.device(), m_vertex_shader);
		auto frag = Shader::create_shader(ctx.device(), m_fragment_shader);
		if (!vert.has_value() || !frag.has_value())
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		const std::array		color_formats{m_color_format};
		GraphicsPipelineBuilder builder(vert.value(), frag.value());
		builder.color_formats(color_formats)
			.rasterization(vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise);
		if (has_depth)
		{
			builder.depth_format(m_depth_format);
		}
		auto pipeline = builder.build(ctx.context());
		if (!pipeline.has_value())
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		m_pipeline = std::move(pipeline.value());

		// Remember each descriptor by its reflected name (binding + type), so add_uniform /
		// add_sampled_image edges can be matched to a binding by name and written with the
		// right kind (the pipeline owns only the merged layout, not the names). Same merge as
		// the pipeline builder: both stages, keyed by name.
		for (const DescriptorInfo& info : vert.value().get_descriptor_infos())
		{
			m_descriptors_by_name[info.name] = info;
		}
		for (const DescriptorInfo& info : frag.value().get_descriptor_infos())
		{
			m_descriptors_by_name[info.name] = info;
		}
	}

	// A node that samples images needs a sampler; create one lazily (linear, clamp-to-edge —
	// the right default for post-process passes reading a full-screen texture). Owned here and
	// freed in the destructor; m_device is captured for that.
	if (!m_sampled_images.empty() && !m_sampler)
	{
		m_device		   = ctx.device();
		const auto info	   = vk::SamplerCreateInfo()
								 .setMagFilter(vk::Filter::eLinear)
								 .setMinFilter(vk::Filter::eLinear)
								 .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
								 .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
								 .setAddressModeW(vk::SamplerAddressMode::eClampToEdge);
		const auto sampler = ctx.device().createSampler(info);
		if (sampler.result != vk::Result::eSuccess)
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		m_sampler = sampler.value;
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

	// Declare the pool resources once, then acquire this frame's physical copies. The pool owns
	// the targets and N-buffers them (a copy per in-flight frame), so the node holds no VkImage of
	// its own — frame N+1 never stomps a target frame N's GPU work is still reading, and a resize
	// is just the pool reallocating on a new extent.
	if (!m_declared)
	{
		m_color_id = ctx.pool().declare_image(m_color_format, vk::ImageUsageFlagBits::eColorAttachment |
																  vk::ImageUsageFlagBits::eTransferSrc |
																  vk::ImageUsageFlagBits::eSampled);
		if (has_depth)
		{
			m_depth_id = ctx.pool().declare_image(m_depth_format, vk::ImageUsageFlagBits::eDepthStencilAttachment,
												  vk::ImageAspectFlagBits::eDepth);
		}
		m_declared = true;
	}
	auto color = ctx.pool().acquire_image(m_color_id, extent);
	if (!color.has_value())
	{
		return std::unexpected(graph::ExecError::NODE_FAILED);
	}
	Image* const color_image = color.value();
	m_last_color			 = color_image;
	Image* depth_image		 = nullptr;
	if (has_depth)
	{
		auto depth = ctx.pool().acquire_image(m_depth_id, extent);
		if (!depth.has_value())
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		depth_image = depth.value();
	}

	const vk::CommandBuffer cmd = ctx.command_buffer();

	// Auto-tracked transitions: the pool inserts the barrier and updates its tracker, so the
	// CONSUMER side (prepare_for, for sampled inputs) can reason about layout correctly without
	// a desync between actual GPU state and the pool's record.
	ctx.pool().transition_image(m_color_id, cmd, vk::ImageLayout::eColorAttachmentOptimal,
								vk::PipelineStageFlagBits2::eColorAttachmentOutput,
								vk::AccessFlagBits2::eColorAttachmentWrite);
	if (has_depth)
	{
		ctx.pool().transition_image(m_depth_id, cmd, vk::ImageLayout::eDepthStencilAttachmentOptimal,
									vk::PipelineStageFlagBits2::eEarlyFragmentTests,
									vk::AccessFlagBits2::eDepthStencilAttachmentWrite);
	}

	const auto color_clear		= vk::ClearValue().setColor(vk::ClearColorValue(m_clear_color));
	const auto color_attachment = vk::RenderingAttachmentInfo()
									  .setImageView(color_image->view())
									  .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
									  .setLoadOp(vk::AttachmentLoadOp::eClear)
									  .setStoreOp(vk::AttachmentStoreOp::eStore)
									  .setClearValue(color_clear);
	const auto depth_clear		= vk::ClearValue().setDepthStencil(vk::ClearDepthStencilValue(1.0F, 0));
	const auto depth_attachment = vk::RenderingAttachmentInfo()
									  .setImageView(has_depth ? depth_image->view() : vk::ImageView{})
									  .setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
									  .setLoadOp(vk::AttachmentLoadOp::eClear)
									  .setStoreOp(vk::AttachmentStoreOp::eDontCare)
									  .setClearValue(depth_clear);
	auto	   rendering		= vk::RenderingInfo()
									  .setRenderArea(vk::Rect2D({0, 0}, extent))
									  .setLayerCount(1)
									  .setColorAttachments(color_attachment);
	if (has_depth)
	{
		rendering.setPDepthAttachment(&depth_attachment);
	}
	cmd.beginRendering(rendering);

	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline->pipeline());
	cmd.setViewport(
		0, vk::Viewport(0.0F, 0.0F, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0F, 1.0F));
	cmd.setScissor(0, vk::Rect2D({0, 0}, extent));

	// Descriptors: write the uniform-buffer + sampled-image edges into a set at their name-matched
	// bindings (plus the node's sampler at every reflected SamplerState), then bind it. The set is
	// PER frame slot and rewritten each record: a slot's set was last used by the frame that
	// previously held that slot (now retired), so re-writing it is safe at any frames-in-flight,
	// and it always reflects this frame's uniform-buffer copies + sampled views (both N-buffered).
	if (!m_uniforms.empty() || !m_sampled_images.empty())
	{
		const std::size_t slot = ctx.frame_slot();
		if (!m_descriptors.has_value())
		{
			m_descriptors.emplace(ctx.device());
		}
		if (slot >= m_descriptor_sets.size())
		{
			m_descriptor_sets.resize(slot + 1, vk::DescriptorSet{});
		}
		if (!m_descriptor_sets[slot])
		{
			auto set = m_descriptors->allocate(m_pipeline->descriptor_set_layout());
			if (!set.has_value())
			{
				return std::unexpected(graph::ExecError::NODE_FAILED);
			}
			m_descriptor_sets[slot] = set.value();
		}
		const vk::DescriptorSet set = m_descriptor_sets[slot];

		// Reserved so .back() addresses stay stable for the WriteDescriptorSet proxies.
		std::vector<vk::DescriptorBufferInfo> buffer_infos;
		std::vector<vk::DescriptorImageInfo>  image_infos;
		std::vector<vk::WriteDescriptorSet>	  writes;
		buffer_infos.reserve(m_uniforms.size());
		image_infos.reserve(m_sampled_images.size() + m_descriptors_by_name.size());
		writes.reserve(m_uniforms.size() + m_sampled_images.size() + m_descriptors_by_name.size());

		for (const graph::DataHandle handle : m_uniforms)
		{
			const auto* uniform = dynamic_cast<graph::ValueData<gpu::UniformRef>*>(ctx.data(handle));
			if (uniform == nullptr)
			{
				return std::unexpected(graph::ExecError::MISSING_INPUT);
			}
			const gpu::UniformRef ref = uniform->value();
			const auto			  it  = m_descriptors_by_name.find(ref.name);
			if (it == m_descriptors_by_name.end())
			{
				return std::unexpected(graph::ExecError::NODE_FAILED); // no such reflected binding
			}
			buffer_infos.push_back(vk::DescriptorBufferInfo().setBuffer(ref.buffer).setOffset(0).setRange(ref.size));
			writes.push_back(vk::WriteDescriptorSet()
								 .setDstSet(set)
								 .setDstBinding(static_cast<std::uint32_t>(it->second.binding))
								 .setDstArrayElement(0)
								 .setDescriptorType(vk::DescriptorType::eUniformBuffer)
								 .setBufferInfo(buffer_infos.back()));
		}

		for (const SampledBinding& binding : m_sampled_images)
		{
			const auto* sampled =
				dynamic_cast<graph::ValueData<gpu::ImageRef>*>(ctx.data(m_inputs[binding.input_index]));
			const auto it = m_descriptors_by_name.find(binding.name);
			if (sampled == nullptr)
			{
				return std::unexpected(graph::ExecError::MISSING_INPUT);
			}
			const gpu::ImageRef ref = sampled->value();
			if (it == m_descriptors_by_name.end() || !ref.view)
			{
				return std::unexpected(graph::ExecError::NODE_FAILED);
			}
			// Retain the pooled copy we sample so it is not recycled while this frame is in flight
			// (the producer may be cached this frame, leaving us reading an older physical copy).
			if (ref.pool_id != gpu::ImageRef::INVALID_POOL_ID)
			{
				ctx.pool().touch(ref.pool_id);
			}
			image_infos.push_back(vk::DescriptorImageInfo().setImageView(ref.view).setImageLayout(
				vk::ImageLayout::eShaderReadOnlyOptimal));
			writes.push_back(vk::WriteDescriptorSet()
								 .setDstSet(set)
								 .setDstBinding(static_cast<std::uint32_t>(it->second.binding))
								 .setDstArrayElement(0)
								 .setDescriptorType(vk::DescriptorType::eSampledImage)
								 .setImageInfo(image_infos.back()));
		}

		// One node-owned sampler -> every reflected SamplerState binding.
		for (const auto& [name, info] : m_descriptors_by_name)
		{
			if (info.type == vk::DescriptorType::eSampler)
			{
				image_infos.push_back(vk::DescriptorImageInfo().setSampler(m_sampler));
				writes.push_back(vk::WriteDescriptorSet()
									 .setDstSet(set)
									 .setDstBinding(static_cast<std::uint32_t>(info.binding))
									 .setDstArrayElement(0)
									 .setDescriptorType(vk::DescriptorType::eSampler)
									 .setImageInfo(image_infos.back()));
			}
		}

		ctx.device().updateDescriptorSets(writes, {});
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipeline->layout(), 0, set, {});
	}

	// Record each draw: its push constants, then its geometry. All draws share the pipeline,
	// target and descriptor set bound above — only the push constants and the mesh vary. A node
	// with no configured draws falls back to a single SV_VertexID draw of vertex_count (the
	// fullscreen / triangle case). Buffer-backed geometry (a MeshNode) is drawn indexed when the
	// ref carries an index buffer, else a non-indexed draw of its vertex count.
	const auto record_draw = [&](const Draw& draw) -> std::expected<void, graph::ExecError>
	{
		for (const PushBinding& push : draw.push_constants)
		{
			if (const void* bytes = push.read(ctx); bytes != nullptr)
			{
				cmd.pushConstants<std::byte>(
					m_pipeline->layout(), push.stage, push.offset,
					vk::ArrayProxy<const std::byte>(push.size, static_cast<const std::byte*>(bytes)));
			}
		}
		if (draw.mesh.has_value())
		{
			const auto* mesh_data = dynamic_cast<graph::ValueData<gpu::MeshRef>*>(ctx.data(*draw.mesh));
			if (mesh_data == nullptr)
			{
				return std::unexpected(graph::ExecError::MISSING_INPUT);
			}
			const gpu::MeshRef	 mesh	= mesh_data->value();
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
		else
		{
			cmd.draw(draw.vertex_count, 1, 0, 0);
		}
		return {};
	};

	if (m_draws.empty())
	{
		if (auto drawn = record_draw(Draw{.vertex_count = m_vertex_count}); !drawn.has_value())
		{
			return std::unexpected(drawn.error());
		}
	}
	else
	{
		for (const Draw& draw : m_draws)
		{
			if (auto drawn = record_draw(draw); !drawn.has_value())
			{
				return std::unexpected(drawn.error());
			}
		}
	}
	cmd.endRendering();

	// No end-of-pass transition: the image stays in eColorAttachmentOptimal. The CONSUMER's
	// declared `image_usages` drives the layout it actually needs (sampled / transfer-src), and
	// the executor inserts the transition via `prepare_for` — the engine owns layouts, not the
	// producer. Tests that read back directly transition the image themselves via the pool.

	// Publish a ref to the rendered target tagged with its pool id so a consumer can retain this
	// physical copy while in flight. Non-comparable, so each produce is a change — a consumer
	// re-evaluates whenever we re-render, and caches when we do not.
	if (auto* out = dynamic_cast<graph::ValueData<gpu::ImageRef>*>(ctx.data(m_output)); out != nullptr)
	{
		++m_version; // bumped on every produce → comparable ImageRef sees the change
		(void)out->produce(gpu::ImageRef{.image	  = color_image->image(),
										 .view	  = color_image->view(),
										 .extent  = extent,
										 .format  = m_color_format,
										 .pool_id = m_color_id,
										 .version = m_version});
	}
	return true;
}
} // namespace veng::nodes
