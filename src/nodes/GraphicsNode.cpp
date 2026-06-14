/**
 * @file
 * @author chris
 * @brief GraphicsNode implementation: pipeline build, descriptor writes, and multi-draw recording.
 * @ingroup graph_nodes
 */

#include <array>
#include <cstddef>
#include <span>
#include <utility>
#include <variant>
#include <vector>
#include <veng/gpu/BufferRef.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/gpu/UniformRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/resources/SamplerConvert.hpp>
#include <veng/rhi/Convert.hpp>
#include <veng/shader/Shader.hpp>

namespace veng::nodes
{
GraphicsNode::GraphicsNode(std::string vertex_shader, std::string fragment_shader, rhi::Format color_format,
						   rhi::Format depth_format, std::uint32_t vertex_count, graph::DataHandle screen_size,
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
		usages.push_back(gpu::ImageUsage{.id = ref.pool_id, .usage = rhi::TextureUsage::SAMPLED});
	}
	return usages;
}

std::expected<bool, graph::ExecError> GraphicsNode::record(gpu::GpuExecContext& ctx)
{
	const bool has_depth = m_depth_format != rhi::Format::UNDEFINED;

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
		// Clamp the requested MSAA level to the device once, then configure both the pipeline and the
		// render targets with the same sample count (Vulkan requires the pipeline match its attachments).
		const vk::SampleCountFlagBits samples = clamp_sample_count(ctx.context(), rhi::to_vk(m_samples));
		m_targets.configure(rhi::to_vk(m_color_format), has_depth ? rhi::to_vk(m_depth_format) : vk::Format::eUndefined,
							samples);

		const std::array		color_formats{m_color_format};
		GraphicsPipelineBuilder builder(vert.value(), frag.value());
		builder.color_formats(color_formats)
			.topology(m_topology)
			.sample_count(rhi::to_rhi(samples))
			.rasterization(rhi::PolygonMode::FILL, rhi::CullMode::NONE, rhi::FrontFace::COUNTER_CLOCKWISE);
		if (has_depth)
		{
			builder.depth_format(m_depth_format).depth_write(m_depth_write);
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

		// Capture the reflected stride of vertex binding 0 so each bound MeshRef can be validated
		// against the layout the shader actually expects. A shader with no vertex inputs (the
		// SV_VertexID fullscreen / cube path) leaves this empty and is never checked.
		if (const auto* vertex_details = std::get_if<VertexDetails>(&vert.value().get_details()))
		{
			for (const VertexBinding& binding : vertex_details->bindings)
			{
				if (binding.binding == 0)
				{
					m_expected_vertex_stride = binding.stride;
					break;
				}
			}
		}
	}

	// A node that samples images needs a sampler; create one lazily (linear, clamp-to-edge —
	// the right default for post-process passes reading a full-screen texture). Owned here and
	// freed in the destructor; m_device is captured for that.
	if (!m_sampled_images.empty() && !m_sampler_handle.valid())
	{
		auto sampler = ctx.rhi().create_sampler(gpu::to_create_info(m_sampler_config));
		if (!sampler.has_value())
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		m_sampler_handle = sampler.value();
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

	// Validate vertex strides up front, before any GPU recording opens a render pass: a mesh whose
	// byte stride disagrees with the shader's reflected vertex layout would be strided wrong and
	// draw garbage. Fail with a typed error here rather than silently corrupt. Only checked when
	// both sides are known (the shader has vertex inputs and the mesh carries a stride).
	if (m_expected_vertex_stride.has_value())
	{
		for (const Draw& draw : m_draws)
		{
			if (!draw.mesh.has_value())
			{
				continue;
			}
			const auto* mesh_data = dynamic_cast<graph::ValueData<gpu::MeshRef>*>(ctx.data(*draw.mesh));
			if (mesh_data == nullptr)
			{
				return std::unexpected(graph::ExecError::MISSING_INPUT);
			}
			if (const std::uint32_t stride = mesh_data->value().vertex_stride;
				stride != 0 && stride != *m_expected_vertex_stride)
			{
				Logger::instance().error(
					"GraphicsNode vertex stride mismatch: mesh has {} bytes/vertex but shader '{}' reflects {} "
					"— the vertex struct does not match the shader's input layout",
					stride, m_vertex_shader, *m_expected_vertex_stride);
				return std::unexpected(graph::ExecError::NODE_FAILED);
			}
		}
	}

	// The pool owns the render targets (N-buffered, a copy per in-flight frame), so the node holds
	// no VkImage of its own — frame N+1 never stomps a target frame N's GPU work is still reading.
	// The RenderTargetSet declares them once, acquires this frame's copies, and (under MSAA)
	// resolves the multisampled attachment into the single-sample image consumers read.
	if (auto acquired = m_targets.acquire(ctx.pool(), extent); !acquired.has_value())
	{
		return std::unexpected(graph::ExecError::NODE_FAILED);
	}
	Image* const color_image = m_targets.color();
	m_last_color			 = color_image;

	rhi::CommandEncoder& enc = ctx.encoder();
	m_targets.begin(ctx.pool(), enc, extent, m_clear_color);

	enc.bind_pipeline(m_pipeline->handle());
	enc.set_viewport_scissor(rhi::to_rhi(extent));

	// Descriptors: write the uniform-buffer + sampled-image edges into a set at their name-matched
	// bindings (plus the node's sampler at every reflected SamplerState), then bind it. The set is
	// PER frame slot and rewritten each record: a slot's set was last used by the frame that
	// previously held that slot (now retired), so re-writing it is safe at any frames-in-flight,
	// and it always reflects this frame's uniform-buffer copies + sampled views (both N-buffered).
	if (!m_uniforms.empty() || !m_storage_buffers.empty() || !m_sampled_images.empty())
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
		buffer_infos.reserve(m_uniforms.size() + m_storage_buffers.size());
		image_infos.reserve(m_sampled_images.size() + m_descriptors_by_name.size());
		writes.reserve(m_uniforms.size() + m_storage_buffers.size() + m_sampled_images.size() +
					   m_descriptors_by_name.size());

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
			buffer_infos.push_back(
				vk::DescriptorBufferInfo().setBuffer(ctx.rhi().buffer(ref.buffer)).setOffset(0).setRange(ref.size));
			writes.push_back(vk::WriteDescriptorSet()
								 .setDstSet(set)
								 .setDstBinding(static_cast<std::uint32_t>(it->second.binding))
								 .setDstArrayElement(0)
								 .setDescriptorType(vk::DescriptorType::eUniformBuffer)
								 .setBufferInfo(buffer_infos.back()));
		}

		// Storage-buffer edges (SSBOs / StructuredBuffers): identical to the uniform path but
		// the descriptor type is eStorageBuffer, and the published BufferRef carries the
		// per-element stride + count (the count drives `instanceCount` of any draw that opts
		// in via set_instances_from). The shader sees the raw array; we do not bind by stride
		// here — std430 layout in the shader is the contract.
		for (const graph::DataHandle handle : m_storage_buffers)
		{
			const auto* storage = dynamic_cast<graph::ValueData<gpu::BufferRef>*>(ctx.data(handle));
			if (storage == nullptr)
			{
				return std::unexpected(graph::ExecError::MISSING_INPUT);
			}
			const gpu::BufferRef ref = storage->value();
			const auto			 it	 = m_descriptors_by_name.find(ref.name);
			if (it == m_descriptors_by_name.end())
			{
				return std::unexpected(graph::ExecError::NODE_FAILED); // no such reflected binding
			}
			buffer_infos.push_back(
				vk::DescriptorBufferInfo().setBuffer(ctx.rhi().buffer(ref.buffer)).setOffset(0).setRange(ref.size));
			writes.push_back(vk::WriteDescriptorSet()
								 .setDstSet(set)
								 .setDstBinding(static_cast<std::uint32_t>(it->second.binding))
								 .setDstArrayElement(0)
								 .setDescriptorType(vk::DescriptorType::eStorageBuffer)
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
			if (it == m_descriptors_by_name.end() || !ctx.rhi().view(ref.texture))
			{
				return std::unexpected(graph::ExecError::NODE_FAILED);
			}
			// Retain the pooled copy we sample so it is not recycled while this frame is in flight
			// (the producer may be cached this frame, leaving us reading an older physical copy).
			ctx.pool().consume(ref);
			image_infos.push_back(vk::DescriptorImageInfo()
									  .setImageView(ctx.rhi().view(ref.texture))
									  .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal));
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
				image_infos.push_back(vk::DescriptorImageInfo().setSampler(ctx.rhi().sampler(m_sampler_handle)));
				writes.push_back(vk::WriteDescriptorSet()
									 .setDstSet(set)
									 .setDstBinding(static_cast<std::uint32_t>(info.binding))
									 .setDstArrayElement(0)
									 .setDescriptorType(vk::DescriptorType::eSampler)
									 .setImageInfo(image_infos.back()));
			}
		}

		ctx.device().updateDescriptorSets(writes, {});
		enc.bind_descriptor_set(m_pipeline->handle(), set);
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
				enc.push_constants(m_pipeline->handle(), push.stage, push.offset,
								   std::span<const std::byte>(static_cast<const std::byte*>(bytes), push.size));
			}
		}

		// Compute the instance count: a `set_instances_from(handle)` reads the count off the
		// published BufferRef each record (so a CPU resize of the source array drives the
		// draw size); otherwise the fixed `instance_count` (default 1) applies. A 0-count
		// storage-driven instanced draw emits no instances and is a legitimate no-op (the
		// physics scene has no bodies this step) — not a node failure.
		std::uint32_t instance_count = draw.instance_count;
		if (draw.instance_count_source.has_value())
		{
			const auto* slot = dynamic_cast<graph::ValueData<gpu::BufferRef>*>(ctx.data(*draw.instance_count_source));
			if (slot == nullptr)
			{
				return std::unexpected(graph::ExecError::MISSING_INPUT);
			}
			instance_count = slot->value().count;
		}
		if (instance_count == 0)
		{
			return {}; // nothing to draw this frame — empty body list, no work
		}

		if (draw.mesh.has_value())
		{
			const auto* mesh_data = dynamic_cast<graph::ValueData<gpu::MeshRef>*>(ctx.data(*draw.mesh));
			if (mesh_data == nullptr)
			{
				return std::unexpected(graph::ExecError::MISSING_INPUT);
			}
			const gpu::MeshRef mesh = mesh_data->value();
			enc.bind_vertex_buffer(mesh.vertex_buffer);
			if (mesh.index_buffer.valid())
			{
				enc.bind_index_buffer(mesh.index_buffer, mesh.index_type);
				enc.draw_indexed(mesh.index_count, instance_count);
			}
			else
			{
				enc.draw(mesh.vertex_count, instance_count);
			}
		}
		else
		{
			enc.draw(draw.vertex_count, instance_count);
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
	enc.end_rendering();

	// No end-of-pass transition: the image stays in eColorAttachmentOptimal. The CONSUMER's
	// declared `image_usages` drives the layout it actually needs (sampled / transfer-src), and
	// the executor inserts the transition via `prepare_for` — the engine owns layouts, not the
	// producer. Tests that read back directly transition the image themselves via the pool.

	// Publish a ref to the rendered target tagged with its pool id so a consumer can retain this
	// physical copy while in flight. Non-comparable, so each produce is a change — a consumer
	// re-evaluates whenever we re-render, and caches when we do not.
	m_versioned.publish(ctx, m_output,
						gpu::ImageRef{.texture		= color_image->handle(),
									  .extent		= rhi::to_rhi(extent),
									  .format		= m_color_format,
									  .sample_count = rhi::to_rhi(color_image->sample_count()),
									  .pool_id		= m_targets.color_id()});
	return true;
}
} // namespace veng::nodes
