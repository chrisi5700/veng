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

		// Remember each descriptor's binding by its reflected name, so add_uniform edges can
		// be matched to a binding by name (the pipeline owns only the merged layout, not the
		// names). Same merge as the pipeline builder: both stages, keyed by name.
		for (const DescriptorInfo& info : vert.value().get_descriptor_infos())
		{
			m_binding_by_name[info.name] = static_cast<std::uint32_t>(info.binding);
		}
		for (const DescriptorInfo& info : frag.value().get_descriptor_infos())
		{
			m_binding_by_name[info.name] = static_cast<std::uint32_t>(info.binding);
		}
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

	// (Re)create the owned target(s) from ScreenSize — a resize reallocates, which is just
	// an ordinary invalidation of this node.
	if (!m_color.has_value() || m_extent.width != extent.width || m_extent.height != extent.height)
	{
		auto color = Image::create(ctx.allocator(), ctx.device(), extent, m_color_format,
								   vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc);
		if (!color.has_value())
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		if (has_depth)
		{
			auto depth =
				Image::create(ctx.allocator(), ctx.device(), extent, m_depth_format,
							  vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::ImageAspectFlagBits::eDepth);
			if (!depth.has_value())
			{
				return std::unexpected(graph::ExecError::NODE_FAILED);
			}
			m_depth = std::move(depth.value());
		}
		m_color	 = std::move(color.value());
		m_extent = extent;
	}

	const vk::CommandBuffer cmd = ctx.command_buffer();

	CommandManager::image_barrier(
		cmd, m_color->image(), vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eNone,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite);
	if (has_depth)
	{
		CommandManager::image_barrier(
			cmd, m_depth->image(), vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal,
			vk::PipelineStageFlagBits2::eEarlyFragmentTests, vk::AccessFlagBits2::eNone,
			vk::PipelineStageFlagBits2::eEarlyFragmentTests, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
			vk::ImageAspectFlagBits::eDepth);
	}

	const auto color_clear		= vk::ClearValue().setColor(vk::ClearColorValue(m_clear_color));
	const auto color_attachment = vk::RenderingAttachmentInfo()
									  .setImageView(m_color->view())
									  .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
									  .setLoadOp(vk::AttachmentLoadOp::eClear)
									  .setStoreOp(vk::AttachmentStoreOp::eStore)
									  .setClearValue(color_clear);
	const auto depth_clear		= vk::ClearValue().setDepthStencil(vk::ClearDepthStencilValue(1.0F, 0));
	const auto depth_attachment = vk::RenderingAttachmentInfo()
									  .setImageView(has_depth ? m_depth->view() : vk::ImageView{})
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

	for (const PushBinding& push : m_push_constants)
	{
		if (const void* bytes = push.read(ctx); bytes != nullptr)
		{
			cmd.pushConstants<std::byte>(
				m_pipeline->layout(), push.stage, push.offset,
				vk::ArrayProxy<const std::byte>(push.size, static_cast<const std::byte*>(bytes)));
		}
	}

	// Uniform descriptors: write each add_uniform edge's buffer into a set at its
	// name-matched binding, then bind the set. The set is built once and reused — the uniform
	// buffers are persistent, so contents change in place; we only rebuild (and reset the
	// pool) when a buffer *handle* changes, which is safe because the prior frame using the
	// set has retired by the time this node re-records.
	if (!m_uniforms.empty())
	{
		std::vector<gpu::UniformRef> refs;
		refs.reserve(m_uniforms.size());
		for (const graph::DataHandle handle : m_uniforms)
		{
			const auto* slot = dynamic_cast<graph::ValueData<gpu::UniformRef>*>(ctx.data(handle));
			if (slot == nullptr)
			{
				return std::unexpected(graph::ExecError::MISSING_INPUT);
			}
			refs.push_back(slot->value());
		}

		std::vector<vk::Buffer> buffers;
		buffers.reserve(refs.size());
		for (const gpu::UniformRef& ref : refs)
		{
			buffers.push_back(ref.buffer);
		}

		if (!m_descriptor_set || buffers != m_bound_buffers)
		{
			if (!m_descriptors.has_value())
			{
				m_descriptors.emplace(ctx.device());
			}
			m_descriptors->reset();
			auto set = m_descriptors->allocate(m_pipeline->descriptor_set_layout());
			if (!set.has_value())
			{
				return std::unexpected(graph::ExecError::NODE_FAILED);
			}

			std::vector<vk::DescriptorBufferInfo> infos;
			std::vector<vk::WriteDescriptorSet>	  writes;
			infos.reserve(refs.size());
			writes.reserve(refs.size());
			for (const gpu::UniformRef& ref : refs)
			{
				const auto binding = m_binding_by_name.find(ref.name);
				if (binding == m_binding_by_name.end())
				{
					return std::unexpected(graph::ExecError::NODE_FAILED); // no such reflected binding
				}
				infos.push_back(vk::DescriptorBufferInfo().setBuffer(ref.buffer).setOffset(0).setRange(ref.size));
				writes.push_back(vk::WriteDescriptorSet()
									 .setDstSet(set.value())
									 .setDstBinding(binding->second)
									 .setDstArrayElement(0)
									 .setDescriptorType(vk::DescriptorType::eUniformBuffer)
									 .setBufferInfo(infos.back())); // infos is reserved -> address stable
			}
			ctx.device().updateDescriptorSets(writes, {});
			m_descriptor_set = set.value();
			m_bound_buffers	 = std::move(buffers);
		}

		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipeline->layout(), 0, m_descriptor_set, {});
	}

	// Buffer-backed geometry (a MeshNode) when a mesh edge is bound, else the SV_VertexID
	// draw of the constructor's vertex_count.
	if (m_mesh.has_value())
	{
		const auto* mesh_data = dynamic_cast<graph::ValueData<gpu::MeshRef>*>(ctx.data(*m_mesh));
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
		cmd.draw(m_vertex_count, 1, 0, 0);
	}
	cmd.endRendering();

	CommandManager::image_barrier(
		cmd, m_color->image(), vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);

	// Publish a ref to the rendered target (TRANSFER_SRC) on the scene edge for the blit.
	if (auto* out = dynamic_cast<graph::ValueData<gpu::ImageRef>*>(ctx.data(m_output)); out != nullptr)
	{
		(void)out->produce(gpu::ImageRef{
			.image = m_color->image(), .view = m_color->view(), .extent = m_extent, .format = m_color_format});
	}
	return true;
}
} // namespace veng::nodes
