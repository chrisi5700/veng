//
// Created by chris on 5/25/26.
//
// See RasterCubeNode.hpp and design.md §L4.
//

#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <utility>
#include <veng/gpu/ImageRef.hpp>
#include <veng/managers/CommandManager.hpp>
#include <veng/nodes/RasterCubeNode.hpp>
#include <veng/rendergraph/data/Data.hpp>

namespace veng::nodes
{
RasterCubeNode::RasterCubeNode(GraphicsPipeline pipeline, vk::Format color_format, vk::Format depth_format,
							   graph::DataHandle screen_size, graph::DataHandle angle,
							   graph::DataHandle output) noexcept
	: m_pipeline(std::move(pipeline))
	, m_color_format(color_format)
	, m_depth_format(depth_format)
	, m_inputs{screen_size, angle}
	, m_output(output)
{
}

std::expected<bool, graph::ExecError> RasterCubeNode::record(gpu::GpuExecContext& ctx)
{
	const auto* size  = dynamic_cast<graph::ValueData<vk::Extent2D>*>(ctx.data(m_inputs[0]));
	const auto* angle = dynamic_cast<graph::ValueData<float>*>(ctx.data(m_inputs[1]));
	if (size == nullptr || angle == nullptr)
	{
		return std::unexpected(graph::ExecError::MISSING_INPUT);
	}
	const vk::Extent2D extent = size->value();
	if (extent.width == 0 || extent.height == 0)
	{
		return std::unexpected(graph::ExecError::NODE_FAILED);
	}

	// (Re)create the persistent color + depth targets from ScreenSize — a resize
	// reallocates both, which is just an ordinary invalidation of this node.
	if (!m_scene.has_value() || m_extent.width != extent.width || m_extent.height != extent.height)
	{
		auto color = Image::create(ctx.allocator(), ctx.device(), extent, m_color_format,
								   vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc);
		if (!color.has_value())
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		auto depth = Image::create(ctx.allocator(), ctx.device(), extent, m_depth_format,
								   vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::ImageAspectFlagBits::eDepth);
		if (!depth.has_value())
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		m_scene	 = std::move(color.value());
		m_depth	 = std::move(depth.value());
		m_extent = extent;
	}

	const vk::CommandBuffer cmd = ctx.command_buffer();

	CommandManager::image_barrier(
		cmd, m_scene->image(), vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eNone,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite);
	CommandManager::image_barrier(cmd, m_depth->image(), vk::ImageLayout::eUndefined,
								  vk::ImageLayout::eDepthStencilAttachmentOptimal,
								  vk::PipelineStageFlagBits2::eEarlyFragmentTests, vk::AccessFlagBits2::eNone,
								  vk::PipelineStageFlagBits2::eEarlyFragmentTests,
								  vk::AccessFlagBits2::eDepthStencilAttachmentWrite, vk::ImageAspectFlagBits::eDepth);

	const auto color_clear		= vk::ClearValue().setColor(vk::ClearColorValue(std::array{0.02F, 0.02F, 0.05F, 1.0F}));
	const auto color_attachment = vk::RenderingAttachmentInfo()
									  .setImageView(m_scene->view())
									  .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
									  .setLoadOp(vk::AttachmentLoadOp::eClear)
									  .setStoreOp(vk::AttachmentStoreOp::eStore)
									  .setClearValue(color_clear);
	const auto depth_clear		= vk::ClearValue().setDepthStencil(vk::ClearDepthStencilValue(1.0F, 0));
	const auto depth_attachment = vk::RenderingAttachmentInfo()
									  .setImageView(m_depth->view())
									  .setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
									  .setLoadOp(vk::AttachmentLoadOp::eClear)
									  .setStoreOp(vk::AttachmentStoreOp::eDontCare)
									  .setClearValue(depth_clear);
	cmd.beginRendering(vk::RenderingInfo()
						   .setRenderArea(vk::Rect2D({0, 0}, extent))
						   .setLayerCount(1)
						   .setColorAttachments(color_attachment)
						   .setPDepthAttachment(&depth_attachment));

	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline.pipeline());
	cmd.setViewport(
		0, vk::Viewport(0.0F, 0.0F, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0F, 1.0F));
	cmd.setScissor(0, vk::Rect2D({0, 0}, extent));

	// MVP from the reactive angle. glm is column-major (GLM_FORCE_DEPTH_ZERO_TO_ONE is
	// set for the Vulkan [0,1] depth range); proj[1][1] is negated for Vulkan's y-down
	// clip space. The matrix is uploaded raw — Slang lays the push constant out
	// row-major, so the shader applies it from the right (mul(pos, mvp)) to get M * v.
	const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
	glm::mat4	proj   = glm::perspective(glm::radians(55.0F), aspect, 0.1F, 20.0F);
	proj[1][1] *= -1.0F;
	const glm::mat4 view  = glm::lookAt(glm::vec3(2.4F, 1.7F, 2.4F), glm::vec3(0.0F), glm::vec3(0.0F, 1.0F, 0.0F));
	const glm::mat4 model = glm::rotate(glm::mat4(1.0F), angle->value(), glm::normalize(glm::vec3(0.3F, 1.0F, 0.2F)));
	const glm::mat4 mvp	  = proj * view * model;
	cmd.pushConstants<glm::mat4>(m_pipeline.layout(), vk::ShaderStageFlagBits::eVertex, 0, mvp);

	cmd.draw(36, 1, 0, 0);
	cmd.endRendering();

	CommandManager::image_barrier(
		cmd, m_scene->image(), vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);

	// Publish a ref to the rendered target (TRANSFER_SRC) on the scene edge for the blit.
	if (auto* out = dynamic_cast<graph::ValueData<gpu::ImageRef>*>(ctx.data(m_output)); out != nullptr)
	{
		(void)out->produce(gpu::ImageRef{
			.image = m_scene->image(), .view = m_scene->view(), .extent = m_extent, .format = m_color_format});
	}
	return true;
}
} // namespace veng::nodes
