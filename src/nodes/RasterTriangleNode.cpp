//
// Created by chris on 5/25/26.
//
// See RasterTriangleNode.hpp and design.md §L4.
//

#include <array>
#include <utility>
#include <veng/gpu/ImageRef.hpp>
#include <veng/managers/CommandManager.hpp>
#include <veng/nodes/RasterTriangleNode.hpp>
#include <veng/rendergraph/data/Data.hpp>

namespace veng::nodes
{
RasterTriangleNode::RasterTriangleNode(GraphicsPipeline pipeline, vk::Format color_format,
									   graph::DataHandle screen_size, graph::DataHandle output) noexcept
	: m_pipeline(std::move(pipeline))
	, m_color_format(color_format)
	, m_screen_size(screen_size)
	, m_output(output)
{
}

std::expected<bool, graph::ExecError> RasterTriangleNode::record(gpu::GpuExecContext& ctx)
{
	const auto* size = dynamic_cast<graph::ValueData<vk::Extent2D>*>(ctx.data(m_screen_size));
	if (size == nullptr)
	{
		return std::unexpected(graph::ExecError::MISSING_INPUT);
	}
	const vk::Extent2D extent = size->value();

	// (Re)create the persistent scene target from ScreenSize — a resize reallocates it.
	if (!m_scene.has_value() || m_scene_extent.width != extent.width || m_scene_extent.height != extent.height)
	{
		auto image = Image::create(ctx.allocator(), ctx.device(), extent, m_color_format,
								   vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc);
		if (!image.has_value())
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		m_scene		   = std::move(image.value());
		m_scene_extent = extent;
	}

	const vk::CommandBuffer cmd = ctx.command_buffer();

	// Undefined -> color attachment for this frame's render.
	CommandManager::image_barrier(
		cmd, m_scene->image(), vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eNone,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite);

	const auto clear	  = vk::ClearValue().setColor(vk::ClearColorValue(std::array{0.0F, 0.0F, 0.0F, 1.0F}));
	const auto attachment = vk::RenderingAttachmentInfo()
								.setImageView(m_scene->view())
								.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
								.setLoadOp(vk::AttachmentLoadOp::eClear)
								.setStoreOp(vk::AttachmentStoreOp::eStore)
								.setClearValue(clear);
	cmd.beginRendering(
		vk::RenderingInfo().setRenderArea(vk::Rect2D({0, 0}, extent)).setLayerCount(1).setColorAttachments(attachment));

	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline.pipeline());
	cmd.setViewport(
		0, vk::Viewport(0.0F, 0.0F, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0F, 1.0F));
	cmd.setScissor(0, vk::Rect2D({0, 0}, extent));
	cmd.draw(3, 1, 0, 0);
	cmd.endRendering();

	// Color attachment -> transfer source, ready for the present/blit (or a readback).
	CommandManager::image_barrier(
		cmd, m_scene->image(), vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);

	// Publish a ref to the rendered target (TRANSFER_SRC) on the scene edge for the blit.
	if (auto* out = dynamic_cast<graph::ValueData<gpu::ImageRef>*>(ctx.data(m_output)); out != nullptr)
	{
		(void)out->produce(gpu::ImageRef{
			.image = m_scene->image(), .view = m_scene->view(), .extent = m_scene_extent, .format = m_color_format});
	}
	return true;
}
} // namespace veng::nodes
