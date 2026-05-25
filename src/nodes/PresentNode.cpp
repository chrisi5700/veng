//
// Created by chris on 5/25/26.
//
// See PresentNode.hpp and design.md §L4.
//

#include <veng/managers/CommandManager.hpp>
#include <veng/nodes/PresentNode.hpp>

namespace veng::nodes
{
PresentNode::PresentNode(const RasterTriangleNode& scene, Image& target, graph::DataHandle scene_token,
						 graph::DataHandle swapchain_source, graph::DataHandle output) noexcept
	: m_scene(&scene)
	, m_target(&target)
	, m_inputs{scene_token, swapchain_source}
	, m_output(output)
{
}

std::expected<bool, graph::ExecError> PresentNode::record(gpu::GpuExecContext& ctx)
{
	// Read the scene target fresh: a resize gives the raster node a new Image.
	const Image* scene = m_scene->scene();
	if (scene == nullptr)
	{
		return std::unexpected(graph::ExecError::NODE_FAILED);
	}

	const vk::CommandBuffer cmd	   = ctx.command_buffer();
	const vk::Extent2D		extent = scene->extent();

	// Acquired target: undefined -> transfer dst (we overwrite the whole image).
	CommandManager::image_barrier(cmd, m_target->image(), vk::ImageLayout::eUndefined,
								  vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eTopOfPipe,
								  vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eTransfer,
								  vk::AccessFlagBits2::eTransferWrite);

	// Copy the cached scene (already TRANSFER_SRC from the raster node) into it. Same
	// size/format here, so a copy suffices; a real swapchain of differing size/format
	// would use vkCmdBlitImage.
	const auto layers = vk::ImageSubresourceLayers().setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1);
	const auto region = vk::ImageCopy().setSrcSubresource(layers).setDstSubresource(layers).setExtent(
		vk::Extent3D{extent.width, extent.height, 1});
	cmd.copyImage(scene->image(), vk::ImageLayout::eTransferSrcOptimal, m_target->image(),
				  vk::ImageLayout::eTransferDstOptimal, region);

	// Leave the target in TRANSFER_SRC so the test (or a real present) can read/show it.
	CommandManager::image_barrier(cmd, m_target->image(), vk::ImageLayout::eTransferDstOptimal,
								  vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eTransfer,
								  vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer,
								  vk::AccessFlagBits2::eTransferRead);

	++m_record_count;
	return true;
}
} // namespace veng::nodes
