/**
 * @file
 * @author chris
 * @brief Host-interop helper: blit a graph-produced image into a host-owned `VkImage`.
 *
 * The embedding seam (see @ref veng::Context::adopt). A host that owns its own swapchain — a Qt
 * `QVulkanWindow`, or veng's own GLFW front-ends — renders a veng frame into an offscreen target and
 * must copy it into the image it is about to present. That copy is a fixed sequence of layout
 * transitions plus a `vkCmdBlitImage`; @ref veng::gpu::blit_into_host_image records it so a consumer
 * never hand-writes the barriers. This is the one place a `vk::Image` legitimately crosses veng's
 * surface — it is the host-interop boundary, where the host inherently speaks Vulkan.
 *
 * @ingroup gpu_handles
 */

#ifndef VENG_HOSTBLIT_HPP
#define VENG_HOSTBLIT_HPP

#include <array>
#include <cstdint>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::gpu
{
/**
 * @brief Record a blit of a graph-produced image into a host-owned destination image.
 *
 * Into this frame's command buffer (from @p ctx) this records, in order:
 *   1. a pool-aware transition of @p source to `TransferSrcOptimal` (the producing pass leaves it in
 *      colour-attachment layout);
 *   2. a transition of @p dst from `Undefined` to `TransferDstOptimal` (its prior contents are
 *      discarded — the blit overwrites the whole image);
 *   3. a linear-filtered `vkCmdBlitImage` from @p source to @p dst (scaling between their extents);
 *   4. a transition of @p dst to @p dst_final_layout, ready for the host to present or submit.
 *
 * The host still owns image acquisition, queue submission, and synchronization; this only records
 * the copy. Call it after `Graph::execute` has produced @p source, on the same command buffer.
 *
 * @param ctx              The GPU execution context for this frame (its command buffer is recorded into).
 * @param source           The graph-produced image to copy from (must be a pool-owned @ref ImageRef).
 * @param dst              The host-owned destination image (e.g. the acquired swapchain image).
 * @param dst_extent       Pixel dimensions of @p dst (the blit scales @p source to fit).
 * @param dst_final_layout Layout to leave @p dst in; defaults to `PresentSrcKHR` for direct present.
 */
inline void blit_into_host_image(GpuExecContext& ctx, const ImageRef& source, vk::Image dst, vk::Extent2D dst_extent,
								 vk::ImageLayout dst_final_layout = vk::ImageLayout::ePresentSrcKHR)
{
	const vk::CommandBuffer cmd = ctx.encoder().vk();
	const vk::Image			src = ctx.rhi().image(source.texture);

	// (1) Source -> transfer-src, through the pool so its tracked layout/stage stay correct.
	ctx.pool().transition_image(source.pool_id, cmd, vk::ImageLayout::eTransferSrcOptimal,
								vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);

	const auto range =
		vk::ImageSubresourceRange().setAspectMask(vk::ImageAspectFlagBits::eColor).setLevelCount(1).setLayerCount(1);
	const auto layers = vk::ImageSubresourceLayers().setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1);

	// (2) Destination undefined -> transfer-dst (we overwrite all of it). The barrier is a named lvalue
	// because setImageMemoryBarriers takes an ArrayProxyNoTemporaries (it rejects a temporary).
	const auto to_transfer_dst = vk::ImageMemoryBarrier2()
									 .setSrcStageMask(vk::PipelineStageFlagBits2::eTopOfPipe)
									 .setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
									 .setDstAccessMask(vk::AccessFlagBits2::eTransferWrite)
									 .setOldLayout(vk::ImageLayout::eUndefined)
									 .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
									 .setImage(dst)
									 .setSubresourceRange(range);
	cmd.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(to_transfer_dst));

	// (3) Linear blit, scaling source -> destination extent.
	const std::array src_offsets{vk::Offset3D{0, 0, 0},
								 vk::Offset3D{static_cast<std::int32_t>(source.extent.width),
											  static_cast<std::int32_t>(source.extent.height), 1}};
	const std::array dst_offsets{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<std::int32_t>(dst_extent.width),
																	 static_cast<std::int32_t>(dst_extent.height), 1}};
	const auto		 region = vk::ImageBlit2()
								  .setSrcOffsets(src_offsets)
								  .setDstOffsets(dst_offsets)
								  .setSrcSubresource(layers)
								  .setDstSubresource(layers);
	cmd.blitImage2(vk::BlitImageInfo2()
					   .setSrcImage(src)
					   .setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal)
					   .setDstImage(dst)
					   .setDstImageLayout(vk::ImageLayout::eTransferDstOptimal)
					   .setRegions(region)
					   .setFilter(vk::Filter::eLinear));

	// (4) Destination transfer-dst -> final layout for the host to present/submit.
	const auto to_final = vk::ImageMemoryBarrier2()
							  .setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
							  .setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite)
							  .setDstStageMask(vk::PipelineStageFlagBits2::eBottomOfPipe)
							  .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
							  .setNewLayout(dst_final_layout)
							  .setImage(dst)
							  .setSubresourceRange(range);
	cmd.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(to_final));
}
} // namespace veng::gpu

#endif // VENG_HOSTBLIT_HPP
