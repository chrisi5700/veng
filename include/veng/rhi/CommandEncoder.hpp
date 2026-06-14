/**
 * @file
 * @author chris
 * @brief The RHI command encoder — the mid-level recording surface a GPU node draws/copies into,
 *        in place of a raw `vk::CommandBuffer`.
 *
 * A node's `record` no longer receives a `vk::CommandBuffer`; it receives a `CommandEncoder` and
 * speaks in RHI vocabulary: it binds a @ref veng::rhi::PipelineHandle, sets the viewport from an
 * @ref veng::rhi::Extent2D, binds @ref veng::rhi::BufferHandle vertex/index buffers, draws, and
 * transitions a @ref veng::rhi::TextureHandle between @ref veng::rhi::TextureUsage states. The
 * encoder resolves handles to their Vulkan objects through the @ref veng::rhi::Device and is the
 * single place those `vk::cmd*` calls are made — so node/pass code names no Vulkan command at all.
 *
 * The encoder is a thin, non-owning view over the frame's command buffer (the @ref veng::Context
 * owns the buffer's lifetime); it is constructed per frame by @ref veng::gpu::GpuExecContext.
 *
 * @ingroup rhi
 */

#ifndef VENG_RHI_COMMANDENCODER_HPP
#define VENG_RHI_COMMANDENCODER_HPP

#include <array>
#include <cstdint>
#include <span>
#include <vector>
#include <veng/rhi/Convert.hpp>
#include <veng/rhi/Device.hpp>
#include <veng/rhi/Enums.hpp>
#include <veng/rhi/Handles.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::rhi
{
/// @brief One color target of a render pass, in RHI vocabulary. @see CommandEncoder::begin_rendering
struct ColorAttachment
{
	TextureHandle		 texture;							  ///< The target to render into.
	LoadOp				 load = LoadOp::CLEAR;				  ///< What to do with its existing contents.
	std::array<float, 4> clear_color{0.0F, 0.0F, 0.0F, 1.0F}; ///< Clear value when `load == CLEAR`.
};

/// @brief A dynamic-rendering pass description. @see CommandEncoder::begin_rendering
struct RenderPassDesc
{
	Extent2D						 area;				///< The render area (and viewport-sized region).
	std::span<const ColorAttachment> color_attachments; ///< Color targets, in attachment order.
};

/**
 * @brief Records GPU commands for one frame in RHI vocabulary, resolving handles via the @ref Device.
 *
 * Non-owning: holds the frame's `vk::CommandBuffer` (owned by the engine context) and a pointer to
 * the device registry. Cheap to copy/pass by value, but conventionally passed by reference.
 *
 * @ingroup rhi
 */
class CommandEncoder
{
	 public:
	/// @param command_buffer The frame's recording command buffer.
	/// @param device         The handle registry used to resolve pipelines/buffers/textures.
	CommandEncoder(vk::CommandBuffer command_buffer, Device& device) noexcept
		: m_cmd(command_buffer)
		, m_device(&device)
	{
	}

	// --- graphics: render pass + pipeline + dynamic state -----------------------------------

	/**
	 * @brief Begin a dynamic-rendering pass into @p desc's color targets.
	 *
	 * Each target must already be in @ref TextureUsage::COLOR_ATTACHMENT (transition it first). The
	 * encoder resolves each handle's view and builds the rendering info — no caller names a
	 * `vk::RenderingInfo` or image layout. Pair with @ref end_rendering.
	 */
	void begin_rendering(const RenderPassDesc& desc) const
	{
		std::vector<vk::RenderingAttachmentInfo> color;
		color.reserve(desc.color_attachments.size());
		for (const ColorAttachment& attachment : desc.color_attachments)
		{
			color.push_back(vk::RenderingAttachmentInfo()
								.setImageView(m_device->view(attachment.texture))
								.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
								.setLoadOp(to_vk(attachment.load))
								.setStoreOp(vk::AttachmentStoreOp::eStore)
								.setClearValue(vk::ClearValue().setColor(vk::ClearColorValue(attachment.clear_color))));
		}
		m_cmd.beginRendering(vk::RenderingInfo()
								 .setRenderArea(vk::Rect2D({0, 0}, to_vk(desc.area)))
								 .setLayerCount(1)
								 .setColorAttachments(color));
	}

	/// @brief Bind a graphics pipeline for subsequent draws.
	void bind_pipeline(PipelineHandle pipeline) const
	{
		m_cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_device->pipeline(pipeline));
	}

	/// @brief Set the (dynamic) viewport and scissor to cover the full @p extent.
	void set_viewport_scissor(Extent2D extent) const
	{
		m_cmd.setViewport(0, vk::Viewport(0.0F, 0.0F, static_cast<float>(extent.width),
										  static_cast<float>(extent.height), 0.0F, 1.0F));
		m_cmd.setScissor(0, vk::Rect2D({0, 0}, to_vk(extent)));
	}

	/// @brief Push a range of constant bytes into @p pipeline's layout at @p offset for @p stages.
	void push_constants(PipelineHandle pipeline, ShaderStage stages, std::uint32_t offset,
						std::span<const std::byte> bytes) const
	{
		m_cmd.pushConstants<std::byte>(
			m_device->pipeline_layout(pipeline), to_vk(stages), offset,
			vk::ArrayProxy<const std::byte>(static_cast<std::uint32_t>(bytes.size()), bytes.data()));
	}

	// --- graphics: resource binding ---------------------------------------------------------

	/// @brief Bind @p group as descriptor set 0 using @p pipeline's layout.
	void bind_descriptor_set(PipelineHandle pipeline, const BindGroup& group) const
	{
		m_cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_device->pipeline_layout(pipeline), 0, group.set,
								 {});
	}

	/// @brief Bind a vertex buffer at binding 0.
	void bind_vertex_buffer(BufferHandle buffer, vk::DeviceSize offset = 0) const
	{
		m_cmd.bindVertexBuffers(0, m_device->buffer(buffer), offset);
	}

	/// @brief Bind the index buffer for indexed draws.
	void bind_index_buffer(BufferHandle buffer, IndexType type, vk::DeviceSize offset = 0) const
	{
		m_cmd.bindIndexBuffer(m_device->buffer(buffer), offset, to_vk(type));
	}

	// --- graphics: draws --------------------------------------------------------------------

	/// @brief Non-indexed draw.
	void draw(std::uint32_t vertex_count, std::uint32_t instance_count, std::uint32_t first_vertex = 0,
			  std::uint32_t first_instance = 0) const
	{
		m_cmd.draw(vertex_count, instance_count, first_vertex, first_instance);
	}

	/// @brief Indexed draw.
	void draw_indexed(std::uint32_t index_count, std::uint32_t instance_count, std::uint32_t first_index = 0,
					  std::int32_t vertex_offset = 0, std::uint32_t first_instance = 0) const
	{
		m_cmd.drawIndexed(index_count, instance_count, first_index, vertex_offset, first_instance);
	}

	/// @brief End the current dynamic-rendering pass.
	void end_rendering() const { m_cmd.endRendering(); }

	// --- synchronization + transfer ---------------------------------------------------------

	/**
	 * @brief Insert an image barrier moving @p texture from one usage to another (color aspect).
	 *
	 * The source synchronization scope comes from @p from's state and the destination scope from
	 * @p to's state (see @ref veng::rhi::to_state). Use @ref TextureUsage::UNDEFINED as @p from to
	 * discard prior contents (e.g. a fresh swapchain image before a blit).
	 */
	void transition(TextureHandle texture, TextureUsage from, TextureUsage to) const
	{
		const ImageState src	 = to_state(from);
		const ImageState dst	 = to_state(to);
		const auto		 range	 = vk::ImageSubresourceRange()
									   .setAspectMask(vk::ImageAspectFlagBits::eColor)
									   .setLevelCount(1)
									   .setLayerCount(1);
		const auto		 barrier = vk::ImageMemoryBarrier2()
									   .setSrcStageMask(src.stage)
									   .setSrcAccessMask(src.access)
									   .setDstStageMask(dst.stage)
									   .setDstAccessMask(dst.access)
									   .setOldLayout(src.layout)
									   .setNewLayout(dst.layout)
									   .setImage(m_device->image(texture))
									   .setSubresourceRange(range);
		m_cmd.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(barrier));
	}

	/**
	 * @brief Blit @p src (in @ref TextureUsage::TRANSFER_SRC) into @p dst (in @ref TextureUsage::TRANSFER_DST).
	 *
	 * A blit (not a copy) absorbs any size/format mismatch between the two — which is what the
	 * scene-to-swapchain present path needs across a resize.
	 */
	void blit(TextureHandle src, Extent2D src_size, TextureHandle dst, Extent2D dst_size, Filter filter) const
	{
		const auto layers =
			vk::ImageSubresourceLayers().setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1);
		const auto region =
			vk::ImageBlit()
				.setSrcSubresource(layers)
				.setDstSubresource(layers)
				.setSrcOffsets({vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<std::int32_t>(src_size.width),
																	static_cast<std::int32_t>(src_size.height), 1}})
				.setDstOffsets({vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<std::int32_t>(dst_size.width),
																	static_cast<std::int32_t>(dst_size.height), 1}});
		m_cmd.blitImage(m_device->image(src), vk::ImageLayout::eTransferSrcOptimal, m_device->image(dst),
						vk::ImageLayout::eTransferDstOptimal, region, to_vk(filter));
	}

	/**
	 * @brief Copy @p src (in @ref TextureUsage::TRANSFER_SRC) into host-visible @p dst, then make the
	 *        write visible to a host read (the readback/screenshot path).
	 */
	void copy_texture_to_host_buffer(TextureHandle src, BufferHandle dst, Extent2D extent) const
	{
		const auto layers =
			vk::ImageSubresourceLayers().setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1);
		m_cmd.copyImageToBuffer(m_device->image(src), vk::ImageLayout::eTransferSrcOptimal, m_device->buffer(dst),
								vk::BufferImageCopy().setImageSubresource(layers).setImageExtent(
									vk::Extent3D{extent.width, extent.height, 1}));
		m_cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {},
							  vk::MemoryBarrier()
								  .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
								  .setDstAccessMask(vk::AccessFlagBits::eHostRead),
							  {}, {});
	}

	/**
	 * @brief Upload a host buffer into @p dst's mip 0 and generate its full mip chain, leaving every
	 *        level in @ref TextureUsage::SAMPLED (shader-read).
	 *
	 * @p dst must have been created with `SAMPLED | TRANSFER_SRC | TRANSFER_DST` and @p levels mips.
	 * This encapsulates the whole texture-upload sequence — the all-levels layout transition, the
	 * buffer→image copy into level 0, and the blit-down pyramid — so an asset loader (the @ref
	 * veng::assets::Texture path) names no Vulkan. Requires the format to support a linear blit, which
	 * R8G8B8A8 UNORM/SRGB universally do.
	 */
	void upload_mipped_texture(BufferHandle src, TextureHandle dst, Extent2D extent, std::uint32_t levels) const
	{
		const vk::Image		image	   = m_device->image(dst);
		const std::uint32_t mip_levels = levels == 0 ? 1 : levels;

		// Per-level color-aspect barrier (the encoder is the seam, so it speaks raw vk here).
		const auto barrier = [&](vk::ImageLayout old_layout, vk::ImageLayout new_layout,
								 vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
								 vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access,
								 std::uint32_t base_level, std::uint32_t level_count)
		{
			const auto range		 = vk::ImageSubresourceRange()
										   .setAspectMask(vk::ImageAspectFlagBits::eColor)
										   .setBaseMipLevel(base_level)
										   .setLevelCount(level_count)
										   .setLayerCount(1);
			const auto image_barrier = vk::ImageMemoryBarrier2()
										   .setSrcStageMask(src_stage)
										   .setSrcAccessMask(src_access)
										   .setDstStageMask(dst_stage)
										   .setDstAccessMask(dst_access)
										   .setOldLayout(old_layout)
										   .setNewLayout(new_layout)
										   .setImage(image)
										   .setSubresourceRange(range);
			m_cmd.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(image_barrier));
		};

		// Whole image: UNDEFINED -> TRANSFER_DST (all levels), then upload level 0.
		barrier(vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
				vk::PipelineStageFlagBits2::eTopOfPipe, vk::AccessFlagBits2::eNone,
				vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, 0, mip_levels);

		const auto copy = vk::BufferImageCopy()
							  .setImageSubresource(
								  vk::ImageSubresourceLayers().setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1))
							  .setImageExtent(vk::Extent3D{extent.width, extent.height, 1});
		m_cmd.copyBufferToImage(m_device->buffer(src), image, vk::ImageLayout::eTransferDstOptimal, copy);

		// Blit-down pyramid: each level is half its predecessor, which becomes shader-read once consumed.
		auto	   mip_width  = static_cast<std::int32_t>(extent.width);
		auto	   mip_height = static_cast<std::int32_t>(extent.height);
		const auto layers	  = [](std::uint32_t level)
		{
			return vk::ImageSubresourceLayers()
				.setAspectMask(vk::ImageAspectFlagBits::eColor)
				.setMipLevel(level)
				.setLayerCount(1);
		};
		for (std::uint32_t level = 1; level < mip_levels; ++level)
		{
			barrier(vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal,
					vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
					vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead, level - 1, 1);

			const std::int32_t next_width  = mip_width > 1 ? mip_width / 2 : 1;
			const std::int32_t next_height = mip_height > 1 ? mip_height / 2 : 1;
			m_cmd.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal,
							vk::ImageBlit()
								.setSrcSubresource(layers(level - 1))
								.setSrcOffsets({vk::Offset3D{0, 0, 0}, vk::Offset3D{mip_width, mip_height, 1}})
								.setDstSubresource(layers(level))
								.setDstOffsets({vk::Offset3D{0, 0, 0}, vk::Offset3D{next_width, next_height, 1}}),
							vk::Filter::eLinear);

			barrier(vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
					vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
					vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead, level - 1, 1);

			mip_width  = next_width;
			mip_height = next_height;
		}

		// The last level was only ever a blit destination: TRANSFER_DST -> SHADER_READ_ONLY.
		barrier(vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
				vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
				vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead, mip_levels - 1, 1);
	}

	/// @brief The underlying command buffer — a migration escape hatch for the not-yet-ported
	///        RenderTargetSet begin path; do not reach for this in new node code.
	[[nodiscard]] vk::CommandBuffer vk() const noexcept { return m_cmd; }

	/// @brief The handle registry, for the rare node that must resolve a handle outside the encoder API.
	[[nodiscard]] Device& device() const noexcept { return *m_device; }

	 private:
	vk::CommandBuffer m_cmd;
	Device*			  m_device;
};
} // namespace veng::rhi

#endif // VENG_RHI_COMMANDENCODER_HPP
