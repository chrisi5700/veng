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

#include <cstdint>
#include <span>
#include <veng/rhi/Convert.hpp>
#include <veng/rhi/Device.hpp>
#include <veng/rhi/Enums.hpp>
#include <veng/rhi/Handles.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::rhi
{
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

	// --- graphics: pipeline + dynamic state -------------------------------------------------

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
