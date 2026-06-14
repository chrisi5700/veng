/**
 * @file
 * @author chris
 * @brief @ref veng::RenderTargetSet + @ref veng::clamp_sample_count implementation.
 * @ingroup resources
 */

#include <array>
#include <veng/context/Context.hpp>
#include <veng/resources/RenderTargetSet.hpp>

namespace veng
{
vk::SampleCountFlagBits clamp_sample_count(const Context& ctx, vk::SampleCountFlagBits requested) noexcept
{
	const vk::PhysicalDeviceLimits limits = ctx.physical_device().getProperties().limits;
	const vk::SampleCountFlags supported  = limits.framebufferColorSampleCounts & limits.framebufferDepthSampleCounts;
	// Walk down from the request to the first power-of-two count the device offers for both
	// color and depth framebuffers; e1 is always supported, so the loop terminates.
	for (const vk::SampleCountFlagBits bit :
		 {vk::SampleCountFlagBits::e64, vk::SampleCountFlagBits::e32, vk::SampleCountFlagBits::e16,
		  vk::SampleCountFlagBits::e8, vk::SampleCountFlagBits::e4, vk::SampleCountFlagBits::e2})
	{
		if (static_cast<unsigned>(requested) >= static_cast<unsigned>(bit) && (supported & bit))
		{
			return bit;
		}
	}
	return vk::SampleCountFlagBits::e1;
}

void RenderTargetSet::configure(vk::Format color_format, vk::Format depth_format,
								vk::SampleCountFlagBits samples) noexcept
{
	m_color_format = color_format;
	m_depth_format = depth_format;
	m_samples	   = samples;
}

std::expected<void, vk::Result> RenderTargetSet::acquire(ResourcePool& pool, vk::Extent2D extent)
{
	// Declare the logical resources once. The single-sample color image is always the one a
	// consumer reads (sampled / transfer-src); under MSAA it is the resolve target, and a separate
	// multisampled attachment is what the pass actually renders into. Depth matches the color
	// attachment's sample count (Vulkan requires all attachments in a pass share a sample count).
	if (!m_declared)
	{
		m_color_id = pool.declare_image(m_color_format,
										vk::ImageUsageFlagBits::eColorAttachment |
											vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled,
										vk::ImageAspectFlagBits::eColor, vk::SampleCountFlagBits::e1);
		if (multisampled())
		{
			m_msaa_id = pool.declare_image(m_color_format, vk::ImageUsageFlagBits::eColorAttachment,
										   vk::ImageAspectFlagBits::eColor, m_samples);
		}
		if (has_depth())
		{
			m_depth_id = pool.declare_image(m_depth_format, vk::ImageUsageFlagBits::eDepthStencilAttachment,
											vk::ImageAspectFlagBits::eDepth, m_samples);
		}
		m_declared = true;
	}

	auto color = pool.acquire_image(m_color_id, extent);
	if (!color.has_value())
	{
		return std::unexpected(color.error());
	}
	m_color = color.value();

	if (multisampled())
	{
		auto msaa = pool.acquire_image(m_msaa_id, extent);
		if (!msaa.has_value())
		{
			return std::unexpected(msaa.error());
		}
		m_msaa = msaa.value();
	}

	if (has_depth())
	{
		auto depth = pool.acquire_image(m_depth_id, extent);
		if (!depth.has_value())
		{
			return std::unexpected(depth.error());
		}
		m_depth = depth.value();
	}
	return {};
}

void RenderTargetSet::begin(ResourcePool& pool, rhi::CommandEncoder& enc, vk::Extent2D extent,
							std::array<float, 4> clear_color)
{
	// RenderTargetSet is the engine's MSAA/attachment plumbing — legitimately Vulkan-aware. It records
	// the attachment transitions + beginRendering through the encoder's underlying command buffer.
	const vk::CommandBuffer cmd	 = enc.vk();
	const bool				msaa = multisampled();

	// Auto-tracked transitions: the pool inserts the barrier and updates its layout record so the
	// consumer side can reason about layout without desyncing from actual GPU state. Under MSAA both
	// the multisampled attachment and the resolve target render as color attachments.
	if (msaa)
	{
		pool.transition_image(m_msaa_id, cmd, vk::ImageLayout::eColorAttachmentOptimal,
							  vk::PipelineStageFlagBits2::eColorAttachmentOutput,
							  vk::AccessFlagBits2::eColorAttachmentWrite);
	}
	pool.transition_image(m_color_id, cmd, vk::ImageLayout::eColorAttachmentOptimal,
						  vk::PipelineStageFlagBits2::eColorAttachmentOutput,
						  vk::AccessFlagBits2::eColorAttachmentWrite);
	if (has_depth())
	{
		pool.transition_image(m_depth_id, cmd, vk::ImageLayout::eDepthStencilAttachmentOptimal,
							  vk::PipelineStageFlagBits2::eEarlyFragmentTests,
							  vk::AccessFlagBits2::eDepthStencilAttachmentWrite);
	}

	const auto color_clear	= vk::ClearValue().setColor(vk::ClearColorValue(clear_color));
	auto	   color_attach = vk::RenderingAttachmentInfo()
								  .setImageView(msaa ? m_msaa->view() : m_color->view())
								  .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
								  .setLoadOp(vk::AttachmentLoadOp::eClear)
								  // The multisampled image is consumed by the resolve, never read after; only the
								  // resolved single-sample copy is kept (eStore on the resolve via eDontCare here).
								  .setStoreOp(msaa ? vk::AttachmentStoreOp::eDontCare : vk::AttachmentStoreOp::eStore)
								  .setClearValue(color_clear);
	if (msaa)
	{
		color_attach.setResolveMode(vk::ResolveModeFlagBits::eAverage)
			.setResolveImageView(m_color->view())
			.setResolveImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
	}

	const auto depth_clear	= vk::ClearValue().setDepthStencil(vk::ClearDepthStencilValue(1.0F, 0));
	const auto depth_attach = vk::RenderingAttachmentInfo()
								  .setImageView(has_depth() ? m_depth->view() : vk::ImageView{})
								  .setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
								  .setLoadOp(vk::AttachmentLoadOp::eClear)
								  .setStoreOp(vk::AttachmentStoreOp::eDontCare)
								  .setClearValue(depth_clear);

	auto rendering = vk::RenderingInfo()
						 .setRenderArea(vk::Rect2D({0, 0}, extent))
						 .setLayerCount(1)
						 .setColorAttachments(color_attach);
	if (has_depth())
	{
		rendering.setPDepthAttachment(&depth_attach);
	}
	cmd.beginRendering(rendering);
}
} // namespace veng
