/**
 * @file
 * @author chris
 * @brief @ref veng::RenderTargetSet + @ref veng::clamp_sample_count implementation.
 * @ingroup resources
 */

#include <array>
#include <veng/context/Context.hpp>
#include <veng/resources/RenderTargetSet.hpp>
#include <veng/rhi/Convert.hpp>

namespace veng
{
rhi::SampleCount clamp_sample_count(const Context& ctx, rhi::SampleCount requested) noexcept
{
	const vk::SampleCountFlagBits   req		= rhi::to_vk(requested);
	const vk::PhysicalDeviceLimits limits	= ctx.physical_device().getProperties().limits;
	const vk::SampleCountFlags	   supported = limits.framebufferColorSampleCounts & limits.framebufferDepthSampleCounts;
	// Walk down from the request to the first power-of-two count the device offers for both
	// color and depth framebuffers; e1 is always supported, so the loop terminates.
	for (const vk::SampleCountFlagBits bit :
		 {vk::SampleCountFlagBits::e64, vk::SampleCountFlagBits::e32, vk::SampleCountFlagBits::e16,
		  vk::SampleCountFlagBits::e8, vk::SampleCountFlagBits::e4, vk::SampleCountFlagBits::e2})
	{
		if (static_cast<unsigned>(req) >= static_cast<unsigned>(bit) && (supported & bit))
		{
			return rhi::to_rhi(bit);
		}
	}
	return rhi::SampleCount::X1;
}

void RenderTargetSet::configure(rhi::Format color_format, rhi::Format depth_format, rhi::SampleCount samples) noexcept
{
	m_color_format = rhi::to_vk(color_format);
	m_depth_format = rhi::to_vk(depth_format);
	m_samples	   = rhi::to_vk(samples);
}

std::expected<void, vk::Result> RenderTargetSet::acquire(ResourcePool& pool, rhi::Extent2D extent)
{
	const vk::Extent2D vk_extent = rhi::to_vk(extent);
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

	auto color = pool.acquire_image(m_color_id, vk_extent);
	if (!color.has_value())
	{
		return std::unexpected(color.error());
	}
	m_color = color.value();

	if (multisampled())
	{
		auto msaa = pool.acquire_image(m_msaa_id, vk_extent);
		if (!msaa.has_value())
		{
			return std::unexpected(msaa.error());
		}
		m_msaa = msaa.value();
	}

	if (has_depth())
	{
		auto depth = pool.acquire_image(m_depth_id, vk_extent);
		if (!depth.has_value())
		{
			return std::unexpected(depth.error());
		}
		m_depth = depth.value();
	}
	return {};
}

void RenderTargetSet::begin(ResourcePool& pool, rhi::CommandEncoder& enc, rhi::Extent2D extent,
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
						 .setRenderArea(vk::Rect2D({0, 0}, rhi::to_vk(extent)))
						 .setLayerCount(1)
						 .setColorAttachments(color_attach);
	if (has_depth())
	{
		rendering.setPDepthAttachment(&depth_attach);
	}
	cmd.beginRendering(rendering);
}
} // namespace veng
