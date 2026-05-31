/**
 * @file
 * @author chris
 * @brief Implementation of @ref veng::nodes::ScreenshotNode.
 * @ingroup graph_nodes
 */

#include <cstdint>
#include <cstdio>
#include <utility>
#include <veng/gpu/ImageRef.hpp>
#include <veng/nodes/ScreenshotNode.hpp>
#include <veng/rendergraph/data/Data.hpp>

namespace veng::nodes
{
ScreenshotNode::ScreenshotNode(graph::DataHandle source, graph::DataHandle output, std::string path) noexcept
	: m_input(source)
	, m_output(output)
	, m_path(std::move(path))
{
}

std::vector<gpu::ImageUsage> ScreenshotNode::image_usages(graph::ExecContext& ctx)
{
	// Read-side dependency for the executor: copyImageToBuffer needs the source in TRANSFER_SRC.
	const auto* src = dynamic_cast<graph::ValueData<gpu::ImageRef>*>(ctx.data(m_input));
	if (src == nullptr || src->value().pool_id == gpu::ImageRef::INVALID_POOL_ID)
	{
		return {};
	}
	return {gpu::ImageUsage{.id		= src->value().pool_id,
							.layout = vk::ImageLayout::eTransferSrcOptimal,
							.stage	= vk::PipelineStageFlagBits2::eTransfer,
							.access = vk::AccessFlagBits2::eTransferRead}};
}

std::expected<bool, graph::ExecError> ScreenshotNode::record(gpu::GpuExecContext& ctx)
{
	const auto* src = dynamic_cast<graph::ValueData<gpu::ImageRef>*>(ctx.data(m_input));
	if (src == nullptr)
	{
		return std::unexpected(graph::ExecError::MISSING_INPUT);
	}
	const gpu::ImageRef image = src->value();
	if (!image.image)
	{
		return std::unexpected(graph::ExecError::NODE_FAILED);
	}

	// RGBA8 staging: the demo formats are 4 bytes/pixel; on_retired writes RGB to the PPM and
	// drops alpha. (A more general node would inspect the format and adapt the conversion.)
	constexpr std::size_t bytes_per_pixel = 4;
	const vk::DeviceSize size = static_cast<vk::DeviceSize>(image.extent.width) * image.extent.height * bytes_per_pixel;
	if (!m_staging.has_value() || m_staging->size() < size)
	{
		auto buf =
			Buffer::create(ctx.allocator(), size, vk::BufferUsageFlagBits::eTransferDst, vma::MemoryUsage::eAuto,
						   vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessRandom);
		if (!buf.has_value() || buf->mapped() == nullptr)
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		m_staging = std::move(buf.value());
	}
	m_extent = image.extent;

	// Retain the pool copy we read while this frame is in flight (the producer may be cached).
	ctx.pool().consume(image);

	const vk::CommandBuffer cmd = ctx.command_buffer();
	const auto layers = vk::ImageSubresourceLayers().setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1);
	cmd.copyImageToBuffer(image.image, vk::ImageLayout::eTransferSrcOptimal, m_staging->buffer(),
						  vk::BufferImageCopy().setImageSubresource(layers).setImageExtent(
							  vk::Extent3D{image.extent.width, image.extent.height, 1}));
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {},
						vk::MemoryBarrier()
							.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
							.setDstAccessMask(vk::AccessFlagBits::eHostRead),
						{}, {});
	m_pending_write = true;
	return true;
}

void ScreenshotNode::on_retired(gpu::SubmitContext& /*ctx*/) noexcept
{
	// The slot's fence has signalled by the time the driver calls this, so the staging buffer
	// is fully populated. Write a PPM (P6 binary RGB), dropping alpha. Errors are swallowed —
	// a failed file write is logged through capture_count not advancing.
	if (!m_pending_write || !m_staging.has_value())
	{
		return;
	}
	m_pending_write = false;
	std::FILE* file = std::fopen(m_path.c_str(), "wb");
	if (file == nullptr)
	{
		return;
	}
	(void)std::fprintf(file, "P6\n%u %u\n255\n", m_extent.width, m_extent.height);
	const auto* pixels = static_cast<const std::uint8_t*>(m_staging->mapped());
	for (std::uint32_t i = 0; i < m_extent.width * m_extent.height; ++i)
	{
		(void)std::fputc(static_cast<int>(pixels[(i * 4) + 0]), file);
		(void)std::fputc(static_cast<int>(pixels[(i * 4) + 1]), file);
		(void)std::fputc(static_cast<int>(pixels[(i * 4) + 2]), file);
	}
	(void)std::fclose(file);
	++m_capture_count;
}
} // namespace veng::nodes
