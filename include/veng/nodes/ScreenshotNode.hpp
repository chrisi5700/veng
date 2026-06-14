/**
 * @file
 * @author chris
 * @brief L4 sink node that copies a rendered image to a host-visible buffer and writes a PPM file.
 *
 * A screenshot is a peer sink: demand it alongside present (e.g. `resolve({frame_done,
 * shot_done})`) the frame you want a capture. The node records a `vkCmdCopyImageToBuffer`
 * from its input image into a host-visible staging buffer in the shared command buffer, then
 * writes the file to disk in `on_retired` — the post-fence hook that fires after the slot's
 * in-flight fence has signalled, guaranteeing the staging buffer is fully populated. The driver
 * never special-cases screenshots; the same mechanism would carry a video-encode sink.
 *
 * File format: PPM (P6 binary RGB), no external library required. Input is assumed to be RGBA8;
 * RGB is written and the alpha channel is dropped.
 *
 * @ingroup graph_nodes
 * @see PresentNode
 * @see BlitNode
 */

#ifndef VENG_SCREENSHOTNODE_HPP
#define VENG_SCREENSHOTNODE_HPP

#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/gpu/Sink.hpp>
#include <veng/gpu/SubmitContext.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/rhi/Enums.hpp>

namespace veng::nodes
{
/**
 * @brief Concrete @ref veng::gpu::GpuNode and @ref veng::gpu::Sink that reads back a rendered image and writes a PPM
 * file.
 *
 * @ingroup graph_nodes
 * @see PresentNode
 * @see BlitNode
 */
class ScreenshotNode final : public gpu::GpuNode, public gpu::Sink
{
	 public:
	/**
	 * @brief Construct a screenshot node.
	 *
	 * Captures the image on `source` (a `ValueData<gpu::ImageRef>` left in `TRANSFER_SRC` by
	 * its producer) and writes it to `path` on the host once the frame's fence signals.
	 * `output` is the done-token sink the driver demands.
	 *
	 * @param source  Data handle for the source @ref veng::gpu::ImageRef edge (must be in `TRANSFER_SRC`).
	 * @param output  Data handle for the done-token sink.
	 * @param path    Filesystem path at which the PPM file will be written.
	 */
	ScreenshotNode(graph::DataHandle source, graph::DataHandle output, std::string path) noexcept;

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return {&m_input, 1}; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	/**
	 * @brief Number of PPM files this node has successfully written.
	 *
	 * Useful as a test lens: a value that does not advance after a capture request indicates
	 * a failed file write.
	 *
	 * @return Total number of captured frames written to disk.
	 */
	[[nodiscard]] std::size_t capture_count() const noexcept { return m_capture_count; }

	 protected:
	[[nodiscard]] std::expected<bool, graph::ExecError> record(gpu::GpuExecContext& ctx) override;
	void												on_retired(gpu::SubmitContext& ctx) noexcept override;
	std::vector<gpu::ImageUsage>						image_usages(graph::ExecContext& ctx) override;

	 private:
	graph::DataHandle	  m_input;
	graph::DataHandle	  m_output;
	std::string			  m_path;
	std::optional<Buffer> m_staging;
	rhi::Extent2D		  m_extent{};
	bool				  m_pending_write = false;
	std::size_t			  m_capture_count = 0;
};
} // namespace veng::nodes

#endif // VENG_SCREENSHOTNODE_HPP
