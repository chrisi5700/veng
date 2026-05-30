//
// L4 sink node — a screenshot is a peer sink ([[pass-draw-redesign]]). Demand the screenshot
// alongside present (e.g. `resolve({frame_done, shot_done})`) the frame you want a capture; the
// node records a copy from its input image into a host-visible staging buffer (recordable, runs
// in the shared CB like any other node), and writes the file to disk in `on_retired` — the
// post-fence hook that fires after the slot's in-flight fence has signalled, so the staging
// buffer is guaranteed populated. The driver never special-cases screenshots; the same
// mechanism would carry a video-encode sink.
//
// File format: PPM (P6 binary RGB), no library dependency. Channels: assumes RGBA8 input; RGB is
// written, alpha dropped.
//

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
#include <vulkan/vulkan.hpp>

namespace veng::nodes
{
class ScreenshotNode final : public gpu::GpuNode, public gpu::Sink
{
	 public:
	/// Capture the image on `source` (a `ValueData<gpu::ImageRef>` left in TRANSFER_SRC by its
	/// producer) and write it to `path` on the host once the frame's fence signals. `output` is
	/// the done-token sink the driver demands.
	ScreenshotNode(graph::DataHandle source, graph::DataHandle output, std::string path) noexcept;

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return {&m_input, 1}; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	/// Number of files this node has written (test lens).
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
	vk::Extent2D		  m_extent{};
	bool				  m_pending_write = false;
	std::size_t			  m_capture_count = 0;
};
} // namespace veng::nodes

#endif // VENG_SCREENSHOTNODE_HPP
