/**
 * @file
 * @author chris
 * @brief L4 graph node that blits one image into another via `vkCmdBlitImage`.
 *
 * Reads two image edges, `src` and `dst` (both `ImageData` = `ValueData<ImageRef>`), and
 * records a `vkCmdBlitImage` of src -> dst, absorbing any size or format difference using a
 * linear filter. The node knows nothing about the swapchain: "present" is simply a blit whose
 * `dst` happens to be the acquired swapchain image and whose `final_layout` is
 * `ePresentSrcKHR`. Its output carries the `dst` ref forward so the next consumer (e.g.
 * @ref veng::nodes::PresentNode) is ordered after it and sees the now-written image.
 *
 * `src` is expected in `TRANSFER_SRC` layout (the convention a raster node leaves its target
 * in); `dst` is overwritten in full, so its prior contents and layout are discarded.
 *
 * @ingroup graph_nodes
 */

#ifndef VENG_BLITNODE_HPP
#define VENG_BLITNODE_HPP

#include <array>
#include <cstddef>
#include <expected>
#include <span>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/gpu/VersionedOutput.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <veng/rhi/Enums.hpp>

namespace veng::nodes
{
/**
 * @brief Concrete @ref veng::gpu::GpuNode that records a `vkCmdBlitImage` each frame.
 *
 * @ingroup graph_nodes
 * @see PresentNode
 * @see ScreenshotNode
 */
class BlitNode final : public gpu::GpuNode
{
	 public:
	/**
	 * @brief Construct a blit node.
	 *
	 * `src` and `dst` are the source and destination `ImageData` edges. `output` carries the
	 * written `dst` ref forward to the next consumer. `final_layout` is the layout `dst` is
	 * left in: use `ePresentSrcKHR` when the next step is presentation, or
	 * `eTransferSrcOptimal` for a readback or a further blit (the default).
	 *
	 * @param src         Data handle for the source @ref veng::gpu::ImageRef edge.
	 * @param dst         Data handle for the destination @ref veng::gpu::ImageRef edge.
	 * @param output      Data handle into which the written `dst` ref is forwarded.
	 * @param final_usage Usage to leave `dst` in after the blit completes (`PRESENT` for the
	 *                    present path, `TRANSFER_SRC` for a readback or further blit — the default).
	 */
	BlitNode(graph::DataHandle src, graph::DataHandle dst, graph::DataHandle output,
			 rhi::TextureUsage final_usage = rhi::TextureUsage::TRANSFER_SRC) noexcept;

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return m_inputs; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	/**
	 * @brief How many times this node has actually recorded — the lens for the caching proof.
	 * @return The total number of `record` invocations.
	 */
	[[nodiscard]] std::size_t record_count() const noexcept { return m_record_count; }

	 protected:
	[[nodiscard]] std::expected<bool, graph::ExecError> record(gpu::GpuExecContext& ctx) override;
	std::vector<gpu::ImageUsage>						image_usages(graph::ExecContext& ctx) override;

	 private:
	std::array<graph::DataHandle, 2> m_inputs; ///< [src, dst]
	graph::DataHandle				 m_output;
	rhi::TextureUsage				 m_final_usage;
	std::size_t						 m_record_count = 0;
	gpu::VersionedOutput m_versioned; ///< Owns the per-produce version bump for the forwarded @ref veng::gpu::ImageRef.
};
} // namespace veng::nodes

#endif // VENG_BLITNODE_HPP
