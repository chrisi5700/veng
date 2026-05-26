//
// Created by chris on 5/26/26.
//
// L4 node — a generic image blit (design.md §L4). Reads two image edges, `src` and
// `dst` (both `ImageData` = `ValueData<ImageRef>`), and records a `vkCmdBlitImage` of
// src -> dst, absorbing any size/format difference (linear filter). It knows nothing
// about the swapchain: "present" is just a blit whose `dst` happens to be the acquired
// swapchain image and whose `final_layout` is `ePresentSrcKHR`. Its output carries the
// `dst` ref forward so the next consumer (e.g. PresentNode) is ordered after it and sees
// the now-written image.
//
// `src` is expected in TRANSFER_SRC layout (the convention a raster node leaves its
// target in); `dst` is overwritten in full, so its prior contents/layout are discarded.
//

#ifndef VENG_BLITNODE_HPP
#define VENG_BLITNODE_HPP

#include <array>
#include <cstddef>
#include <expected>
#include <span>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::nodes
{
class BlitNode final : public gpu::GpuNode
{
	 public:
	/// `src`/`dst` are the source and destination `ImageData` edges; `output` carries the
	/// written `dst` ref forward. `final_layout` is the layout `dst` is left in —
	/// `ePresentSrcKHR` when the next step is presentation, or `eTransferSrcOptimal` for a
	/// readback (the default; also the layout a blit-into-blit chain expects).
	BlitNode(graph::DataHandle src, graph::DataHandle dst, graph::DataHandle output,
			 vk::ImageLayout final_layout = vk::ImageLayout::eTransferSrcOptimal) noexcept;

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return m_inputs; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	/// How many times this node has actually recorded — the lens for the caching proof.
	[[nodiscard]] std::size_t record_count() const noexcept { return m_record_count; }

	 protected:
	[[nodiscard]] std::expected<bool, graph::ExecError> record(gpu::GpuExecContext& ctx) override;

	 private:
	std::array<graph::DataHandle, 2> m_inputs; // [src, dst]
	graph::DataHandle				 m_output;
	vk::ImageLayout					 m_final_layout;
	std::size_t						 m_record_count = 0;
};
} // namespace veng::nodes

#endif // VENG_BLITNODE_HPP
