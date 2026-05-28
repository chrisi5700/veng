//
// Created by chris on 5/25/26.
//
// L4 sink node — present is a peer sink ([[pass-draw-redesign]]). `vkQueuePresentKHR` is a
// queue op, not a recordable command, so it has to run *after* the frame's command buffer is
// submitted. The driver/executor owns the CB lifecycle + submit; this node's `record` is a
// no-op and the present call lives in `on_submitted`, the post-submit hook every `GpuNode`
// inherits. That removes the old "unique terminal that ends the shared buffer" invariant — a
// screenshot or video-encode sink can now coexist with present in the same frame, and a
// headless renderer just has no present node at all.
//
// Its single input is the written swapchain `ImageData` (BlitNode's output), which both
// orders it after the blit and carries the acquired image index + the render-finished
// semaphore the present call must wait on. The driver acquires and feeds that ref in.
//

#ifndef VENG_PRESENTNODE_HPP
#define VENG_PRESENTNODE_HPP

#include <cstddef>
#include <expected>
#include <span>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/managers/SwapchainManager.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::nodes
{
class PresentNode final : public gpu::GpuNode
{
	 public:
	/// `swap` owns the swapchain/surface and does the present. `presented_image` is the
	/// written swapchain ref to present (BlitNode's output) — it carries the present
	/// semaphores and the slot's in-flight fence that the submit signals (all managed by
	/// the SwapchainManager). `output` is the frame-done token the driver demands as the
	/// graph sink.
	PresentNode(SwapchainManager& swap, graph::DataHandle presented_image, graph::DataHandle output) noexcept;

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return {&m_input, 1}; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	/// True if the last present reported the swapchain out-of-date/suboptimal — the driver
	/// should rebuild before the next frame.
	[[nodiscard]] bool out_of_date() const noexcept { return m_out_of_date; }

	/// How many frames this node has actually closed (submitted + presented).
	[[nodiscard]] std::size_t present_count() const noexcept { return m_present_count; }

	 protected:
	[[nodiscard]] std::expected<bool, graph::ExecError> record(gpu::GpuExecContext& ctx) override;
	void												on_submitted(gpu::SubmitContext& ctx) noexcept override;

	 private:
	SwapchainManager* m_swap;
	graph::DataHandle m_input;
	graph::DataHandle m_output;
	bool			  m_out_of_date	  = false;
	std::size_t		  m_present_count = 0;
};
} // namespace veng::nodes

#endif // VENG_PRESENTNODE_HPP
