//
// Created by chris on 5/25/26.
//
// L4 node — the frame terminator (design.md §L4 swapchain & present). It is the one
// place a queue operation lives, because `vkQueuePresentKHR` is not a command that can
// be recorded into the frame's command buffer — it must run *after* that buffer is
// submitted. So when this node executes (last, as the graph sink) it: ends the recorded
// command buffer, submits it (waiting on the acquire semaphore, signalling
// render-finished), and presents. Everything upstream stays pure command-recording.
//
// Its single input is the written swapchain `ImageData` (BlitNode's output), which both
// orders it after the blit and carries the acquired image index + the two present
// semaphores. The driver only acquires and feeds that ref in; this node closes the frame.
//
// PRECONDITION: this must be the unique terminal node — it ends the shared command
// buffer, so any recording node not upstream of it would be truncated.
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
	/// `swap` owns the swapchain/surface and does the present. `fence` is the per-frame
	/// fence the submit signals (the driver waits on it before reusing the frame).
	/// `presented_image` is the written swapchain ref to present (BlitNode's output);
	/// `output` is the frame-done token the driver demands as the graph sink.
	PresentNode(SwapchainManager& swap, vk::Fence fence, graph::DataHandle presented_image,
				graph::DataHandle output) noexcept;

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return {&m_input, 1}; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	/// True if the last present reported the swapchain out-of-date/suboptimal — the driver
	/// should rebuild before the next frame.
	[[nodiscard]] bool out_of_date() const noexcept { return m_out_of_date; }

	/// How many frames this node has actually closed (submitted + presented).
	[[nodiscard]] std::size_t present_count() const noexcept { return m_present_count; }

	 protected:
	[[nodiscard]] std::expected<bool, graph::ExecError> record(gpu::GpuExecContext& ctx) override;

	 private:
	SwapchainManager* m_swap;
	vk::Fence		  m_fence;
	graph::DataHandle m_input;
	graph::DataHandle m_output;
	bool			  m_out_of_date	  = false;
	std::size_t		  m_present_count = 0;
};
} // namespace veng::nodes

#endif // VENG_PRESENTNODE_HPP
