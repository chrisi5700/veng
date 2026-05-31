/**
 * @file
 * @author chris
 * @brief L4 sink node that issues `vkQueuePresentKHR` after the frame is submitted.
 *
 * Present is a peer sink: `vkQueuePresentKHR` is a queue operation, not a recordable command,
 * so it must run after the frame's command buffer is submitted. The driver/executor owns the
 * command buffer lifecycle and submission; this node's `record` is intentionally a no-op, and
 * the present call lives in `on_submitted`, the post-submit hook provided by the @ref veng::gpu::Sink
 * mixin. This design removes the old "unique terminal that ends the shared buffer" invariant —
 * a @ref veng::nodes::ScreenshotNode or a video-encode sink can now coexist with @ref veng::nodes::PresentNode in the
 * same frame, and a headless renderer simply has no present node at all.
 *
 * The single input is the written swapchain `ImageData` (@ref veng::nodes::BlitNode's output), which both
 * orders this node after the blit and carries the acquired image index plus the render-finished
 * semaphore the present call must wait on. The driver acquires and feeds that ref in.
 *
 * @ingroup graph_nodes
 * @see BlitNode
 * @see ScreenshotNode
 */

#ifndef VENG_PRESENTNODE_HPP
#define VENG_PRESENTNODE_HPP

#include <cstddef>
#include <expected>
#include <span>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/gpu/Sink.hpp>
#include <veng/managers/SwapchainManager.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::nodes
{
/**
 * @brief Concrete @ref veng::gpu::GpuNode and @ref veng::gpu::Sink that closes each frame with a present.
 *
 * @ingroup graph_nodes
 * @see BlitNode
 * @see ScreenshotNode
 * @see SwapchainManager
 */
class PresentNode final : public gpu::GpuNode, public gpu::Sink
{
	 public:
	/**
	 * @brief Construct a present node.
	 *
	 * `swap` owns the swapchain/surface and performs the present call. `presented_image` is
	 * the written swapchain @ref veng::gpu::ImageRef to present (@ref veng::nodes::BlitNode's output) — it
	 * carries the present semaphores and the slot's in-flight fence that the submit signals
	 * (all managed by @ref veng::SwapchainManager). `output` is the frame-done token the driver
	 * demands as the graph sink.
	 *
	 * @param swap            The @ref veng::SwapchainManager that owns the surface and issues presents.
	 * @param presented_image Data handle for the written swapchain @ref veng::gpu::ImageRef edge.
	 * @param output          Data handle for the frame-done sink token.
	 */
	PresentNode(SwapchainManager& swap, graph::DataHandle presented_image, graph::DataHandle output) noexcept;

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return {&m_input, 1}; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	/**
	 * @brief Whether the last present reported the swapchain out-of-date or suboptimal.
	 *
	 * When `true` the driver should rebuild the swapchain before the next frame.
	 *
	 * @return `true` if the swapchain must be recreated.
	 */
	[[nodiscard]] bool out_of_date() const noexcept { return m_out_of_date; }

	/**
	 * @brief How many frames this node has actually closed (submitted + presented).
	 * @return Total number of successful `on_submitted` calls.
	 */
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
