//
// FrameExecutor end-to-end over a headless surface (lavapipe in CI; SKIPped on stock NVIDIA).
//
// Wires the real frame tail — a triangle GraphicsNode -> BlitNode -> PresentNode — and drives it
// through FrameExecutor::run_frame, the per-frame engine loop that previously had zero coverage:
// slot selection, swap acquire + in-flight fence wait, command-buffer lifecycle, graph execute,
// queue submit, and the on_submitted/on_retired sink dispatch (PresentNode issues the present).
// Each Catch SECTION re-runs the shared setup, so every case gets a fresh graph + swapchain.
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <memory>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/managers/CommandManager.hpp>
#include <veng/managers/FrameExecutor.hpp>
#include <veng/managers/SwapchainManager.hpp>
#include <veng/nodes/BlitNode.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/nodes/PresentNode.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/ResourcePool.hpp>
#include <veng/rhi/Convert.hpp>

#include "support/Headless.hpp"
#include "support/VkFault.hpp"

using namespace veng::graph;

namespace
{
constexpr veng::rhi::Extent2D EXTENT{64, 64};
constexpr std::size_t		  FIF = 2;
} // namespace

TEST_CASE("FrameExecutor drives the headless frame loop", "[managers][frameexecutor][headless]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx	  = veng::test::headless_context();
	auto swap_res = veng::SwapchainManager::create(ctx, EXTENT, FIF);
	REQUIRE(swap_res.has_value());
	veng::SwapchainManager swap = std::move(*swap_res);

	veng::ResourcePool	 pool(ctx.device(), ctx.rhi(), ctx.allocator(), FIF);
	veng::CommandManager commands(ctx);
	InlineScheduler		 scheduler;

	// Frame tail: screen-sized triangle -> scene image -> blit into the acquired swapchain image
	// (left in PRESENT_SRC) -> present. The swapchain source is fed by the executor each frame.
	Graph			 graph;
	auto			 screen			 = graph.add_source<veng::rhi::Extent2D>(swap.extent());
	auto			 swapchain_image = graph.add_source<veng::gpu::ImageRef>(veng::gpu::ImageRef{});
	const DataHandle scene_image = graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));
	const DataHandle presented_image =
		graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));
	const DataHandle frame_done = graph.add(std::make_unique<ValueData<int>>(0));

	auto raster = std::make_unique<veng::nodes::GraphicsNode>("tests/slice/triangle.vert", "tests/slice/triangle.frag",
															  swap.format(), veng::rhi::Format::UNDEFINED, 3, screen,
															  scene_image);
	graph.set_producer(scene_image, graph.add(std::move(raster)));

	auto blit = std::make_unique<veng::nodes::BlitNode>(scene_image, swapchain_image, presented_image,
														veng::rhi::TextureUsage::PRESENT);
	graph.set_producer(presented_image, graph.add(std::move(blit)));

	auto  present	  = std::make_unique<veng::nodes::PresentNode>(swap, presented_image, frame_done);
	auto* present_ptr = present.get();
	graph.set_producer(frame_done, graph.add(std::move(present)));

	const std::array	sinks{frame_done};
	veng::FrameExecutor executor(ctx, swap, pool, commands, scheduler, swapchain_image, FIF);

	SECTION("renders, submits, and presents consecutive frames")
	{
		constexpr int FRAMES = 3;
		for (int i = 0; i < FRAMES; ++i)
		{
			const auto outcome = executor.run_frame(graph, sinks, veng::FrameExecutor::Pacing::Continuous);
			REQUIRE(outcome.status == veng::FrameExecutor::Status::Rendered);
			REQUIRE(outcome.plan.nodes().size() > 1); // more than just the present sink ran
		}
		REQUIRE(present_ptr->present_count() == FRAMES); // every frame closed with a present
		REQUIRE_FALSE(present_ptr->out_of_date());
		REQUIRE(executor.frame_index() == FRAMES);
	}

	SECTION("an out-of-date acquire yields Status::OutOfDate")
	{
		const veng::test::ScopedDispatchFault fault{VULKAN_HPP_DEFAULT_DISPATCHER.vkAcquireNextImageKHR,
													+[](VkDevice, VkSwapchainKHR, uint64_t, VkSemaphore, VkFence,
														uint32_t*) -> VkResult { return VK_ERROR_OUT_OF_DATE_KHR; }};
		const auto outcome = executor.run_frame(graph, sinks, veng::FrameExecutor::Pacing::Continuous);
		REQUIRE(outcome.status == veng::FrameExecutor::Status::OutOfDate);
	}

	SECTION("a fence-wait failure yields Status::AcquireFailed")
	{
		const veng::test::ScopedDispatchFault fault{
			VULKAN_HPP_DEFAULT_DISPATCHER.vkWaitForFences,
			+[](VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t) -> VkResult { return VK_TIMEOUT; }};
		const auto outcome = executor.run_frame(graph, sinks, veng::FrameExecutor::Pacing::Continuous);
		REQUIRE(outcome.status == veng::FrameExecutor::Status::AcquireFailed);
	}

	SECTION("a failing node yields Status::NodeFailed")
	{
		// The raster pipeline is built lazily on first record; force its creation to fail so the
		// node fails, graph.execute returns false, and the frame is dropped (not submitted).
		const veng::test::ScopedDispatchFault fault{
			VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateGraphicsPipelines,
			+[](VkDevice, VkPipelineCache, uint32_t, const VkGraphicsPipelineCreateInfo*, const VkAllocationCallbacks*,
				VkPipeline*) -> VkResult { return VK_ERROR_OUT_OF_DEVICE_MEMORY; }};
		const auto outcome = executor.run_frame(graph, sinks, veng::FrameExecutor::Pacing::Continuous);
		REQUIRE(outcome.status == veng::FrameExecutor::Status::NodeFailed);
	}

	SECTION("OnDemand idles when nothing changed")
	{
		// Warm the graph once so every node is valid, then an OnDemand frame with no data change
		// resolves to an empty plan and idles — no acquire, submit, or present.
		const auto warmed = executor.run_frame(graph, sinks, veng::FrameExecutor::Pacing::OnDemand);
		REQUIRE(warmed.status == veng::FrameExecutor::Status::Rendered);
		const auto idled = executor.run_frame(graph, sinks, veng::FrameExecutor::Pacing::OnDemand);
		REQUIRE(idled.status == veng::FrameExecutor::Status::Idled);
	}

	(void)ctx.device().waitIdle(); // drain any submitted frame before the swapchain/pool tear down
}
