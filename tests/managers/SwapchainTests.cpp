//
// SwapchainManager tests over a window-less VK_EXT_headless_surface (Mesa lavapipe in CI; SKIPped on
// a stock NVIDIA driver). Covers the build, acquire, present, rebuild/resize, and RAII paths, plus
// the out-of-date / fence-timeout / present-error branches via the dispatcher fault seam — the tier
// that previously had zero coverage because it needs a surface.
//

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <optional>
#include <utility>
#include <veng/logging/Logger.hpp>
#include <veng/managers/SwapchainManager.hpp>

#include "support/Headless.hpp"
#include "support/VkFault.hpp"

namespace
{
constexpr veng::rhi::Extent2D START{64, 64};
} // namespace

TEST_CASE("SwapchainManager builds a headless swapchain with images", "[managers][swapchain][headless]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = veng::test::headless_context();

	auto swap = veng::SwapchainManager::create(ctx, START, 2);
	REQUIRE(swap.has_value());
	REQUIRE(swap->extent().width == START.width);
	REQUIRE(swap->extent().height == START.height);
	REQUIRE(swap->format() != veng::rhi::Format::UNDEFINED);
	REQUIRE(swap->image(0)); // at least one swapchain image was retrieved
}

TEST_CASE("SwapchainManager acquires an image for a frame slot", "[managers][swapchain][headless]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx  = veng::test::headless_context();
	auto swap = veng::SwapchainManager::create(ctx, START, 2);
	REQUIRE(swap.has_value());

	const auto acquired = swap->acquire(0);
	REQUIRE(acquired.has_value());	// no hard error
	REQUIRE(acquired->has_value()); // not out-of-date
	const auto frame = acquired->value();
	REQUIRE(frame.image_available);
	REQUIRE(frame.render_finished);
	REQUIRE(frame.in_flight);
}

TEST_CASE("SwapchainManager rebuilds at a new extent", "[managers][swapchain][headless]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx  = veng::test::headless_context();
	auto swap = veng::SwapchainManager::create(ctx, START, 1);
	REQUIRE(swap.has_value());

	REQUIRE(ctx.device().waitIdle() == vk::Result::eSuccess);
	const veng::rhi::Extent2D bigger{128, 96};
	REQUIRE(swap->rebuild(bigger).has_value());
	REQUIRE(swap->extent().width == bigger.width);
	REQUIRE(swap->extent().height == bigger.height);
	// The rebuilt swapchain still acquires cleanly (semaphores/fences survived the rebuild).
	const auto acquired = swap->acquire(0);
	REQUIRE(acquired.has_value());
	REQUIRE(acquired->has_value());
}

TEST_CASE("SwapchainManager move-construction transfers ownership", "[managers][swapchain][headless][raii]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx  = veng::test::headless_context();
	auto swap = veng::SwapchainManager::create(ctx, START, 1);
	REQUIRE(swap.has_value());
	const vk::Image first_image = swap->image(0);

	const veng::SwapchainManager moved = std::move(*swap);
	REQUIRE(moved.image(0) == first_image); // adopted; moved-from is inert and destroyed once
}

TEST_CASE("SwapchainManager::acquire reports out-of-date as nullopt", "[managers][swapchain][headless][error]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	auto ctx  = veng::test::headless_context();
	auto swap = veng::SwapchainManager::create(ctx, START, 1);
	REQUIRE(swap.has_value());

	const veng::test::ScopedDispatchFault fault{VULKAN_HPP_DEFAULT_DISPATCHER.vkAcquireNextImageKHR,
												+[](VkDevice, VkSwapchainKHR, uint64_t, VkSemaphore, VkFence,
													uint32_t*) -> VkResult { return VK_ERROR_OUT_OF_DATE_KHR; }};
	const auto							  acquired = swap->acquire(0);
	REQUIRE(acquired.has_value());		  // not a hard failure...
	REQUIRE_FALSE(acquired->has_value()); // ...just out-of-date -> caller should rebuild
}

TEST_CASE("SwapchainManager::acquire maps a fence-wait failure to device-lost",
		  "[managers][swapchain][headless][error]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	auto ctx  = veng::test::headless_context();
	auto swap = veng::SwapchainManager::create(ctx, START, 1);
	REQUIRE(swap.has_value());

	const veng::test::ScopedDispatchFault fault{VULKAN_HPP_DEFAULT_DISPATCHER.vkWaitForFences,
												+[](VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t) -> VkResult
												{ return VK_TIMEOUT; }};
	const auto							  acquired = swap->acquire(0);
	REQUIRE_FALSE(acquired.has_value());
	REQUIRE(acquired.error() == vk::Result::eErrorDeviceLost);
}

TEST_CASE("SwapchainManager::present maps out-of-date and hard errors", "[managers][swapchain][headless][error]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	auto ctx  = veng::test::headless_context();
	auto swap = veng::SwapchainManager::create(ctx, START, 1);
	REQUIRE(swap.has_value());

	const vk::Queue queue = ctx.graphics_queue();
	const auto		sem	  = ctx.device().createSemaphore({});
	REQUIRE(sem.result == vk::Result::eSuccess);
	const veng::rhi::SemaphoreHandle sem_handle = ctx.rhi().register_semaphore(sem.value);

	SECTION("out-of-date / suboptimal -> rebuild requested (true)")
	{
		const veng::test::ScopedDispatchFault fault{VULKAN_HPP_DEFAULT_DISPATCHER.vkQueuePresentKHR,
													+[](VkQueue, const VkPresentInfoKHR*) -> VkResult
													{ return VK_ERROR_OUT_OF_DATE_KHR; }};
		const auto							  out = swap->present(queue, 0, sem_handle);
		REQUIRE(out.has_value());
		REQUIRE(out.value()); // true == swapchain needs rebuilding
	}
	SECTION("hard error -> unexpected")
	{
		const veng::test::ScopedDispatchFault fault{VULKAN_HPP_DEFAULT_DISPATCHER.vkQueuePresentKHR,
													+[](VkQueue, const VkPresentInfoKHR*) -> VkResult
													{ return VK_ERROR_DEVICE_LOST; }};
		const auto							  out = swap->present(queue, 0, sem_handle);
		REQUIRE_FALSE(out.has_value());
		REQUIRE(out.error() == vk::Result::eErrorDeviceLost);
	}

	ctx.device().destroySemaphore(sem.value);
}

// Sync-object creation failures during create() must return a typed error and tear down cleanly
// (the swapchain + any already-created semaphores are destroyed, not leaked — ASan/UBSan verifies).
// Unlike Context::create, SwapchainManager runs on the already-initialised dispatcher, so the fault
// seam reaches inside it.

TEST_CASE("SwapchainManager::create reports a semaphore failure and cleans up",
		  "[managers][swapchain][headless][error]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	auto ctx = veng::test::headless_context();

	const veng::test::ScopedDispatchFault fault{
		VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateSemaphore,
		+[](VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*, VkSemaphore*) -> VkResult
		{ return VK_ERROR_OUT_OF_DEVICE_MEMORY; }};
	const auto swap = veng::SwapchainManager::create(ctx, START, 2);
	REQUIRE_FALSE(swap.has_value());
	REQUIRE(swap.error() == vk::Result::eErrorOutOfDeviceMemory);
}

TEST_CASE("SwapchainManager::create reports a fence failure and cleans up", "[managers][swapchain][headless][error]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	auto ctx = veng::test::headless_context();

	// The swapchain + render-finished semaphores build fine; the per-slot in-flight fence fails, so
	// create() must destroy what it already made and return the error rather than leak or crash.
	const veng::test::ScopedDispatchFault fault{VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateFence,
												+[](VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*,
													VkFence*) -> VkResult { return VK_ERROR_OUT_OF_DEVICE_MEMORY; }};
	const auto							  swap = veng::SwapchainManager::create(ctx, START, 2);
	REQUIRE_FALSE(swap.has_value());
	REQUIRE(swap.error() == vk::Result::eErrorOutOfDeviceMemory);
}
