//
// Headless surface support for the test suite.
//
// The swapchain / present / FrameExecutor path needs a VkSurfaceKHR, which normally means a window.
// VK_EXT_headless_surface provides a real, window-less surface (vkCreateHeadlessSurfaceEXT) so that
// whole tier is exercisable in CI with no display. Mesa lavapipe (the llvmpipe software rasterizer)
// implements it; the stock NVIDIA ICD does not. The Context already exposes the seam — it takes the
// instance extensions plus a surface_factory and verifies present support on whatever surface it is
// handed — so a headless context is just the right arguments.
//
// `headless_context()` SKIPs (with an install hint) when the extension is absent, so the suite stays
// green for contributors on a stock driver while still running fully under lavapipe / in CI.
//

#ifndef VENG_TESTS_SUPPORT_HEADLESS_HPP
#define VENG_TESTS_SUPPORT_HEADLESS_HPP

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <string_view>
#include <utility>
#include <veng/context/Context.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::test
{
/**
 * @brief A @ref veng::Context backed by a window-less `VK_EXT_headless_surface`.
 *
 * SKIPs the calling test (with an install hint) when a *usable* headless surface can't be made — so
 * the swapchain/present/FrameExecutor tests are no-ops on a stock driver and run under lavapipe.
 *
 * The gate is the actual creation result, not extension enumeration: the Vulkan *loader* advertises
 * `VK_EXT_headless_surface` even with only the NVIDIA ICD installed, but that ICD can't present to
 * the resulting surface, so @ref veng::Context::create fails its present-support check there. Mesa
 * lavapipe both advertises the extension and can present to it, so creation succeeds and the tests
 * run. Basing the SKIP on the create result captures exactly that distinction.
 *
 * @param title Application title for the instance.
 * @return A ready windowed-but-displayless Context.
 */
inline Context headless_context(std::string_view title = "Headless Surface Test")
{
	static constexpr std::array<const char*, 2> extensions{VK_KHR_SURFACE_EXTENSION_NAME,
														   VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME};
	auto										ctx = Context::create(
		title, extensions,
		[](VkInstance instance) -> VkSurfaceKHR
		{
			// The loader exposes this entry point even without an ICD that implements it; guard
			// anyway so an ancient loader yields a null surface (a clean Context error) not a crash.
			if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateHeadlessSurfaceEXT == nullptr)
			{
				return VK_NULL_HANDLE;
			}
			const VkHeadlessSurfaceCreateInfoEXT info{
				.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT, .pNext = nullptr, .flags = 0};
			VkSurfaceKHR						 surface = VK_NULL_HANDLE;
			(void)VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateHeadlessSurfaceEXT(instance, &info, nullptr, &surface);
			return surface;
		});
	if (!ctx.has_value())
	{
		SKIP("No usable headless surface (the stock NVIDIA driver can't present to VK_EXT_headless_surface). "
			 "Install Mesa lavapipe (Arch: `pacman -S vulkan-swrast`) and run the suite under it "
			 "(cmake -DVENG_TEST_VULKAN_ICD=/usr/share/vulkan/icd.d/lvp_icd.json) to cover the "
			 "swapchain / present / FrameExecutor path.");
	}
	return std::move(*ctx);
}
} // namespace veng::test

#endif // VENG_TESTS_SUPPORT_HEADLESS_HPP
