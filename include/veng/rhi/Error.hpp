/**
 * @file
 * @author chris
 * @brief The RHI's own error type — the vk-free stand-in for `vk::Result` on the `veng::rhi` surface.
 *
 * The opaque handles, vocabulary enums, and command encoder already let high-level code speak the
 * RHI without naming Vulkan; @ref veng::rhi::Error closes the last seam: the error half of every
 * `std::expected` the RHI hands back. The `veng::rhi::Device` / `veng::rhi::Swapchain` public methods
 * return `std::expected<T, Error>`, so a caller inspecting *why* a create/acquire/present failed names
 * `rhi::Error`, never `vk::Result`.
 *
 * This header carries no Vulkan dependency (it includes nothing from `<vulkan/...>`), so it can be
 * named from any layer. The one-way mapping from `vk::Result` lives in @ref veng/rhi/Convert.hpp
 * (`to_error`), included only by the driver implementation files — the same split as the rest of the
 * RHI vocabulary.
 *
 * @ingroup rhi
 */

#ifndef VENG_RHI_ERROR_HPP
#define VENG_RHI_ERROR_HPP

#include <cstdint>
#include <string_view>

namespace veng::rhi
{
/**
 * @brief Why an RHI operation failed — the engine-vocabulary stand-in for `vk::Result`'s error codes.
 *
 * A curated set covering the failures the RHI surface actually returns; anything unmapped collapses to
 * @ref Error::UNKNOWN. @ref OUT_OF_DATE / @ref SUBOPTIMAL are the swapchain-rebuild signals.
 *
 * @ingroup rhi
 */
enum class Error : std::int32_t
{
	UNKNOWN = 0,		   ///< Unmapped/unexpected failure.
	OUT_OF_HOST_MEMORY,	   ///< A host (CPU) allocation failed.
	OUT_OF_DEVICE_MEMORY,  ///< A device (GPU) allocation failed.
	INITIALIZATION_FAILED, ///< An object could not be initialised.
	DEVICE_LOST,		   ///< The logical device was lost (a fatal GPU error / reset).
	MEMORY_MAP_FAILED,	   ///< Mapping a host-visible allocation failed.
	FEATURE_NOT_PRESENT,   ///< A required device feature is unavailable.
	SURFACE_LOST,		   ///< The window surface became invalid.
	OUT_OF_DATE,		   ///< The swapchain no longer matches the surface — rebuild it.
	SUBOPTIMAL,			   ///< The swapchain still works but no longer matches the surface optimally.
};

/// @brief A short human-readable name for @p error (for logs / diagnostics).
[[nodiscard]] constexpr std::string_view to_string(Error error) noexcept
{
	switch (error)
	{
		case Error::UNKNOWN: return "unknown error";
		case Error::OUT_OF_HOST_MEMORY: return "out of host memory";
		case Error::OUT_OF_DEVICE_MEMORY: return "out of device memory";
		case Error::INITIALIZATION_FAILED: return "initialization failed";
		case Error::DEVICE_LOST: return "device lost";
		case Error::MEMORY_MAP_FAILED: return "memory map failed";
		case Error::FEATURE_NOT_PRESENT: return "feature not present";
		case Error::SURFACE_LOST: return "surface lost";
		case Error::OUT_OF_DATE: return "swapchain out of date";
		case Error::SUBOPTIMAL: return "swapchain suboptimal";
	}
	return "unknown error";
}
} // namespace veng::rhi

#endif // VENG_RHI_ERROR_HPP
