/**
 * @file
 * @author chris
 * @brief Opaque RHI resource handles — the values that flow on graph edges in place of raw Vulkan
 *        handles.
 *
 * A handle is a small id into a registry owned by @ref veng::rhi::Device. It identifies a GPU
 * resource (texture, buffer, sampler) without naming Vulkan: the graph's `ImageRef`/`BufferRef`/…
 * carry these instead of `vk::Image`/`vk::Buffer`, and a consumer resolves a handle back to its vk
 * objects through the device only at the point it actually records a command. A default-constructed
 * handle is invalid and resolves to a null vk handle — the "not produced yet" sentinel.
 *
 * @ingroup rhi
 */

#ifndef VENG_RHI_HANDLES_HPP
#define VENG_RHI_HANDLES_HPP

#include <cstdint>

namespace veng::rhi
{
/// @brief The invalid handle id (a default-constructed handle).
inline constexpr std::uint32_t INVALID_HANDLE = ~0U;

/// @brief Opaque handle to a registered texture (image + default view). @see Device::texture
struct TextureHandle
{
	std::uint32_t	   id = INVALID_HANDLE;
	[[nodiscard]] bool valid() const noexcept { return id != INVALID_HANDLE; }
	friend bool		   operator==(const TextureHandle&, const TextureHandle&) noexcept = default;
};

/// @brief Opaque handle to a registered buffer. @see Device::buffer
struct BufferHandle
{
	std::uint32_t	   id = INVALID_HANDLE;
	[[nodiscard]] bool valid() const noexcept { return id != INVALID_HANDLE; }
	friend bool		   operator==(const BufferHandle&, const BufferHandle&) noexcept = default;
};

/// @brief Opaque handle to a device-owned sampler. @see Device::create_sampler
struct SamplerHandle
{
	std::uint32_t	   id = INVALID_HANDLE;
	[[nodiscard]] bool valid() const noexcept { return id != INVALID_HANDLE; }
	friend bool		   operator==(const SamplerHandle&, const SamplerHandle&) noexcept = default;
};
} // namespace veng::rhi

#endif // VENG_RHI_HANDLES_HPP
