/**
 * @file
 * @author chris
 * @brief Turn a vk-free @ref veng::gpu::SamplerConfig into a `vk::SamplerCreateInfo`.
 *
 * The conversion seam for samplers: it includes `<vulkan/vulkan.hpp>` and is included only by the
 * implementation files that actually call `createSampler` — so @ref veng/resources/SamplerConfig.hpp
 * can stay a pure description in @ref veng::rhi vocabulary, with no Vulkan dependency.
 *
 * @ingroup resources
 */

#ifndef VENG_RESOURCES_SAMPLERCONVERT_HPP
#define VENG_RESOURCES_SAMPLERCONVERT_HPP

#include <veng/resources/SamplerConfig.hpp>
#include <veng/rhi/Convert.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::gpu
{
/**
 * @brief Build the `vk::SamplerCreateInfo` for a @ref SamplerConfig (one address mode on U/V/W).
 * @param config The sampler description.
 * @return A `vk::SamplerCreateInfo` populated from @p config, ready for `vkCreateSampler`.
 */
[[nodiscard]] inline vk::SamplerCreateInfo to_create_info(const SamplerConfig& config) noexcept
{
	return vk::SamplerCreateInfo()
		.setMagFilter(rhi::to_vk(config.mag_filter))
		.setMinFilter(rhi::to_vk(config.min_filter))
		.setMipmapMode(rhi::to_vk(config.mipmap_mode))
		.setAddressModeU(rhi::to_vk(config.address_mode))
		.setAddressModeV(rhi::to_vk(config.address_mode))
		.setAddressModeW(rhi::to_vk(config.address_mode))
		.setAnisotropyEnable(static_cast<vk::Bool32>(config.anisotropy))
		.setMaxAnisotropy(config.max_anisotropy)
		.setMinLod(config.min_lod)
		.setMaxLod(config.max_lod);
}
} // namespace veng::gpu

#endif // VENG_RESOURCES_SAMPLERCONVERT_HPP
