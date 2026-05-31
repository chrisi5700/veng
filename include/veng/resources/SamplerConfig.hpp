/**
 * @file
 * @author chris
 * @brief Declarative sampler description with preset factory functions.
 *
 * A small, declarative sampler description shared by the nodes that own a `vk::Sampler`.
 * Before this, `GraphicsNode` hard-coded a linear / clamp-to-edge / no-mip sampler — the
 * right default for sampling a full-screen render target, but wrong for material textures,
 * which want repeat addressing, trilinear mip sampling across a full LOD range, and
 * anisotropic filtering. @ref veng::gpu::SamplerConfig makes those knobs explicit and offers two
 * presets: @ref veng::gpu::SamplerConfig::render_target (the historical default) and
 * @ref veng::gpu::SamplerConfig::texture (for sampled assets).
 *
 * @ingroup resources
 */

#ifndef VENG_SAMPLERCONFIG_HPP
#define VENG_SAMPLERCONFIG_HPP

#include <vulkan/vulkan.hpp>

namespace veng::gpu
{
/**
 * @brief Declarative description of a Vulkan sampler, with two built-in presets.
 * @ingroup resources
 */
struct SamplerConfig
{
	vk::Filter			   mag_filter  = vk::Filter::eLinear;			  ///< Magnification filter.
	vk::Filter			   min_filter  = vk::Filter::eLinear;			  ///< Minification filter.
	vk::SamplerMipmapMode  mipmap_mode = vk::SamplerMipmapMode::eNearest; ///< Mip-level selection mode.
	vk::SamplerAddressMode address_mode =
		vk::SamplerAddressMode::eClampToEdge; ///< Address mode applied to U, V, and W.
	bool  anisotropy	 = false;			  ///< Whether anisotropic filtering is enabled.
	float max_anisotropy = 1.0F;			  ///< Maximum anisotropy ratio (used when @ref anisotropy is true).
	float min_lod		 = 0.0F;			  ///< Minimum LOD clamp.
	float max_lod		 = 0.0F; ///< Maximum LOD clamp; set to `VK_LOD_CLAMP_NONE` (1000) to sample the full mip chain.

	/**
	 * @brief Preset for sampling a full-screen render-target texture.
	 *
	 * Linear filtering, clamp-to-edge addressing, no mip sampling. Correct for
	 * single-mip, edge-clamped render targets with UVs in [0, 1].
	 *
	 * @return A @ref veng::gpu::SamplerConfig configured as a render-target sampler.
	 */
	[[nodiscard]] static SamplerConfig render_target() noexcept { return SamplerConfig{}; }

	/**
	 * @brief Preset for sampling a material texture (repeat, trilinear, anisotropic).
	 *
	 * Repeat addressing (UVs tile), trilinear mip filtering across the whole chain
	 * (`max_lod = VK_LOD_CLAMP_NONE`), and anisotropic filtering at `aniso`. The
	 * caller is responsible for clamping `aniso` to the device maximum; the device
	 * feature `samplerAnisotropy` must have been enabled.
	 *
	 * @param aniso Maximum anisotropy ratio; values <= 1.0 disable anisotropic filtering.
	 * @return A @ref veng::gpu::SamplerConfig configured for material-texture sampling.
	 */
	[[nodiscard]] static SamplerConfig texture(float aniso = 8.0F) noexcept
	{
		return SamplerConfig{.mag_filter	 = vk::Filter::eLinear,
							 .min_filter	 = vk::Filter::eLinear,
							 .mipmap_mode	 = vk::SamplerMipmapMode::eLinear,
							 .address_mode	 = vk::SamplerAddressMode::eRepeat,
							 .anisotropy	 = aniso > 1.0F,
							 .max_anisotropy = aniso,
							 .min_lod		 = 0.0F,
							 .max_lod		 = VK_LOD_CLAMP_NONE};
	}

	/**
	 * @brief Convert this config into a `vk::SamplerCreateInfo` ready for `vkCreateSampler`.
	 * @return A `vk::SamplerCreateInfo` populated from this config's fields.
	 */
	[[nodiscard]] vk::SamplerCreateInfo to_create_info() const noexcept
	{
		return vk::SamplerCreateInfo()
			.setMagFilter(mag_filter)
			.setMinFilter(min_filter)
			.setMipmapMode(mipmap_mode)
			.setAddressModeU(address_mode)
			.setAddressModeV(address_mode)
			.setAddressModeW(address_mode)
			.setAnisotropyEnable(static_cast<vk::Bool32>(anisotropy))
			.setMaxAnisotropy(max_anisotropy)
			.setMinLod(min_lod)
			.setMaxLod(max_lod);
	}
};
} // namespace veng::gpu

#endif // VENG_SAMPLERCONFIG_HPP
