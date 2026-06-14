/**
 * @file
 * @author chris
 * @brief Declarative sampler description with preset factory functions.
 *
 * A small, declarative sampler description shared by the nodes that own a sampler. Before this,
 * `GraphicsNode` hard-coded a linear / clamp-to-edge / no-mip sampler — the right default for
 * sampling a full-screen render target, but wrong for material textures, which want repeat
 * addressing, trilinear mip sampling across a full LOD range, and anisotropic filtering.
 * @ref veng::gpu::SamplerConfig makes those knobs explicit and offers two presets:
 * @ref veng::gpu::SamplerConfig::render_target (the historical default) and
 * @ref veng::gpu::SamplerConfig::texture (for sampled assets).
 *
 * The description carries no Vulkan dependency — its fields are @ref veng::rhi vocabulary enums.
 * Turning a config into a `vk::SamplerCreateInfo` lives in @ref veng/resources/SamplerConvert.hpp,
 * the seam included only by the implementation files that actually create samplers.
 *
 * @ingroup resources
 */

#ifndef VENG_SAMPLERCONFIG_HPP
#define VENG_SAMPLERCONFIG_HPP

#include <veng/rhi/Enums.hpp>

namespace veng::gpu
{
/**
 * @brief Declarative description of a texture sampler, with two built-in presets.
 * @ingroup resources
 */
struct SamplerConfig
{
	/// Sentinel `max_lod` value that samples the full mip chain (mirrors `VK_LOD_CLAMP_NONE`).
	static constexpr float LOD_CLAMP_NONE = 1000.0F;

	rhi::Filter		 mag_filter		= rhi::Filter::LINEAR;			   ///< Magnification filter.
	rhi::Filter		 min_filter		= rhi::Filter::LINEAR;			   ///< Minification filter.
	rhi::MipmapMode	 mipmap_mode	= rhi::MipmapMode::NEAREST;		   ///< Mip-level selection mode.
	rhi::AddressMode address_mode	= rhi::AddressMode::CLAMP_TO_EDGE; ///< Address mode applied to U, V, and W.
	bool			 anisotropy		= false;						   ///< Whether anisotropic filtering is enabled.
	float			 max_anisotropy = 1.0F; ///< Maximum anisotropy ratio (used when @ref anisotropy is true).
	float			 min_lod		= 0.0F; ///< Minimum LOD clamp.
	float			 max_lod = 0.0F; ///< Maximum LOD clamp; set to @ref LOD_CLAMP_NONE to sample the full mip chain.

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
	 * (`max_lod = LOD_CLAMP_NONE`), and anisotropic filtering at `aniso`. The caller is
	 * responsible for clamping `aniso` to the device maximum; the device feature
	 * `samplerAnisotropy` must have been enabled.
	 *
	 * @param aniso Maximum anisotropy ratio; values <= 1.0 disable anisotropic filtering.
	 * @return A @ref veng::gpu::SamplerConfig configured for material-texture sampling.
	 */
	[[nodiscard]] static SamplerConfig texture(float aniso = 8.0F) noexcept
	{
		return SamplerConfig{.mag_filter	 = rhi::Filter::LINEAR,
							 .min_filter	 = rhi::Filter::LINEAR,
							 .mipmap_mode	 = rhi::MipmapMode::LINEAR,
							 .address_mode	 = rhi::AddressMode::REPEAT,
							 .anisotropy	 = aniso > 1.0F,
							 .max_anisotropy = aniso,
							 .min_lod		 = 0.0F,
							 .max_lod		 = LOD_CLAMP_NONE};
	}
};
} // namespace veng::gpu

#endif // VENG_SAMPLERCONFIG_HPP
