/**
 * @file
 * @author chris
 * @brief Declarative sampler description (RHI vocabulary) with preset factory functions.
 *
 * A small, declarative sampler description shared by the nodes that own a sampler. Before this,
 * `GraphicsNode` hard-coded a linear / clamp-to-edge / no-mip sampler — the right default for
 * sampling a full-screen render target, but wrong for material textures, which want repeat
 * addressing, trilinear mip sampling across a full LOD range, and anisotropic filtering.
 * @ref veng::rhi::SamplerDesc makes those knobs explicit and offers two presets:
 * @ref veng::rhi::SamplerDesc::render_target (the historical default) and
 * @ref veng::rhi::SamplerDesc::texture (for sampled assets).
 *
 * The description carries no Vulkan dependency — its fields are @ref veng::rhi vocabulary enums, so
 * it is the type @ref veng::rhi::Device::create_sampler takes. Turning a desc into a
 * `vk::SamplerCreateInfo` happens inside `Device.cpp`, the one place that names Vulkan.
 *
 * @ingroup rhi
 */

#ifndef VENG_RHI_SAMPLERDESC_HPP
#define VENG_RHI_SAMPLERDESC_HPP

#include <veng/rhi/Enums.hpp>

namespace veng::rhi
{
/**
 * @brief Declarative description of a texture sampler, with two built-in presets.
 * @ingroup rhi
 */
struct SamplerDesc
{
	/// Sentinel `max_lod` value that samples the full mip chain (mirrors `VK_LOD_CLAMP_NONE`).
	static constexpr float LOD_CLAMP_NONE = 1000.0F;

	Filter		mag_filter	   = Filter::LINEAR;			 ///< Magnification filter.
	Filter		min_filter	   = Filter::LINEAR;			 ///< Minification filter.
	MipmapMode	mipmap_mode	   = MipmapMode::NEAREST;		 ///< Mip-level selection mode.
	AddressMode address_mode   = AddressMode::CLAMP_TO_EDGE; ///< Address mode applied to U, V, and W.
	bool		anisotropy	   = false;						 ///< Whether anisotropic filtering is enabled.
	float		max_anisotropy = 1.0F; ///< Maximum anisotropy ratio (used when @ref anisotropy is true).
	float		min_lod		   = 0.0F; ///< Minimum LOD clamp.
	float		max_lod = 0.0F; ///< Maximum LOD clamp; set to @ref LOD_CLAMP_NONE to sample the full mip chain.

	/**
	 * @brief Preset for sampling a full-screen render-target texture.
	 *
	 * Linear filtering, clamp-to-edge addressing, no mip sampling. Correct for
	 * single-mip, edge-clamped render targets with UVs in [0, 1].
	 *
	 * @return A @ref veng::rhi::SamplerDesc configured as a render-target sampler.
	 */
	[[nodiscard]] static SamplerDesc render_target() noexcept { return SamplerDesc{}; }

	/**
	 * @brief Preset for sampling a material texture (repeat, trilinear, anisotropic).
	 *
	 * Repeat addressing (UVs tile), trilinear mip filtering across the whole chain
	 * (`max_lod = LOD_CLAMP_NONE`), and anisotropic filtering at `aniso`. The caller is
	 * responsible for clamping `aniso` to the device maximum; the device feature
	 * `samplerAnisotropy` must have been enabled.
	 *
	 * @param aniso Maximum anisotropy ratio; values <= 1.0 disable anisotropic filtering.
	 * @return A @ref veng::rhi::SamplerDesc configured for material-texture sampling.
	 */
	[[nodiscard]] static SamplerDesc texture(float aniso = 8.0F) noexcept
	{
		return SamplerDesc{.mag_filter	   = Filter::LINEAR,
						   .min_filter	   = Filter::LINEAR,
						   .mipmap_mode	   = MipmapMode::LINEAR,
						   .address_mode   = AddressMode::REPEAT,
						   .anisotropy	   = aniso > 1.0F,
						   .max_anisotropy = aniso,
						   .min_lod		   = 0.0F,
						   .max_lod		   = LOD_CLAMP_NONE};
	}
};
} // namespace veng::rhi

#endif // VENG_RHI_SAMPLERDESC_HPP
