//
// Created by chris on 5/30/26.
//
// A small, declarative sampler description shared by the nodes that own a `vk::Sampler`
// (review.md item 4). Before this, GraphicsNode hard-coded a linear / clamp-to-edge / no-mip
// sampler — the right default for sampling a full-screen render target, but wrong for material
// textures, which want repeat addressing, trilinear mip sampling across a full LOD range, and
// anisotropic filtering. `SamplerConfig` makes those knobs explicit and offers two presets:
// `render_target()` (the historical default) and `texture()` (for sampled assets).
//

#ifndef VENG_SAMPLERCONFIG_HPP
#define VENG_SAMPLERCONFIG_HPP

#include <vulkan/vulkan.hpp>

namespace veng::gpu
{
struct SamplerConfig
{
	vk::Filter			   mag_filter	  = vk::Filter::eLinear;
	vk::Filter			   min_filter	  = vk::Filter::eLinear;
	vk::SamplerMipmapMode  mipmap_mode	  = vk::SamplerMipmapMode::eNearest;
	vk::SamplerAddressMode address_mode	  = vk::SamplerAddressMode::eClampToEdge; // u, v and w
	bool				   anisotropy	  = false;
	float				   max_anisotropy = 1.0F;
	float				   min_lod		  = 0.0F;
	float				   max_lod		  = 0.0F; // VK_LOD_CLAMP_NONE (1000) to sample the full mip chain

	/// The historical GraphicsNode default: linear, clamp-to-edge, no mip sampling. Correct for
	/// sampling a full-screen render-target texture (one mip, edge-clamped UVs in [0,1]).
	[[nodiscard]] static SamplerConfig render_target() noexcept { return SamplerConfig{}; }

	/// Material-texture sampling: repeat addressing (UVs tile), trilinear mip filtering across the
	/// whole chain, and anisotropic filtering at `aniso` (clamped to the device max by the caller;
	/// the device must have enabled `samplerAnisotropy`).
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
