/**
 * @file
 * @author chris
 * @brief The RHI bind group — an immutable descriptor set described in RHI vocabulary, in place of
 *        a hand-built `vk::WriteDescriptorSet` list.
 *
 * A node no longer assembles `vk::DescriptorBufferInfo`/`DescriptorImageInfo`/`WriteDescriptorSet`
 * by hand. It builds a list of @ref veng::rhi::BindGroupEntry — each one a binding index, a
 * @ref veng::rhi::BindingType, and the @ref veng::rhi::BufferHandle / @ref veng::rhi::TextureHandle /
 * @ref veng::rhi::SamplerHandle it binds — and hands it to @ref veng::rhi::Device::update_bind_group,
 * which resolves the handles and performs the single `updateDescriptorSets`. The resulting
 * @ref veng::rhi::BindGroup is bound through the command encoder. No node names a Vulkan descriptor
 * type, image layout, or write-struct anymore.
 *
 * The underlying `vk::DescriptorSet` is allocated by the node's @ref veng::DescriptorAllocator (the
 * L2 descriptor-pool layer) and wrapped here; the bind group is non-owning.
 *
 * @ingroup rhi
 */

#ifndef VENG_RHI_BINDGROUP_HPP
#define VENG_RHI_BINDGROUP_HPP

#include <cstdint>
#include <veng/rhi/Handles.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::rhi
{
/**
 * @brief The kind of resource bound at a descriptor binding — the stand-in for `vk::DescriptorType`.
 * @ingroup rhi
 */
enum class BindingType : std::uint8_t
{
	UNIFORM_BUFFER, ///< A uniform buffer (cbuffer / `ConstantBuffer`).
	STORAGE_BUFFER, ///< A storage buffer (SSBO / `StructuredBuffer`).
	SAMPLED_IMAGE,	///< A sampled image (`Texture2D`).
	SAMPLER,		///< A sampler (`SamplerState`).
};

/**
 * @brief One resource bound at a descriptor binding, in RHI vocabulary.
 *
 * Set @ref binding and @ref type, then the field matching the type: @ref buffer (+ @ref buffer_size)
 * for the buffer types, @ref texture for @ref BindingType::SAMPLED_IMAGE, @ref sampler for
 * @ref BindingType::SAMPLER. The others stay default (invalid) and are ignored.
 *
 * @ingroup rhi
 */
struct BindGroupEntry
{
	std::uint32_t binding = 0;							 ///< The descriptor binding index in set 0.
	BindingType	  type	  = BindingType::UNIFORM_BUFFER; ///< Which resource kind this entry binds.
	BufferHandle  buffer{};								 ///< Buffer to bind (UNIFORM_BUFFER / STORAGE_BUFFER).
	std::uint64_t buffer_size = 0;						 ///< Bound buffer range in bytes.
	TextureHandle texture{};							 ///< Texture to bind (SAMPLED_IMAGE).
	SamplerHandle sampler{};							 ///< Sampler to bind (SAMPLER).

	/// Value equality over all bound handles — the change-cutoff signal that lets a node skip a
	/// descriptor rewrite when its bindings are identical to the previous frame's.
	friend bool operator==(const BindGroupEntry&, const BindGroupEntry&) noexcept = default;
};

/**
 * @brief An immutable descriptor set, described in RHI vocabulary and bound through the encoder.
 *
 * Wraps the `vk::DescriptorSet` the node allocated; populated by @ref Device::update_bind_group and
 * bound via @ref CommandEncoder::bind_descriptor_set. Non-owning — the descriptor allocator owns the
 * set's lifetime.
 *
 * @ingroup rhi
 */
struct BindGroup
{
	vk::DescriptorSet  set{}; ///< The underlying descriptor set (allocator-owned).
	[[nodiscard]] bool valid() const noexcept { return static_cast<bool>(set); }
};
} // namespace veng::rhi

#endif // VENG_RHI_BINDGROUP_HPP
