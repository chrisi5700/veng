/**
 * @file
 * @author chris
 * @brief One-way translation from RHI vocabulary enums (@ref veng/rhi/Enums.hpp) to Vulkan.
 *
 * This is the single seam where the engine's intent enums become `vk::` values. It includes
 * `<vulkan/vulkan.hpp>` and is therefore included ONLY by implementation files and the lower
 * construction layers (pipeline builder, resource/sampler creation, node record paths) — never by
 * a public node/pass header. Keeping the include here, and out of @ref veng/rhi/Enums.hpp, is what
 * lets high-level headers describe GPU state without pulling in Vulkan.
 *
 * @ingroup rhi
 */

#ifndef VENG_RHI_CONVERT_HPP
#define VENG_RHI_CONVERT_HPP

#include <veng/rhi/Enums.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::rhi
{
/// @brief Translate an @ref rhi::Format to its `vk::Format`.
[[nodiscard]] constexpr vk::Format to_vk(Format format) noexcept
{
	switch (format)
	{
		case Format::UNDEFINED: return vk::Format::eUndefined;
		case Format::RGBA8_UNORM: return vk::Format::eR8G8B8A8Unorm;
		case Format::RGBA8_SRGB: return vk::Format::eR8G8B8A8Srgb;
		case Format::BGRA8_UNORM: return vk::Format::eB8G8R8A8Unorm;
		case Format::BGRA8_SRGB: return vk::Format::eB8G8R8A8Srgb;
		case Format::RGBA16_SFLOAT: return vk::Format::eR16G16B16A16Sfloat;
		case Format::R32_SFLOAT: return vk::Format::eR32Sfloat;
		case Format::R32_SINT: return vk::Format::eR32Sint;
		case Format::R32_UINT: return vk::Format::eR32Uint;
		case Format::RG32_SFLOAT: return vk::Format::eR32G32Sfloat;
		case Format::RG32_SINT: return vk::Format::eR32G32Sint;
		case Format::RG32_UINT: return vk::Format::eR32G32Uint;
		case Format::RGB32_SFLOAT: return vk::Format::eR32G32B32Sfloat;
		case Format::RGB32_SINT: return vk::Format::eR32G32B32Sint;
		case Format::RGB32_UINT: return vk::Format::eR32G32B32Uint;
		case Format::RGBA32_SFLOAT: return vk::Format::eR32G32B32A32Sfloat;
		case Format::RGBA32_SINT: return vk::Format::eR32G32B32A32Sint;
		case Format::RGBA32_UINT: return vk::Format::eR32G32B32A32Uint;
		case Format::D32_SFLOAT: return vk::Format::eD32Sfloat;
	}
	return vk::Format::eUndefined; // unreachable: all enumerators handled above
}

/// @brief Translate a `vk::Format` back to an @ref rhi::Format (`UNDEFINED` for any unmapped value).
[[nodiscard]] constexpr Format to_rhi(vk::Format format) noexcept
{
	switch (format)
	{
		case vk::Format::eR8G8B8A8Unorm: return Format::RGBA8_UNORM;
		case vk::Format::eR8G8B8A8Srgb: return Format::RGBA8_SRGB;
		case vk::Format::eB8G8R8A8Unorm: return Format::BGRA8_UNORM;
		case vk::Format::eB8G8R8A8Srgb: return Format::BGRA8_SRGB;
		case vk::Format::eR16G16B16A16Sfloat: return Format::RGBA16_SFLOAT;
		case vk::Format::eR32Sfloat: return Format::R32_SFLOAT;
		case vk::Format::eR32Sint: return Format::R32_SINT;
		case vk::Format::eR32Uint: return Format::R32_UINT;
		case vk::Format::eR32G32Sfloat: return Format::RG32_SFLOAT;
		case vk::Format::eR32G32Sint: return Format::RG32_SINT;
		case vk::Format::eR32G32Uint: return Format::RG32_UINT;
		case vk::Format::eR32G32B32Sfloat: return Format::RGB32_SFLOAT;
		case vk::Format::eR32G32B32Sint: return Format::RGB32_SINT;
		case vk::Format::eR32G32B32Uint: return Format::RGB32_UINT;
		case vk::Format::eR32G32B32A32Sfloat: return Format::RGBA32_SFLOAT;
		case vk::Format::eR32G32B32A32Sint: return Format::RGBA32_SINT;
		case vk::Format::eR32G32B32A32Uint: return Format::RGBA32_UINT;
		case vk::Format::eD32Sfloat: return Format::D32_SFLOAT;
		default: return Format::UNDEFINED;
	}
}

/// @brief Translate an @ref rhi::Extent2D to a `vk::Extent2D` (layout-compatible width/height).
[[nodiscard]] constexpr vk::Extent2D to_vk(Extent2D extent) noexcept
{
	return vk::Extent2D{extent.width, extent.height};
}

/// @brief Translate a `vk::Extent2D` to an @ref rhi::Extent2D.
[[nodiscard]] constexpr Extent2D to_rhi(vk::Extent2D extent) noexcept
{
	return Extent2D{extent.width, extent.height};
}

/// @brief Translate an @ref rhi::Topology to its `vk::PrimitiveTopology`.
[[nodiscard]] constexpr vk::PrimitiveTopology to_vk(Topology topology) noexcept
{
	switch (topology)
	{
		case Topology::TRIANGLE_LIST: return vk::PrimitiveTopology::eTriangleList;
		case Topology::TRIANGLE_STRIP: return vk::PrimitiveTopology::eTriangleStrip;
		case Topology::LINE_LIST: return vk::PrimitiveTopology::eLineList;
	}
	return vk::PrimitiveTopology::eTriangleList; // unreachable
}

/// @brief Translate an @ref rhi::CullMode to its `vk::CullModeFlags`.
[[nodiscard]] constexpr vk::CullModeFlags to_vk(CullMode cull) noexcept
{
	switch (cull)
	{
		case CullMode::NONE: return vk::CullModeFlagBits::eNone;
		case CullMode::FRONT: return vk::CullModeFlagBits::eFront;
		case CullMode::BACK: return vk::CullModeFlagBits::eBack;
	}
	return vk::CullModeFlagBits::eBack; // unreachable
}

/// @brief Translate an @ref rhi::FrontFace to its `vk::FrontFace`.
[[nodiscard]] constexpr vk::FrontFace to_vk(FrontFace front) noexcept
{
	switch (front)
	{
		case FrontFace::COUNTER_CLOCKWISE: return vk::FrontFace::eCounterClockwise;
		case FrontFace::CLOCKWISE: return vk::FrontFace::eClockwise;
	}
	return vk::FrontFace::eCounterClockwise; // unreachable
}

/// @brief Translate an @ref rhi::PolygonMode to its `vk::PolygonMode`.
[[nodiscard]] constexpr vk::PolygonMode to_vk(PolygonMode polygon) noexcept
{
	switch (polygon)
	{
		case PolygonMode::FILL: return vk::PolygonMode::eFill;
		case PolygonMode::LINE: return vk::PolygonMode::eLine;
	}
	return vk::PolygonMode::eFill; // unreachable
}

/// @brief Translate an @ref rhi::ShaderStage bitmask to `vk::ShaderStageFlags`.
[[nodiscard]] constexpr vk::ShaderStageFlags to_vk(ShaderStage stage) noexcept
{
	vk::ShaderStageFlags flags{};
	if ((stage & ShaderStage::VERTEX) != ShaderStage{})
	{
		flags |= vk::ShaderStageFlagBits::eVertex;
	}
	if ((stage & ShaderStage::FRAGMENT) != ShaderStage{})
	{
		flags |= vk::ShaderStageFlagBits::eFragment;
	}
	if ((stage & ShaderStage::COMPUTE) != ShaderStage{})
	{
		flags |= vk::ShaderStageFlagBits::eCompute;
	}
	return flags;
}

/// @brief Translate an @ref rhi::SampleCount to its `vk::SampleCountFlagBits`.
[[nodiscard]] constexpr vk::SampleCountFlagBits to_vk(SampleCount samples) noexcept
{
	switch (samples)
	{
		case SampleCount::X1: return vk::SampleCountFlagBits::e1;
		case SampleCount::X2: return vk::SampleCountFlagBits::e2;
		case SampleCount::X4: return vk::SampleCountFlagBits::e4;
		case SampleCount::X8: return vk::SampleCountFlagBits::e8;
		case SampleCount::X16: return vk::SampleCountFlagBits::e16;
		case SampleCount::X32: return vk::SampleCountFlagBits::e32;
		case SampleCount::X64: return vk::SampleCountFlagBits::e64;
	}
	return vk::SampleCountFlagBits::e1; // unreachable
}

/// @brief Translate a `vk::SampleCountFlagBits` back to an @ref rhi::SampleCount (`X1` if unmapped).
///
/// Used where a device-clamped count (Vulkan-side, e.g. @ref veng::clamp_sample_count) is fed back
/// into an RHI-typed builder.
[[nodiscard]] constexpr SampleCount to_rhi(vk::SampleCountFlagBits samples) noexcept
{
	switch (samples)
	{
		case vk::SampleCountFlagBits::e1: return SampleCount::X1;
		case vk::SampleCountFlagBits::e2: return SampleCount::X2;
		case vk::SampleCountFlagBits::e4: return SampleCount::X4;
		case vk::SampleCountFlagBits::e8: return SampleCount::X8;
		case vk::SampleCountFlagBits::e16: return SampleCount::X16;
		case vk::SampleCountFlagBits::e32: return SampleCount::X32;
		case vk::SampleCountFlagBits::e64: return SampleCount::X64;
	}
	return SampleCount::X1;
}

/// @brief Translate an @ref rhi::Filter to its `vk::Filter`.
[[nodiscard]] constexpr vk::Filter to_vk(Filter filter) noexcept
{
	switch (filter)
	{
		case Filter::NEAREST: return vk::Filter::eNearest;
		case Filter::LINEAR: return vk::Filter::eLinear;
	}
	return vk::Filter::eLinear; // unreachable
}

/// @brief Translate an @ref rhi::IndexType to its `vk::IndexType`.
[[nodiscard]] constexpr vk::IndexType to_vk(IndexType type) noexcept
{
	switch (type)
	{
		case IndexType::UINT16: return vk::IndexType::eUint16;
		case IndexType::UINT32: return vk::IndexType::eUint32;
	}
	return vk::IndexType::eUint32; // unreachable
}

/**
 * @brief The concrete Vulkan layout + synchronization scope a @ref rhi::TextureUsage maps to.
 *
 * The RHI owns this table so no high-level code names a `vk::ImageLayout`/stage/access again. A
 * barrier between two usages takes the source scope from the old usage's state and the destination
 * scope from the new usage's state (see @ref veng::rhi::CommandEncoder::transition).
 *
 * @ingroup rhi
 */
struct ImageState
{
	vk::ImageLayout			layout; ///< The image layout the usage requires.
	vk::PipelineStageFlags2 stage;	///< The pipeline stage that performs the access.
	vk::AccessFlags2		access; ///< The access scope (read/write kind).
};

/// @brief Map a @ref rhi::TextureUsage to its concrete layout + synchronization scope.
[[nodiscard]] inline ImageState to_state(TextureUsage usage) noexcept
{
	switch (usage)
	{
		case TextureUsage::UNDEFINED:
			return {vk::ImageLayout::eUndefined, vk::PipelineStageFlagBits2::eTopOfPipe, vk::AccessFlagBits2::eNone};
		case TextureUsage::COLOR_ATTACHMENT:
			return {vk::ImageLayout::eColorAttachmentOptimal, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
					vk::AccessFlagBits2::eColorAttachmentWrite};
		case TextureUsage::DEPTH_ATTACHMENT:
			return {vk::ImageLayout::eDepthAttachmentOptimal,
					vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
					vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
						vk::AccessFlagBits2::eDepthStencilAttachmentRead};
		case TextureUsage::SAMPLED:
			return {vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits2::eFragmentShader,
					vk::AccessFlagBits2::eShaderSampledRead};
		case TextureUsage::TRANSFER_SRC:
			return {vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eTransfer,
					vk::AccessFlagBits2::eTransferRead};
		case TextureUsage::TRANSFER_DST:
			return {vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eTransfer,
					vk::AccessFlagBits2::eTransferWrite};
		case TextureUsage::PRESENT:
			return {vk::ImageLayout::ePresentSrcKHR, vk::PipelineStageFlagBits2::eBottomOfPipe,
					vk::AccessFlagBits2::eNone};
	}
	return {vk::ImageLayout::eUndefined, vk::PipelineStageFlagBits2::eTopOfPipe, vk::AccessFlagBits2::eNone};
}

/// @brief Translate an @ref rhi::MipmapMode to its `vk::SamplerMipmapMode`.
[[nodiscard]] constexpr vk::SamplerMipmapMode to_vk(MipmapMode mode) noexcept
{
	switch (mode)
	{
		case MipmapMode::NEAREST: return vk::SamplerMipmapMode::eNearest;
		case MipmapMode::LINEAR: return vk::SamplerMipmapMode::eLinear;
	}
	return vk::SamplerMipmapMode::eNearest; // unreachable
}

/// @brief Translate an @ref rhi::AddressMode to its `vk::SamplerAddressMode`.
[[nodiscard]] constexpr vk::SamplerAddressMode to_vk(AddressMode mode) noexcept
{
	switch (mode)
	{
		case AddressMode::REPEAT: return vk::SamplerAddressMode::eRepeat;
		case AddressMode::MIRRORED_REPEAT: return vk::SamplerAddressMode::eMirroredRepeat;
		case AddressMode::CLAMP_TO_EDGE: return vk::SamplerAddressMode::eClampToEdge;
		case AddressMode::CLAMP_TO_BORDER: return vk::SamplerAddressMode::eClampToBorder;
	}
	return vk::SamplerAddressMode::eClampToEdge; // unreachable
}
} // namespace veng::rhi

#endif // VENG_RHI_CONVERT_HPP
