/**
 * @file
 * @author chris
 * @brief Engine-owned RHI vocabulary enums — the value types high-level code uses to describe
 *        intent without naming Vulkan.
 *
 * These are plain "intent" enums: pixel formats, primitive topology, cull/face/polygon state,
 * shader stages, sample counts, and sampler filtering. They carry no Vulkan dependency — this
 * header includes nothing from `<vulkan/...>` — so a node, pass, or pipeline description can spell
 * `rhi::Format::RGBA16_SFLOAT` or `rhi::CullMode::BACK` in its public API and a caller never types
 * `vk::`. The one-way translation to Vulkan lives in @ref veng/rhi/Convert.hpp, included only by
 * implementation files and the lower construction layers — never by a public node/pass header.
 *
 * This is the vocabulary half of the RHI containment effort (Stage 0): it seals the most visible
 * leak — Vulkan enums surfacing in the "extremely abstracted" pass/node config — while opaque
 * handles, the command encoder, and automatic sync land in later stages.
 *
 * @ingroup rhi
 */

#ifndef VENG_RHI_ENUMS_HPP
#define VENG_RHI_ENUMS_HPP

#include <cstdint>

namespace veng::rhi
{
/**
 * @brief Pixel/vertex-attribute formats the engine understands.
 *
 * Covers the attachment formats high-level code selects (color, HDR, depth, integer picking
 * targets) plus the float/int vertex-attribute formats SPIR-V reflection maps onto. Names follow
 * the channel-then-numeric-type convention (`RGBA16_SFLOAT` = four 16-bit float channels).
 *
 * @ingroup rhi
 */
enum class Format : std::uint8_t
{
	UNDEFINED, ///< No/unknown format (e.g. "no depth attachment").

	RGBA8_UNORM, ///< 4×8-bit unsigned-normalized; the standard LDR color target.
	RGBA8_SRGB,	 ///< 4×8-bit sRGB-encoded color.
	BGRA8_UNORM, ///< 4×8-bit BGRA unsigned-normalized.
	BGRA8_SRGB,	 ///< 4×8-bit BGRA sRGB — the engine's swapchain surface format.

	RGBA16_SFLOAT, ///< 4×16-bit float; the standard HDR color target.

	R32_SFLOAT,	   ///< 1×32-bit float.
	R32_SINT,	   ///< 1×32-bit signed int (e.g. an object-id picking target).
	R32_UINT,	   ///< 1×32-bit unsigned int.
	RG32_SFLOAT,   ///< 2×32-bit float.
	RG32_SINT,	   ///< 2×32-bit signed int.
	RG32_UINT,	   ///< 2×32-bit unsigned int.
	RGB32_SFLOAT,  ///< 3×32-bit float (common vertex position/normal attribute).
	RGB32_SINT,	   ///< 3×32-bit signed int.
	RGB32_UINT,	   ///< 3×32-bit unsigned int.
	RGBA32_SFLOAT, ///< 4×32-bit float.
	RGBA32_SINT,   ///< 4×32-bit signed int.
	RGBA32_UINT,   ///< 4×32-bit unsigned int.

	D32_SFLOAT, ///< 1×32-bit float depth.
};

/**
 * @brief Primitive assembly topology.
 * @ingroup rhi
 */
enum class Topology : std::uint8_t
{
	TRIANGLE_LIST,	///< Independent triangles (the default).
	TRIANGLE_STRIP, ///< Triangle strip.
	LINE_LIST,		///< Independent line segments (debug-line rendering).
};

/**
 * @brief Face culling mode.
 * @ingroup rhi
 */
enum class CullMode : std::uint8_t
{
	NONE,  ///< Cull nothing.
	FRONT, ///< Cull front faces.
	BACK,  ///< Cull back faces (the default).
};

/**
 * @brief Winding order that defines a front face.
 * @ingroup rhi
 */
enum class FrontFace : std::uint8_t
{
	COUNTER_CLOCKWISE, ///< CCW is front (glTF / the engine default).
	CLOCKWISE,		   ///< CW is front.
};

/**
 * @brief Polygon rasterization mode.
 * @ingroup rhi
 */
enum class PolygonMode : std::uint8_t
{
	FILL, ///< Filled polygons (the default).
	LINE, ///< Wireframe edges.
};

/**
 * @brief Shader stages a resource or push-constant range is visible to.
 *
 * A bitmask: combine with `operator|` (e.g. `ShaderStage::VERTEX | ShaderStage::FRAGMENT`).
 *
 * @ingroup rhi
 */
enum class ShaderStage : std::uint32_t
{
	VERTEX	 = 1U << 0U, ///< Vertex stage.
	FRAGMENT = 1U << 1U, ///< Fragment stage.
	COMPUTE	 = 1U << 2U, ///< Compute stage.
};

/// @brief Bitwise-or two stage masks.
[[nodiscard]] constexpr ShaderStage operator|(ShaderStage lhs, ShaderStage rhs) noexcept
{
	return static_cast<ShaderStage>(static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs));
}

/// @brief Bitwise-and two stage masks (test membership: `(s & ShaderStage::VERTEX) != ShaderStage{}`).
[[nodiscard]] constexpr ShaderStage operator&(ShaderStage lhs, ShaderStage rhs) noexcept
{
	return static_cast<ShaderStage>(static_cast<std::uint32_t>(lhs) & static_cast<std::uint32_t>(rhs));
}

/**
 * @brief Rasterization/attachment sample count.
 *
 * The underlying value is the literal sample count, so it can be compared and clamped
 * arithmetically (`static_cast<std::uint32_t>(samples)`).
 *
 * @ingroup rhi
 */
enum class SampleCount : std::uint32_t
{
	X1	= 1,  ///< No MSAA (the default).
	X2	= 2,  ///< 2× MSAA.
	X4	= 4,  ///< 4× MSAA.
	X8	= 8,  ///< 8× MSAA.
	X16 = 16, ///< 16× MSAA.
	X32 = 32, ///< 32× MSAA.
	X64 = 64, ///< 64× MSAA.
};

/**
 * @brief Sampler magnification/minification filter.
 * @ingroup rhi
 */
enum class Filter : std::uint8_t
{
	NEAREST, ///< Nearest-texel sampling.
	LINEAR,	 ///< Linear (bilinear) sampling.
};

/**
 * @brief Sampler mip-level selection mode.
 * @ingroup rhi
 */
enum class MipmapMode : std::uint8_t
{
	NEAREST, ///< Pick the nearest mip level.
	LINEAR,	 ///< Linearly blend between mip levels (trilinear).
};

/**
 * @brief Sampler texture-coordinate address mode.
 * @ingroup rhi
 */
enum class AddressMode : std::uint8_t
{
	REPEAT,			 ///< Tile the texture.
	MIRRORED_REPEAT, ///< Tile with mirroring.
	CLAMP_TO_EDGE,	 ///< Clamp to the edge texel.
	CLAMP_TO_BORDER, ///< Clamp to a border color.
};

/**
 * @brief How a texture is being used at a point in a frame — the engine-vocabulary stand-in for
 *        the `vk::ImageLayout`/`PipelineStage`/`Access` triple.
 *
 * A node declares the usage it needs (via `GpuNode::image_usages`) or asks the command encoder to
 * transition to one (`CommandEncoder::transition`); the RHI maps each usage to the concrete layout,
 * pipeline stage, and access scope. No high-level code names a Vulkan image layout again.
 *
 * @ingroup rhi
 */
enum class TextureUsage : std::uint8_t
{
	UNDEFINED,		  ///< No defined contents/layout — the "discard prior contents" source state.
	COLOR_ATTACHMENT, ///< Written as a color attachment by a render pass.
	DEPTH_ATTACHMENT, ///< Read/written as a depth-stencil attachment.
	SAMPLED,		  ///< Sampled (read) by a shader.
	TRANSFER_SRC,	  ///< Read by a transfer (blit/copy) as the source.
	TRANSFER_DST,	  ///< Written by a transfer (blit/copy/clear) as the destination.
	PRESENT,		  ///< Handed to the presentation engine.
};

/**
 * @brief Index element width for an indexed draw — the stand-in for `vk::IndexType`.
 * @ingroup rhi
 */
enum class IndexType : std::uint8_t
{
	UINT16, ///< 16-bit indices.
	UINT32, ///< 32-bit indices.
};

/**
 * @brief What a texture may be used for over its lifetime — the creation-time capability bitmask
 *        (the stand-in for `vk::ImageUsageFlags`).
 *
 * Distinct from @ref TextureUsage, which is the transient *state* a texture is in at one point in a
 * frame (for barriers). This is the set of things it is *allowed* to be, fixed at creation. Combine
 * with `operator|` (e.g. `TextureUsageFlags::COLOR_ATTACHMENT | TextureUsageFlags::TRANSFER_SRC`).
 *
 * @ingroup rhi
 */
enum class TextureUsageFlags : std::uint32_t
{
	NONE			 = 0,
	SAMPLED			 = 1U << 0U, ///< Sampled by a shader.
	COLOR_ATTACHMENT = 1U << 1U, ///< Rendered into as a color attachment.
	DEPTH_ATTACHMENT = 1U << 2U, ///< Used as a depth-stencil attachment.
	TRANSFER_SRC	 = 1U << 3U, ///< Read by a transfer (blit/copy source).
	TRANSFER_DST	 = 1U << 4U, ///< Written by a transfer (blit/copy/clear destination).
};

/// @brief Bitwise-or two texture-usage masks.
[[nodiscard]] constexpr TextureUsageFlags operator|(TextureUsageFlags lhs, TextureUsageFlags rhs) noexcept
{
	return static_cast<TextureUsageFlags>(static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs));
}
/// @brief Bitwise-and two texture-usage masks (test membership against `TextureUsageFlags::NONE`).
[[nodiscard]] constexpr TextureUsageFlags operator&(TextureUsageFlags lhs, TextureUsageFlags rhs) noexcept
{
	return static_cast<TextureUsageFlags>(static_cast<std::uint32_t>(lhs) & static_cast<std::uint32_t>(rhs));
}

/**
 * @brief What a buffer may be used for — the creation-time capability bitmask (stand-in for
 *        `vk::BufferUsageFlags`). Combine with `operator|`.
 * @ingroup rhi
 */
enum class BufferUsageFlags : std::uint32_t
{
	NONE		 = 0,
	VERTEX		 = 1U << 0U, ///< Bound as a vertex buffer.
	INDEX		 = 1U << 1U, ///< Bound as an index buffer.
	UNIFORM		 = 1U << 2U, ///< Bound as a uniform buffer.
	STORAGE		 = 1U << 3U, ///< Bound as a storage buffer (SSBO).
	TRANSFER_SRC = 1U << 4U, ///< Read by a transfer (copy source).
	TRANSFER_DST = 1U << 5U, ///< Written by a transfer (copy destination).
};

/// @brief Bitwise-or two buffer-usage masks.
[[nodiscard]] constexpr BufferUsageFlags operator|(BufferUsageFlags lhs, BufferUsageFlags rhs) noexcept
{
	return static_cast<BufferUsageFlags>(static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs));
}
/// @brief Bitwise-and two buffer-usage masks (test membership against `BufferUsageFlags::NONE`).
[[nodiscard]] constexpr BufferUsageFlags operator&(BufferUsageFlags lhs, BufferUsageFlags rhs) noexcept
{
	return static_cast<BufferUsageFlags>(static_cast<std::uint32_t>(lhs) & static_cast<std::uint32_t>(rhs));
}

/**
 * @brief Where a buffer's memory lives / how the host reaches it.
 * @ingroup rhi
 */
enum class MemoryAccess : std::uint8_t
{
	GPU_ONLY,	  ///< Device-local; not host-visible (the fast default for GPU-resident data).
	HOST_VISIBLE, ///< Host-visible and persistently mapped (uploads / read-backs); see `Device::mapped`.
};

/**
 * @brief What happens to an attachment's existing contents at the start of a render pass — the
 *        stand-in for `vk::AttachmentLoadOp`.
 * @ingroup rhi
 */
enum class LoadOp : std::uint8_t
{
	CLEAR,	   ///< Clear to a provided value.
	LOAD,	   ///< Preserve the existing contents.
	DONT_CARE, ///< Contents are undefined (a full overwrite follows).
};

/**
 * @brief A 2D pixel size — the engine-vocabulary stand-in for `vk::Extent2D`.
 *
 * Flows on the reactive graph's screen-size edge and rides on `ImageRef` so neither names Vulkan.
 * Layout-compatible with `vk::Extent2D`; convert at the driver boundary with @ref veng::rhi::to_vk.
 *
 * @ingroup rhi
 */
struct Extent2D
{
	std::uint32_t width													= 0;
	std::uint32_t height												= 0;
	friend bool	  operator==(const Extent2D&, const Extent2D&) noexcept = default;
};
} // namespace veng::rhi

#endif // VENG_RHI_ENUMS_HPP
