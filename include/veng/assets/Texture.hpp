/**
 * @file
 * @author chris
 * @brief GPU-resident 2D texture loaded from a file or memory buffer, with a full mip chain.
 *
 * @ref veng::assets::Texture fills the gap between `ResourcePool::constant_image` (a clear-colour fill)
 * and an actual decoded, mipped texture. It owns its backing `Image` and exposes a
 * `gpu::ImageRef` that is fed to a graph source node for sampling.
 *
 * Immutability: a texture is uploaded once and never written again. Unlike a render
 * target, it is NOT pool-backed — its ref's `pool_id` is `INVALID`. One physical copy
 * is safe to sample every frame forever and consumers never `touch` it. Lifetime is the
 * caller's responsibility: keep the @ref veng::assets::Texture alive as long as anything samples its ref.
 *
 * @ingroup assets
 */

#ifndef VENG_TEXTURE_HPP
#define VENG_TEXTURE_HPP

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <veng/gpu/ImageRef.hpp>
#include <veng/resources/Image.hpp>
#include <vulkan/vulkan.hpp>

namespace veng
{
class Context;
}

namespace veng::assets
{
/**
 * @brief Whether a texture's bytes are sRGB-encoded colour or linear data.
 *
 * Drives the chosen `vk::Format` so the sampler hardware decodes colour to linear on
 * read and leaves linear data untouched. Getting this wrong — e.g. decoding a normal
 * map as sRGB — is the most common glTF rendering artefact.
 *
 * @ingroup assets
 */
enum class ColorSpace : std::uint8_t
{
	Srgb,	///< Colour data (baseColor, emissive) → `eR8G8B8A8Srgb`; decoded to linear by the sampler.
	Linear, ///< Linear data (normal, metallic-roughness, occlusion) → `eR8G8B8A8Unorm`.
};

/**
 * @brief Error codes returned by @ref veng::assets::Texture factory methods.
 * @ingroup assets
 */
enum class TextureError : std::uint8_t
{
	FileUnreadable, ///< `stb_image` could not open or decode the file.
	DecodeFailed,	///< The bytes were not a recognisable image format.
	GpuAllocation,	///< `Image` or staging `Buffer` allocation failed.
	Upload,			///< The staging upload or mip-generation submit failed.
};

/**
 * @brief Stringify a @ref TextureError for logging and error reporting.
 * @param error The error code to render.
 * @return A human-readable description of the error.
 */
[[nodiscard]] constexpr std::string_view to_string(TextureError error) noexcept
{
	switch (error)
	{
		case TextureError::FileUnreadable: return "texture file could not be read";
		case TextureError::DecodeFailed: return "texture bytes could not be decoded";
		case TextureError::GpuAllocation: return "texture GPU allocation failed";
		case TextureError::Upload: return "texture upload / mip generation failed";
	}
	return "unknown texture error";
}

/**
 * @brief RAII, move-only GPU-resident 2D texture with a full mip chain.
 *
 * Three factory paths cover the loading scenarios that arise in practice: filesystem
 * files (`from_file`), pre-decoded RGBA8 pixel buffers (`from_pixels`), and
 * in-memory encoded image bytes (`from_encoded`). All three produce the same immutable,
 * pool-independent result.
 *
 * @ingroup assets
 * @see GltfLoader
 */
class Texture
{
	 public:
	/**
	 * @brief Decode an image file to RGBA8 and upload it with a full mip chain.
	 *
	 * Supports any format stb_image recognises (PNG, JPG, TGA, BMP, …).
	 *
	 * @param ctx         The engine context providing the Vulkan device and allocator.
	 * @param path        Filesystem path to the image file.
	 * @param color_space Whether the image contains sRGB colour or linear data.
	 * @return The uploaded @ref veng::assets::Texture, or a @ref TextureError on failure.
	 */
	[[nodiscard]] static std::expected<Texture, TextureError> from_file(const Context& ctx, const std::string& path,
																		ColorSpace color_space);

	/**
	 * @brief Upload tightly-packed RGBA8 pixels to the GPU with a full mip chain.
	 *
	 * The path the @ref veng::assets::load_gltf feeds decoded buffers into, and that tests use with
	 * synthesized pixel data. Pixels are `width * height * 4` bytes, row-major.
	 *
	 * @param ctx         The engine context.
	 * @param rgba8       Tightly-packed RGBA8 pixel data.
	 * @param width       Image width in pixels.
	 * @param height      Image height in pixels.
	 * @param color_space Whether the pixels are sRGB colour or linear data.
	 * @return The uploaded @ref veng::assets::Texture, or a @ref TextureError on failure.
	 */
	[[nodiscard]] static std::expected<Texture, TextureError> from_pixels(const Context&			 ctx,
																		  std::span<const std::byte> rgba8,
																		  std::uint32_t width, std::uint32_t height,
																		  ColorSpace color_space);

	/**
	 * @brief Decode in-memory encoded image bytes (PNG/JPG/…) to RGBA8 and upload with mips.
	 *
	 * The path the @ref veng::assets::load_gltf uses for textures embedded in a `.glb` or referenced
	 * from a buffer view.
	 *
	 * @param ctx         The engine context.
	 * @param encoded     Encoded image bytes (any format stb_image recognises).
	 * @param color_space Whether the image contains sRGB colour or linear data.
	 * @return The uploaded @ref veng::assets::Texture, or a @ref TextureError on failure.
	 */
	[[nodiscard]] static std::expected<Texture, TextureError> from_encoded(const Context&			  ctx,
																		   std::span<const std::byte> encoded,
																		   ColorSpace				  color_space);

	Texture(const Texture&)			   = delete;
	Texture& operator=(const Texture&) = delete;
	Texture(Texture&&)				   = default;
	Texture& operator=(Texture&&)	   = default;
	~Texture()						   = default;

	/**
	 * @brief The `gpu::ImageRef` to feed to a graph source node.
	 *
	 * `pool_id` is `INVALID` (not pool-owned); `version` is a fixed `1` because the
	 * texture is immutable — it changes exactly once (the upload) and then caches.
	 *
	 * @return A `gpu::ImageRef` valid for the lifetime of this @ref veng::assets::Texture.
	 */
	[[nodiscard]] gpu::ImageRef ref() const noexcept
	{
		return gpu::ImageRef{.image	  = m_image.image(),
							 .view	  = m_image.view(),
							 .extent  = m_image.extent(),
							 .format  = m_image.format(),
							 .pool_id = gpu::ImageRef::INVALID_POOL_ID,
							 .version = 1};
	}

	/** @brief The underlying RAII `Image` object. */
	[[nodiscard]] const Image& image() const noexcept { return m_image; }
	/** @brief Number of mip levels in the uploaded image. */
	[[nodiscard]] std::uint32_t mip_levels() const noexcept { return m_image.mip_levels(); }

	 private:
	explicit Texture(Image image) noexcept
		: m_image(std::move(image))
	{
	}

	Image m_image;
};
} // namespace veng::assets

#endif // VENG_TEXTURE_HPP
