//
// Created by chris on 5/30/26.
//
// L2 asset — a GPU-resident 2D texture loaded from a file or memory, with a generated mip chain
// (review.md item 3). The one genuinely new piece of machinery for displaying real assets: it
// fills the gap between `ResourcePool::constant_image` (a clear-colour fill) and an actual
// decoded, mipped texture. It owns its backing `Image` and exposes a `gpu::ImageRef` you feed to
// a graph source to sample it.
//
// Immutability: a texture is uploaded once and never written again, so — unlike a render target —
// it is NOT pool-backed (its ref's pool_id is INVALID). One physical copy is safe to sample every
// frame forever, and consumers never `touch` it. Lifetime is the caller's: keep the `Texture`
// alive as long as anything samples its ref.
//

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
/// Whether a texture's bytes are sRGB-encoded colour (baseColor, emissive) or linear data
/// (normal, metallic-roughness, occlusion). Drives the image format so the sampler hardware
/// decodes colour to linear on read and leaves data untouched (review.md item 8). Getting this
/// wrong — decoding a normal map as sRGB — is the most common glTF rendering bug.
enum class ColorSpace : std::uint8_t
{
	Srgb,	// colour: baseColor, emissive  -> eR8G8B8A8Srgb (sampler decodes to linear)
	Linear, // data:   normal, metallic-roughness, occlusion -> eR8G8B8A8Unorm
};

enum class TextureError : std::uint8_t
{
	FileUnreadable, // stb_image could not open/decode the file
	DecodeFailed,	// the bytes were not a recognisable image
	GpuAllocation,	// Image/Buffer allocation failed
	Upload,			// the staging upload / mip generation submit failed
};

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

class Texture
{
	 public:
	/// Decode an image file (PNG/JPG/TGA/… — whatever stb_image supports) to RGBA8 and upload it
	/// with a full mip chain, choosing the image format from `color_space`.
	[[nodiscard]] static std::expected<Texture, TextureError> from_file(const Context& ctx, const std::string& path,
																		ColorSpace color_space);

	/// Upload tightly-packed RGBA8 pixels (`width * height * 4` bytes, row-major). The in-memory
	/// path the glTF loader feeds decoded buffers into, and tests feed synthesized pixels into.
	[[nodiscard]] static std::expected<Texture, TextureError> from_pixels(const Context&			 ctx,
																		  std::span<const std::byte> rgba8,
																		  std::uint32_t width, std::uint32_t height,
																		  ColorSpace color_space);

	/// Decode an in-memory *encoded* image (PNG/JPG/... bytes) to RGBA8 and upload it. The path the
	/// glTF loader uses for textures embedded in a .glb or referenced from a buffer view.
	[[nodiscard]] static std::expected<Texture, TextureError> from_encoded(const Context&			  ctx,
																		   std::span<const std::byte> encoded,
																		   ColorSpace				  color_space);

	Texture(const Texture&)			   = delete;
	Texture& operator=(const Texture&) = delete;
	Texture(Texture&&)				   = default;
	Texture& operator=(Texture&&)	   = default;
	~Texture()						   = default;

	/// The edge value to feed a graph source. pool_id is INVALID (not pool-owned); version is a
	/// fixed 1 (the texture is immutable, so it changes exactly once — cold — then caches).
	[[nodiscard]] gpu::ImageRef ref() const noexcept
	{
		return gpu::ImageRef{.image	  = m_image.image(),
							 .view	  = m_image.view(),
							 .extent  = m_image.extent(),
							 .format  = m_image.format(),
							 .pool_id = gpu::ImageRef::INVALID_POOL_ID,
							 .version = 1};
	}

	[[nodiscard]] const Image&	image() const noexcept { return m_image; }
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
