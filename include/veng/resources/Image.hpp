//
// Created by chris on 5/25/26.
//
// L1 resource — RAII, move-only 2D GPU image over VMA, bundled with a default view
// (design.md §L1). Owns VkImage + VmaAllocation + VkImageView and frees all three.
//

#ifndef VENG_IMAGE_HPP
#define VENG_IMAGE_HPP

#include <cstdint>
#include <expected>
#include <vulkan-memory-allocator-hpp/vk_mem_alloc.hpp>
#include <vulkan/vulkan.hpp>

namespace veng
{
class Image
{
	 public:
	/// Allocate a device-local 2D image (`mip_levels` mips, 1 layer, 1 sample) and a matching
	/// view spanning all mips. `aspect` selects the view aspect (color/depth). `mip_levels` > 1
	/// is for sampled textures whose mip chain is generated after upload (a blit-down pyramid);
	/// the usage must then include eTransferSrc + eTransferDst so the generator can blit between
	/// levels. The default (1) preserves the render-target / single-level behaviour.
	[[nodiscard]] static std::expected<Image, vk::Result> create(
		vma::Allocator allocator, vk::Device device, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage,
		vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor, std::uint32_t mip_levels = 1);

	Image(const Image&)			   = delete;
	Image& operator=(const Image&) = delete;
	Image(Image&& other) noexcept;
	Image& operator=(Image&& other) noexcept;
	~Image();

	[[nodiscard]] vk::Image		image() const noexcept { return m_image; }
	[[nodiscard]] vk::ImageView view() const noexcept { return m_view; }
	[[nodiscard]] vk::Format	format() const noexcept { return m_format; }
	[[nodiscard]] vk::Extent2D	extent() const noexcept { return m_extent; }
	[[nodiscard]] std::uint32_t mip_levels() const noexcept { return m_mip_levels; }

	 private:
	Image(vma::Allocator allocator, vk::Device device, vk::Image image, vma::Allocation allocation, vk::ImageView view,
		  vk::Format format, vk::Extent2D extent, std::uint32_t mip_levels) noexcept;
	void destroy() noexcept;

	vma::Allocator	m_allocator;
	vk::Device		m_device;
	vk::Image		m_image;
	vma::Allocation m_allocation;
	vk::ImageView	m_view;
	vk::Format		m_format;
	vk::Extent2D	m_extent;
	std::uint32_t	m_mip_levels;
};
} // namespace veng

#endif // VENG_IMAGE_HPP
