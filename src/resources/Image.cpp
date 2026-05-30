//
// Created by chris on 5/25/26.
//
// See Image.hpp and design.md §L1.
//

#include <utility>
#include <veng/resources/Image.hpp>

namespace veng
{
std::expected<Image, vk::Result> Image::create(vma::Allocator allocator, vk::Device device, vk::Extent2D extent,
											   vk::Format format, vk::ImageUsageFlags usage,
											   vk::ImageAspectFlags aspect, std::uint32_t mip_levels)
{
	const std::uint32_t levels	   = mip_levels == 0 ? 1 : mip_levels;
	const auto			image_info = vk::ImageCreateInfo()
										 .setImageType(vk::ImageType::e2D)
										 .setFormat(format)
										 .setExtent(vk::Extent3D{extent.width, extent.height, 1})
										 .setMipLevels(levels)
										 .setArrayLayers(1)
										 .setSamples(vk::SampleCountFlagBits::e1)
										 .setTiling(vk::ImageTiling::eOptimal)
										 .setUsage(usage)
										 .setSharingMode(vk::SharingMode::eExclusive)
										 .setInitialLayout(vk::ImageLayout::eUndefined);

	vma::AllocationCreateInfo alloc_info{};
	alloc_info.usage = vma::MemoryUsage::eAutoPreferDevice;

	vk::Image		 image;
	vma::Allocation	 allocation;
	const vk::Result result = allocator.createImage(&image_info, &alloc_info, &image, &allocation, nullptr);
	if (result != vk::Result::eSuccess)
	{
		return std::unexpected(result);
	}

	// Only create a view when the usage permits one (vkCreateImageView requires a
	// view-capable usage). A transfer-only image (e.g. a present/blit target) gets no
	// view; `view()` then returns a null handle.
	constexpr vk::ImageUsageFlags VIEW_USAGES =
		vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eColorAttachment |
		vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eInputAttachment |
		vk::ImageUsageFlagBits::eTransientAttachment;

	vk::ImageView view{};
	if (usage & VIEW_USAGES)
	{
		const auto view_info   = vk::ImageViewCreateInfo()
									 .setImage(image)
									 .setViewType(vk::ImageViewType::e2D)
									 .setFormat(format)
									 .setSubresourceRange(vk::ImageSubresourceRange()
															  .setAspectMask(aspect)
															  .setBaseMipLevel(0)
															  .setLevelCount(levels)
															  .setBaseArrayLayer(0)
															  .setLayerCount(1));
		const auto view_result = device.createImageView(view_info);
		if (view_result.result != vk::Result::eSuccess)
		{
			allocator.destroyImage(image, allocation);
			return std::unexpected(view_result.result);
		}
		view = view_result.value;
	}

	return Image(allocator, device, image, allocation, view, format, extent, levels);
}

Image::Image(vma::Allocator allocator, vk::Device device, vk::Image image, vma::Allocation allocation,
			 vk::ImageView view, vk::Format format, vk::Extent2D extent, std::uint32_t mip_levels) noexcept
	: m_allocator(allocator)
	, m_device(device)
	, m_image(image)
	, m_allocation(allocation)
	, m_view(view)
	, m_format(format)
	, m_extent(extent)
	, m_mip_levels(mip_levels)
{
}

void Image::destroy() noexcept
{
	if (m_image)
	{
		if (m_view)
		{
			m_device.destroyImageView(m_view);
		}
		m_allocator.destroyImage(m_image, m_allocation);
	}
	m_image		 = nullptr;
	m_view		 = nullptr;
	m_allocation = nullptr;
	m_allocator	 = nullptr;
	m_device	 = nullptr;
}

Image::Image(Image&& other) noexcept
	: m_allocator(other.m_allocator)
	, m_device(other.m_device)
	, m_image(other.m_image)
	, m_allocation(other.m_allocation)
	, m_view(other.m_view)
	, m_format(other.m_format)
	, m_extent(other.m_extent)
	, m_mip_levels(other.m_mip_levels)
{
	other.m_image	   = nullptr;
	other.m_view	   = nullptr;
	other.m_allocation = nullptr;
	other.m_allocator  = nullptr;
	other.m_device	   = nullptr;
}

Image& Image::operator=(Image&& other) noexcept
{
	if (this != &other)
	{
		destroy();
		m_allocator		   = other.m_allocator;
		m_device		   = other.m_device;
		m_image			   = other.m_image;
		m_allocation	   = other.m_allocation;
		m_view			   = other.m_view;
		m_format		   = other.m_format;
		m_extent		   = other.m_extent;
		m_mip_levels	   = other.m_mip_levels;
		other.m_image	   = nullptr;
		other.m_view	   = nullptr;
		other.m_allocation = nullptr;
		other.m_allocator  = nullptr;
		other.m_device	   = nullptr;
	}
	return *this;
}

Image::~Image()
{
	destroy();
}
} // namespace veng
