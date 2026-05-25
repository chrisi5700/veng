//
// Created by chris on 5/25/26.
//
// L1 resource — RAII, move-only GPU buffer over VMA (design.md §L1). Owns its
// VkBuffer + VmaAllocation and frees both on destruction.
//

#ifndef VENG_BUFFER_HPP
#define VENG_BUFFER_HPP

#include <expected>
#include <vulkan-memory-allocator-hpp/vk_mem_alloc.hpp>
#include <vulkan/vulkan.hpp>

namespace veng
{
class Buffer
{
	 public:
	/// Allocate a buffer of `size` bytes. `memory` defaults to `eAuto` (VMA picks the
	/// best heap from the usage); pass `eAutoPreferHost` + a host-access flag for
	/// staging/uniform buffers you intend to map.
	[[nodiscard]] static std::expected<Buffer, vk::Result> create(vma::Allocator allocator, vk::DeviceSize size,
																  vk::BufferUsageFlags usage,
																  vma::MemoryUsage	   memory = vma::MemoryUsage::eAuto,
																  vma::AllocationCreateFlags flags = {});

	Buffer(const Buffer&)			 = delete;
	Buffer& operator=(const Buffer&) = delete;
	Buffer(Buffer&& other) noexcept;
	Buffer& operator=(Buffer&& other) noexcept;
	~Buffer();

	[[nodiscard]] vk::Buffer	  buffer() const noexcept { return m_buffer; }
	[[nodiscard]] vma::Allocation allocation() const noexcept { return m_allocation; }
	[[nodiscard]] vk::DeviceSize  size() const noexcept { return m_size; }

	 private:
	Buffer(vma::Allocator allocator, vk::Buffer buffer, vma::Allocation allocation, vk::DeviceSize size) noexcept;
	void destroy() noexcept;

	vma::Allocator	m_allocator;
	vk::Buffer		m_buffer;
	vma::Allocation m_allocation;
	vk::DeviceSize	m_size;
};
} // namespace veng

#endif // VENG_BUFFER_HPP
