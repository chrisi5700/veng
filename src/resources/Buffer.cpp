//
// Created by chris on 5/25/26.
//
// See Buffer.hpp and design.md §L1.
//

#include <utility>
#include <veng/resources/Buffer.hpp>

namespace veng
{
std::expected<Buffer, vk::Result> Buffer::create(vma::Allocator allocator, vk::DeviceSize size,
												 vk::BufferUsageFlags usage, vma::MemoryUsage memory,
												 vma::AllocationCreateFlags flags)
{
	const auto buffer_info =
		vk::BufferCreateInfo().setSize(size).setUsage(usage).setSharingMode(vk::SharingMode::eExclusive);

	vma::AllocationCreateInfo alloc_info{};
	alloc_info.usage = memory;
	alloc_info.flags = flags;

	vk::Buffer			buffer;
	vma::Allocation		allocation;
	vma::AllocationInfo allocation_info{};
	const vk::Result result = allocator.createBuffer(&buffer_info, &alloc_info, &buffer, &allocation, &allocation_info);
	if (result != vk::Result::eSuccess)
	{
		return std::unexpected(result);
	}
	// pMappedData is non-null only when the eMapped flag was set on a host-visible heap.
	return Buffer(allocator, buffer, allocation, size, allocation_info.pMappedData);
}

Buffer::Buffer(vma::Allocator allocator, vk::Buffer buffer, vma::Allocation allocation, vk::DeviceSize size,
			   void* mapped) noexcept
	: m_allocator(allocator)
	, m_buffer(buffer)
	, m_allocation(allocation)
	, m_size(size)
	, m_mapped(mapped)
{
}

void Buffer::destroy() noexcept
{
	if (m_buffer)
	{
		m_allocator.destroyBuffer(m_buffer, m_allocation);
	}
	m_buffer	 = nullptr;
	m_allocation = nullptr;
	m_allocator	 = nullptr;
	m_size		 = 0;
	m_mapped	 = nullptr;
}

Buffer::Buffer(Buffer&& other) noexcept
	: m_allocator(other.m_allocator)
	, m_buffer(other.m_buffer)
	, m_allocation(other.m_allocation)
	, m_size(other.m_size)
	, m_mapped(other.m_mapped)
{
	other.m_buffer	   = nullptr;
	other.m_allocation = nullptr;
	other.m_allocator  = nullptr;
	other.m_size	   = 0;
	other.m_mapped	   = nullptr;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept
{
	if (this != &other)
	{
		destroy();
		m_allocator		   = other.m_allocator;
		m_buffer		   = other.m_buffer;
		m_allocation	   = other.m_allocation;
		m_size			   = other.m_size;
		m_mapped		   = other.m_mapped;
		other.m_buffer	   = nullptr;
		other.m_allocation = nullptr;
		other.m_allocator  = nullptr;
		other.m_size	   = 0;
		other.m_mapped	   = nullptr;
	}
	return *this;
}

Buffer::~Buffer()
{
	destroy();
}
} // namespace veng
