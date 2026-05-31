/**
 * @file
 * @author chris
 * @brief RAII, move-only GPU buffer backed by VMA.
 *
 * L1 resource that owns a `VkBuffer` and a `VmaAllocation` and frees both on destruction.
 * Allocation is done through @ref veng::Buffer::create; the resulting object is non-copyable and
 * must be moved to transfer ownership (or held inside a @ref veng::ResourcePool copy slot).
 *
 * @ingroup resources
 */

#ifndef VENG_BUFFER_HPP
#define VENG_BUFFER_HPP

#include <expected>
#include <vulkan-memory-allocator-hpp/vk_mem_alloc.hpp>
#include <vulkan/vulkan.hpp>

namespace veng
{
/**
 * @brief RAII, move-only GPU buffer backed by a VMA allocation.
 * @ingroup resources
 * @see ResourcePool
 */
class Buffer
{
	 public:
	/**
	 * @brief Allocate a buffer of `size` bytes with the given usage and memory strategy.
	 *
	 * `memory` defaults to `eAuto` (VMA picks the best heap from the usage flags); pass
	 * `eAutoPreferHost` combined with a host-access flag for staging or uniform buffers
	 * you intend to map.
	 *
	 * @param allocator The VMA allocator to allocate from.
	 * @param size      Number of bytes to allocate.
	 * @param usage     Vulkan buffer-usage flags (vertex, index, uniform, storage, …).
	 * @param memory    VMA memory-usage hint controlling which heap to prefer.
	 * @param flags     VMA allocation-create flags (e.g. `eMapped`, `eHostAccessSequentialWrite`).
	 * @return The constructed @ref veng::Buffer on success, or the `vk::Result` error code on failure.
	 */
	[[nodiscard]] static std::expected<Buffer, vk::Result> create(vma::Allocator allocator, vk::DeviceSize size,
																  vk::BufferUsageFlags usage,
																  vma::MemoryUsage	   memory = vma::MemoryUsage::eAuto,
																  vma::AllocationCreateFlags flags = {});

	Buffer(const Buffer&)			 = delete;
	Buffer& operator=(const Buffer&) = delete;
	Buffer(Buffer&& other) noexcept;
	Buffer& operator=(Buffer&& other) noexcept;
	~Buffer();

	/// @brief The underlying Vulkan buffer handle.
	[[nodiscard]] vk::Buffer buffer() const noexcept { return m_buffer; }
	/// @brief The VMA allocation backing this buffer.
	[[nodiscard]] vma::Allocation allocation() const noexcept { return m_allocation; }
	/// @brief The allocated size in bytes.
	[[nodiscard]] vk::DeviceSize size() const noexcept { return m_size; }

	/**
	 * @brief Persistently-mapped host pointer, or `nullptr` if not host-visible/mapped.
	 *
	 * Non-null only when created with `vma::AllocationCreateFlagBits::eMapped` on a
	 * host-visible allocation (e.g. staging or uniform buffers).
	 *
	 * @return A pointer to the persistently-mapped region, or `nullptr`.
	 */
	[[nodiscard]] void* mapped() const noexcept { return m_mapped; }

	 private:
	Buffer(vma::Allocator allocator, vk::Buffer buffer, vma::Allocation allocation, vk::DeviceSize size,
		   void* mapped) noexcept;
	void destroy() noexcept;

	vma::Allocator	m_allocator;
	vk::Buffer		m_buffer;
	vma::Allocation m_allocation;
	vk::DeviceSize	m_size;
	void*			m_mapped;
};
} // namespace veng

#endif // VENG_BUFFER_HPP
