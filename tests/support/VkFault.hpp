//
// Fault-injection seam for the test suite.
//
// veng reaches Vulkan two ways, and both share one swappable table of function pointers:
//   * vulkan-hpp routes every `device.createX()` through the global dynamic dispatcher
//     (`VULKAN_HPP_DEFAULT_DISPATCHER`, a process-global whose `PFN_*` members are public).
//   * VMA is seeded from that same dispatcher via `vma::functionsFromDispatcher` (see
//     Context.cpp), and snapshots the pointers into the allocator at creation time.
//
// So inducing a Vulkan failure needs no production change and no mock device — just a doctored
// function pointer. Two tools cover the two layers:
//
//   * ScopedDispatchFault — RAII swap of one entry point in the global dispatcher, for calls that
//     go straight through `vk::Device` (descriptor pools, pipelines, shader modules, image views,
//     fences, command pools). Restores the original on scope exit, so tests stay isolated; Catch2
//     runs serially in-process, so there is no cross-test race.
//
//   * FailingAllocator — a real `vma::Allocator` built from a live Context but with one allocation
//     entry point forced to fail. Because VMA snapshots its table at creation, the failure must be
//     baked in up front; the constructors at L1 (`Buffer::create` / `Image::create`) already take
//     the allocator by parameter, so this is pure dependency injection through the existing API.
//
// Both report `vk::Result::eErrorOutOfDeviceMemory`, the canonical out-of-memory failure the L1/L2
// `std::expected` error branches are written against.
//

#ifndef VENG_TESTS_SUPPORT_VKFAULT_HPP
#define VENG_TESTS_SUPPORT_VKFAULT_HPP

#include <veng/context/Context.hpp>
#include <vulkan-memory-allocator-hpp/vk_mem_alloc.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::test
{
/**
 * @brief RAII swap of a single function pointer in the global Vulkan dynamic dispatcher.
 *
 * Construct with a reference to one `PFN_*` slot of `VULKAN_HPP_DEFAULT_DISPATCHER` and a stub to
 * install; the original is restored on destruction. Scope the guard tightly around the one
 * operation under test so unrelated Vulkan calls keep working.
 *
 * @code
 * veng::test::ScopedDispatchFault fault{
 *     VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDescriptorPool,
 *     +[](VkDevice, const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*,
 *         VkDescriptorPool*) -> VkResult { return VK_ERROR_OUT_OF_DEVICE_MEMORY; }};
 * @endcode
 */
template <typename Pfn>
class ScopedDispatchFault
{
	 public:
	ScopedDispatchFault(Pfn& slot, Pfn stub) noexcept
		: m_slot(slot)
		, m_saved(slot)
	{
		m_slot = stub;
	}
	~ScopedDispatchFault() { m_slot = m_saved; }

	ScopedDispatchFault(const ScopedDispatchFault&)			   = delete;
	ScopedDispatchFault& operator=(const ScopedDispatchFault&) = delete;
	ScopedDispatchFault(ScopedDispatchFault&&)				   = delete;
	ScopedDispatchFault& operator=(ScopedDispatchFault&&)	   = delete;

	 private:
	Pfn& m_slot;
	Pfn	 m_saved;
};

/**
 * @brief A `vma::Allocator` (owned, destroyed on scope exit) with one allocation entry point forced
 *        to return `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
 *
 * Built from a live @ref veng::Context so every other entry point is the real driver's; only the
 * selected operation fails. Pass `.get()` to `Buffer::create` / `Image::create` / a `ResourcePool`
 * to drive their allocation-failure branches deterministically.
 */
class FailingAllocator
{
	 public:
	/** @brief Which VMA allocation operation should fail. */
	enum class Fail
	{
		AllocateMemory, ///< `vkAllocateMemory` fails — surfaces from both buffer and image creation.
		CreateBuffer,	///< `vkCreateBuffer` fails — buffer creation only.
		CreateImage,	///< `vkCreateImage` fails — image creation only.
	};

	FailingAllocator(const Context& ctx, Fail which)
	{
		vma::VulkanFunctions fns = vma::functionsFromDispatcher(VULKAN_HPP_DEFAULT_DISPATCHER);
		switch (which)
		{
			case Fail::AllocateMemory:
				fns.vkAllocateMemory = +[](VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*,
										   VkDeviceMemory*) -> VkResult { return VK_ERROR_OUT_OF_DEVICE_MEMORY; };
				break;
			case Fail::CreateBuffer:
				fns.vkCreateBuffer = +[](VkDevice, const VkBufferCreateInfo*, const VkAllocationCallbacks*,
										 VkBuffer*) -> VkResult { return VK_ERROR_OUT_OF_DEVICE_MEMORY; };
				break;
			case Fail::CreateImage:
				fns.vkCreateImage = +[](VkDevice, const VkImageCreateInfo*, const VkAllocationCallbacks*,
										VkImage*) -> VkResult { return VK_ERROR_OUT_OF_DEVICE_MEMORY; };
				break;
		}

		vma::AllocatorCreateInfo info{};
		info.physicalDevice	  = ctx.physical_device();
		info.device			  = ctx.device();
		info.instance		  = ctx.instance();
		info.vulkanApiVersion = VK_API_VERSION_1_3;
		info.pVulkanFunctions = &fns; // VMA snapshots these into the allocator here.
		(void)vma::createAllocator(&info, &m_allocator);
	}

	~FailingAllocator()
	{
		if (m_allocator)
		{
			m_allocator.destroy();
		}
	}

	FailingAllocator(const FailingAllocator&)			 = delete;
	FailingAllocator& operator=(const FailingAllocator&) = delete;
	FailingAllocator(FailingAllocator&&)				 = delete;
	FailingAllocator& operator=(FailingAllocator&&)		 = delete;

	/** @return The doctored allocator handle (null only if allocator creation itself failed). */
	[[nodiscard]] vma::Allocator get() const noexcept { return m_allocator; }

	 private:
	vma::Allocator m_allocator{};
};
} // namespace veng::test

#endif // VENG_TESTS_SUPPORT_VKFAULT_HPP
