//
// Created by chris on 5/25/26.
//
// L2 construction layer — descriptor allocation (design.md §L2.2). A growable pool
// allocator: hand it a descriptor-set layout (e.g. the one a ComputePipeline exposes)
// and it returns a set, transparently spawning a fresh pool when the current one is
// exhausted. RAII, move-only; `reset()` recycles every set for the next frame.
//

#ifndef VENG_DESCRIPTORALLOCATOR_HPP
#define VENG_DESCRIPTORALLOCATOR_HPP

#include <cstdint>
#include <expected>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace veng
{
class DescriptorAllocator
{
	 public:
	/// One descriptor type and how many of it to budget per `sets_per_pool` sets.
	struct PoolSizeRatio
	{
		vk::DescriptorType type;
		float			   ratio;
	};

	explicit DescriptorAllocator(vk::Device device, std::uint32_t sets_per_pool = 1024,
								 std::vector<PoolSizeRatio> ratios = default_ratios());

	DescriptorAllocator(const DescriptorAllocator&)			   = delete;
	DescriptorAllocator& operator=(const DescriptorAllocator&) = delete;
	DescriptorAllocator(DescriptorAllocator&& other) noexcept;
	DescriptorAllocator& operator=(DescriptorAllocator&& other) noexcept;
	~DescriptorAllocator();

	/// Allocate a set for `layout`, growing the pool list on exhaustion. The returned
	/// set's lifetime is tied to this allocator (freed by `reset()` or destruction).
	[[nodiscard]] std::expected<vk::DescriptorSet, vk::Result> allocate(vk::DescriptorSetLayout layout);

	/// Recycle every set previously handed out by resetting all pools. Pools are kept
	/// for reuse, so steady-state framing does not churn pool creation.
	void reset();

	[[nodiscard]] std::size_t pool_count() const noexcept { return m_pools.size(); }

	[[nodiscard]] static std::vector<PoolSizeRatio> default_ratios();

	 private:
	[[nodiscard]] vk::DescriptorPool create_pool() const;
	void							 destroy() noexcept;

	vk::Device						m_device;
	std::uint32_t					m_sets_per_pool;
	std::vector<PoolSizeRatio>		m_ratios;
	std::vector<vk::DescriptorPool> m_pools;	   // every pool, for reset()/destroy()
	std::size_t						m_current = 0; // index of the pool we allocate from
};
} // namespace veng

#endif // VENG_DESCRIPTORALLOCATOR_HPP
