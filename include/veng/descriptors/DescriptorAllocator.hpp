/**
 * @file
 * @author chris
 * @brief Growable descriptor-pool allocator: hand it a layout and it returns a set,
 *        transparently spawning a fresh pool when the current one is exhausted.
 *
 * This is the L2 descriptor-allocation layer. Hand @ref veng::DescriptorAllocator a
 * `vk::DescriptorSetLayout` (for example the one @ref veng::ComputePipeline or
 * @ref veng::GraphicsPipeline exposes) and it returns a set, transparently spawning a fresh
 * pool when the current one is exhausted. The allocator is RAII and move-only;
 * `reset()` recycles every set for the next frame without destroying pool objects, so
 * steady-state framing does not churn pool creation.
 *
 * @ingroup descriptors
 */

#ifndef VENG_DESCRIPTORALLOCATOR_HPP
#define VENG_DESCRIPTORALLOCATOR_HPP

#include <cstdint>
#include <expected>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace veng::rhi
{
class Device;
}

namespace veng
{
/**
 * @brief RAII, move-only descriptor-set allocator backed by a growable list of
 *        `vk::DescriptorPool` objects.
 *
 * Each pool is sized by `sets_per_pool` and a list of @ref PoolSizeRatio entries.
 * When a pool is exhausted, a new one is created transparently. Calling `reset()`
 * recycles all allocated sets by resetting every pool; the pools themselves are kept
 * alive so steady-state per-frame allocation causes no Vulkan object churn.
 *
 * @ingroup descriptors
 * @see ComputePipeline
 * @see GraphicsPipeline
 */
class DescriptorAllocator
{
	 public:
	/**
	 * @brief One descriptor type and how many of it to budget per `sets_per_pool` sets.
	 * @ingroup descriptors
	 */
	struct PoolSizeRatio
	{
		vk::DescriptorType type;  ///< The descriptor type to reserve space for.
		float			   ratio; ///< Multiplied by `sets_per_pool` to get the descriptor count.
	};

	/**
	 * @brief Construct an allocator for the given device.
	 * @param device        The Vulkan device that owns all pools and sets.
	 * @param sets_per_pool Capacity of each pool in descriptor sets (default: 1024).
	 * @param ratios        Per-type size ratios; defaults to @ref default_ratios.
	 */
	explicit DescriptorAllocator(vk::Device device, std::uint32_t sets_per_pool = 1024,
								 std::vector<PoolSizeRatio> ratios = default_ratios());

	/**
	 * @brief RHI-vocabulary overload for past-L3 callers: a node/pass builds an allocator naming only
	 *        the @ref veng::rhi::Device; delegates to the `vk::Device` overload with `rhi.device()`.
	 * @param rhi           The RHI device whose logical device owns all pools and sets.
	 * @param sets_per_pool Capacity of each pool in descriptor sets (default: 1024).
	 * @param ratios        Per-type size ratios; defaults to @ref default_ratios.
	 */
	explicit DescriptorAllocator(rhi::Device& rhi, std::uint32_t sets_per_pool = 1024,
								 std::vector<PoolSizeRatio> ratios = default_ratios());

	DescriptorAllocator(const DescriptorAllocator&)			   = delete;
	DescriptorAllocator& operator=(const DescriptorAllocator&) = delete;
	DescriptorAllocator(DescriptorAllocator&& other) noexcept;
	DescriptorAllocator& operator=(DescriptorAllocator&& other) noexcept;
	~DescriptorAllocator();

	/**
	 * @brief Allocate a descriptor set for `layout`, growing the pool list on exhaustion.
	 *
	 * The returned set's lifetime is tied to this allocator: it is freed when `reset()` is
	 * called or when the allocator is destroyed.
	 *
	 * @param layout The descriptor-set layout the set must conform to.
	 * @return The allocated `vk::DescriptorSet`, or a `vk::Result` error code on failure.
	 */
	[[nodiscard]] std::expected<vk::DescriptorSet, vk::Result> allocate(vk::DescriptorSetLayout layout);

	/**
	 * @brief Recycle every set previously handed out by resetting all pools.
	 *
	 * Pools are kept alive for reuse so steady-state per-frame allocation causes no
	 * Vulkan object churn. Call once per frame before re-allocating sets.
	 */
	void reset();

	/** @brief The number of `vk::DescriptorPool` objects currently held. */
	[[nodiscard]] std::size_t pool_count() const noexcept { return m_pools.size(); }

	/**
	 * @brief Default pool-size ratios: one-to-one budget for uniform/storage buffers,
	 *        combined/sampled/storage images, and half-budget for samplers and texel buffers.
	 * @return A vector of @ref PoolSizeRatio covering common descriptor types.
	 */
	[[nodiscard]] static std::vector<PoolSizeRatio> default_ratios();

	 private:
	[[nodiscard]] vk::DescriptorPool create_pool() const;
	void							 destroy() noexcept;

	vk::Device						m_device;
	std::uint32_t					m_sets_per_pool;
	std::vector<PoolSizeRatio>		m_ratios;
	std::vector<vk::DescriptorPool> m_pools;	   ///< Every pool, for `reset()` / destruction.
	std::size_t						m_current = 0; ///< Index of the pool we currently allocate from.
};
} // namespace veng

#endif // VENG_DESCRIPTORALLOCATOR_HPP
