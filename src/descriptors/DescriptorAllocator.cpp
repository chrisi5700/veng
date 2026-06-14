/**
 * @file
 * @author chris
 * @brief @ref veng::DescriptorAllocator implementation.
 * @ingroup descriptors
 */

#include <algorithm>
#include <utility>
#include <veng/descriptors/DescriptorAllocator.hpp>
#include <veng/rhi/Device.hpp>

namespace veng
{
std::vector<DescriptorAllocator::PoolSizeRatio> DescriptorAllocator::default_ratios()
{
	return {
		{vk::DescriptorType::eUniformBuffer, 1.0F},		   {vk::DescriptorType::eStorageBuffer, 1.0F},
		{vk::DescriptorType::eCombinedImageSampler, 1.0F}, {vk::DescriptorType::eSampledImage, 1.0F},
		{vk::DescriptorType::eStorageImage, 1.0F},		   {vk::DescriptorType::eSampler, 0.5F},
		{vk::DescriptorType::eUniformTexelBuffer, 0.5F},   {vk::DescriptorType::eStorageTexelBuffer, 0.5F},
		{vk::DescriptorType::eInputAttachment, 0.5F},
	};
}

DescriptorAllocator::DescriptorAllocator(vk::Device device, std::uint32_t sets_per_pool,
										 std::vector<PoolSizeRatio> ratios)
	: m_device(device)
	, m_sets_per_pool(std::max(sets_per_pool, 1U))
	, m_ratios(std::move(ratios))
{
}

DescriptorAllocator::DescriptorAllocator(rhi::Device& rhi, std::uint32_t sets_per_pool,
										 std::vector<PoolSizeRatio> ratios)
	: DescriptorAllocator(rhi.device(), sets_per_pool, std::move(ratios))
{
}

void DescriptorAllocator::destroy() noexcept
{
	if (!m_device)
	{
		return;
	}
	for (const vk::DescriptorPool pool : m_pools)
	{
		m_device.destroyDescriptorPool(pool);
	}
	m_pools.clear();
	m_device = nullptr;
}

DescriptorAllocator::DescriptorAllocator(DescriptorAllocator&& other) noexcept
	: m_device(other.m_device)
	, m_sets_per_pool(other.m_sets_per_pool)
	, m_ratios(std::move(other.m_ratios))
	, m_pools(std::move(other.m_pools))
	, m_current(other.m_current)
{
	other.m_device = nullptr;
	other.m_pools.clear();
	other.m_current = 0;
}

DescriptorAllocator& DescriptorAllocator::operator=(DescriptorAllocator&& other) noexcept
{
	if (this != &other)
	{
		destroy();
		m_device		= other.m_device;
		m_sets_per_pool = other.m_sets_per_pool;
		m_ratios		= std::move(other.m_ratios);
		m_pools			= std::move(other.m_pools);
		m_current		= other.m_current;
		other.m_device	= nullptr;
		other.m_pools.clear();
		other.m_current = 0;
	}
	return *this;
}

DescriptorAllocator::~DescriptorAllocator()
{
	destroy();
}

vk::DescriptorPool DescriptorAllocator::create_pool() const
{
	std::vector<vk::DescriptorPoolSize> sizes;
	sizes.reserve(m_ratios.size());
	for (const PoolSizeRatio& ratio : m_ratios)
	{
		const auto count = std::max(1U, static_cast<std::uint32_t>(ratio.ratio * static_cast<float>(m_sets_per_pool)));
		sizes.push_back(vk::DescriptorPoolSize().setType(ratio.type).setDescriptorCount(count));
	}

	const auto info	  = vk::DescriptorPoolCreateInfo().setMaxSets(m_sets_per_pool).setPoolSizes(sizes);
	const auto result = m_device.createDescriptorPool(info);
	return result.result == vk::Result::eSuccess ? result.value : vk::DescriptorPool{};
}

std::expected<vk::DescriptorSet, vk::Result> DescriptorAllocator::allocate(vk::DescriptorSetLayout layout)
{
	if (m_pools.empty())
	{
		const vk::DescriptorPool pool = create_pool();
		if (!pool)
		{
			return std::unexpected(vk::Result::eErrorOutOfPoolMemory);
		}
		m_pools.push_back(pool);
		m_current = 0;
	}

	// At most two attempts: the current pool, then a freshly grown one.
	for (int attempt = 0; attempt < 2; ++attempt)
	{
		const auto info = vk::DescriptorSetAllocateInfo().setDescriptorPool(m_pools[m_current]).setSetLayouts(layout);
		vk::DescriptorSet set;
		const vk::Result  result = m_device.allocateDescriptorSets(&info, &set);
		if (result == vk::Result::eSuccess)
		{
			return set;
		}
		// Only pool exhaustion/fragmentation is recoverable by growing.
		if (result != vk::Result::eErrorOutOfPoolMemory && result != vk::Result::eErrorFragmentedPool)
		{
			return std::unexpected(result);
		}

		const vk::DescriptorPool pool = create_pool();
		if (!pool)
		{
			return std::unexpected(result);
		}
		m_pools.push_back(pool);
		m_current = m_pools.size() - 1;
	}

	return std::unexpected(vk::Result::eErrorOutOfPoolMemory);
}

void DescriptorAllocator::reset()
{
	for (const vk::DescriptorPool pool : m_pools)
	{
		// resetDescriptorPool cannot fail per spec but is [[nodiscard]]; consume it.
		static_cast<void>(m_device.resetDescriptorPool(pool));
	}
	m_current = 0;
}
} // namespace veng
