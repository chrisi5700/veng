//
// Created by chris on 5/25/26.
//
// See SyncManager.hpp and design.md §L2.4.
//

#include <utility>
#include <veng/managers/SyncManager.hpp>

namespace veng
{
std::expected<SyncManager, vk::Result> SyncManager::create(const Context& context)
{
	const vk::Device device = context.device();

	std::array<vk::Semaphore, 2> timelines{};
	for (std::size_t i = 0; i < timelines.size(); ++i)
	{
		const auto type_info =
			vk::SemaphoreTypeCreateInfo().setSemaphoreType(vk::SemaphoreType::eTimeline).setInitialValue(0);
		const auto result = device.createSemaphore(vk::SemaphoreCreateInfo().setPNext(&type_info));
		if (result.result != vk::Result::eSuccess)
		{
			for (std::size_t j = 0; j < i; ++j)
			{
				device.destroySemaphore(timelines[j]);
			}
			return std::unexpected(result.result);
		}
		timelines[i] = result.value;
	}
	return SyncManager(device, timelines);
}

SyncManager::SyncManager(vk::Device device, std::array<vk::Semaphore, 2> timelines) noexcept
	: m_device(device)
	, m_timelines(timelines)
{
}

void SyncManager::destroy() noexcept
{
	if (!m_device)
	{
		return;
	}
	for (const vk::Semaphore semaphore : m_timelines)
	{
		if (semaphore)
		{
			m_device.destroySemaphore(semaphore);
		}
	}
	m_device = nullptr;
	m_timelines.fill(nullptr);
}

SyncManager::SyncManager(SyncManager&& other) noexcept
	: m_device(other.m_device)
	, m_timelines(other.m_timelines)
	, m_counters(other.m_counters)
{
	other.m_device = nullptr;
	other.m_timelines.fill(nullptr);
}

SyncManager& SyncManager::operator=(SyncManager&& other) noexcept
{
	if (this != &other)
	{
		destroy();
		m_device	   = other.m_device;
		m_timelines	   = other.m_timelines;
		m_counters	   = other.m_counters;
		other.m_device = nullptr;
		other.m_timelines.fill(nullptr);
	}
	return *this;
}

SyncManager::~SyncManager()
{
	destroy();
}

std::expected<std::uint64_t, vk::Result> SyncManager::current_value(QueueKind queue) const
{
	const auto result = m_device.getSemaphoreCounterValue(m_timelines[queue_index(queue)]);
	if (result.result != vk::Result::eSuccess)
	{
		return std::unexpected(result.result);
	}
	return result.value;
}
} // namespace veng
