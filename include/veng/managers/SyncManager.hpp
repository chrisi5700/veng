//
// Created by chris on 5/25/26.
//
// L2 manager — synchronization primitives (design.md §L2.4, §5). Owns one timeline
// semaphore per queue: that timeline IS the queue's monotonic revision clock (a
// "done" check is `getSemaphoreCounterValue(sem) >= ready_value`). RAII, move-only.
//

#ifndef VENG_SYNCMANAGER_HPP
#define VENG_SYNCMANAGER_HPP

#include <array>
#include <cstdint>
#include <expected>
#include <veng/context/Context.hpp>
#include <veng/managers/QueueKind.hpp>
#include <vulkan/vulkan.hpp>

namespace veng
{
class SyncManager
{
	 public:
	[[nodiscard]] static std::expected<SyncManager, vk::Result> create(const Context& context);

	SyncManager(const SyncManager&)			   = delete;
	SyncManager& operator=(const SyncManager&) = delete;
	SyncManager(SyncManager&& other) noexcept;
	SyncManager& operator=(SyncManager&& other) noexcept;
	~SyncManager();

	/// The timeline semaphore for `queue` — its monotonic revision clock.
	[[nodiscard]] vk::Semaphore timeline(QueueKind queue) const noexcept { return m_timelines[queue_index(queue)]; }

	/// Reserve and return the next value to signal on `queue`'s timeline (monotonic).
	[[nodiscard]] std::uint64_t next_value(QueueKind queue) noexcept { return ++m_counters[queue_index(queue)]; }

	/// The value the GPU has currently reached on `queue`'s timeline.
	[[nodiscard]] std::expected<std::uint64_t, vk::Result> current_value(QueueKind queue) const;

	 private:
	SyncManager(vk::Device device, std::array<vk::Semaphore, 2> timelines) noexcept;
	void destroy() noexcept;

	vk::Device					 m_device;
	std::array<vk::Semaphore, 2> m_timelines;
	std::array<std::uint64_t, 2> m_counters{};
};
} // namespace veng

#endif // VENG_SYNCMANAGER_HPP
