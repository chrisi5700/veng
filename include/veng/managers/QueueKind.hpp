//
// Created by chris on 5/25/26.
//
// L2 managers — which device queue a manager operation targets (design.md §L2.4).
// The graph targets two queues: graphics and (possibly dedicated) compute.
//

#ifndef VENG_QUEUEKIND_HPP
#define VENG_QUEUEKIND_HPP

#include <cstdint>
#include <veng/context/Context.hpp>
#include <vulkan/vulkan.hpp>

namespace veng
{
enum class QueueKind : std::uint8_t
{
	Graphics,
	Compute,
};

/// Index for per-queue arrays.
[[nodiscard]] constexpr std::size_t queue_index(QueueKind kind) noexcept
{
	return static_cast<std::size_t>(kind);
}

[[nodiscard]] inline vk::Queue queue_of(const Context& context, QueueKind kind) noexcept
{
	return kind == QueueKind::Graphics ? context.graphics_queue() : context.compute_queue();
}

[[nodiscard]] inline std::uint32_t queue_family_of(const Context& context, QueueKind kind) noexcept
{
	return kind == QueueKind::Graphics ? context.queue_indices().graphics : context.queue_indices().compute;
}
} // namespace veng

#endif // VENG_QUEUEKIND_HPP
