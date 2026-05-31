/**
 * @file
 * @author chris
 * @brief L2 managers — enumeration and accessor helpers for the two device queues.
 *
 * The graph targets two queues: graphics and a (possibly dedicated) compute queue.
 *
 * @ingroup managers
 */

#ifndef VENG_QUEUEKIND_HPP
#define VENG_QUEUEKIND_HPP

#include <cstdint>
#include <veng/context/Context.hpp>
#include <vulkan/vulkan.hpp>

namespace veng
{
/**
 * @brief Identifies which of the two device queues a manager operation targets.
 * @ingroup managers
 * @see queue_index
 * @see queue_of
 * @see queue_family_of
 */
enum class QueueKind : std::uint8_t
{
	Graphics, ///< The graphics (and present) queue.
	Compute,  ///< The compute queue (may be shared with graphics).
};

/**
 * @brief Convert a @ref veng::QueueKind to a numeric index for per-queue arrays.
 * @param kind The queue to index.
 * @return 0 for `Graphics`, 1 for `Compute`.
 */
[[nodiscard]] constexpr std::size_t queue_index(QueueKind kind) noexcept
{
	return static_cast<std::size_t>(kind);
}

/**
 * @brief Return the `vk::Queue` handle for @p kind from @p context.
 * @param context The engine @ref veng::Context owning both queues.
 * @param kind    Which queue to retrieve.
 * @return The corresponding `vk::Queue`.
 */
[[nodiscard]] inline vk::Queue queue_of(const Context& context, QueueKind kind) noexcept
{
	return kind == QueueKind::Graphics ? context.graphics_queue() : context.compute_queue();
}

/**
 * @brief Return the Vulkan queue-family index for @p kind from @p context.
 * @param context The engine @ref veng::Context owning both queues.
 * @param kind    Which queue's family to retrieve.
 * @return The `uint32_t` family index as reported by the physical device.
 */
[[nodiscard]] inline std::uint32_t queue_family_of(const Context& context, QueueKind kind) noexcept
{
	return kind == QueueKind::Graphics ? context.queue_indices().graphics : context.queue_indices().compute;
}
} // namespace veng

#endif // VENG_QUEUEKIND_HPP
