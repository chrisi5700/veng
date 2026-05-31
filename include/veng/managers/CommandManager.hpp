/**
 * @file
 * @author chris
 * @brief L2 manager for Vulkan command-buffer pools, one pool per (queue, frame slot,
 *        recording thread).
 *
 * Multi-threaded node recording never shares a pool, which Vulkan requires be externally
 * synchronized. The manager is RAII and move-only.
 *
 * Frames-in-flight contract: @ref veng::CommandManager::reset_frame "reset_frame(slot)" recycles
 * every pool for that slot and MUST be called only once that slot's frame has fully retired
 * on the GPU (its fence/timeline value reached). Recording threads call
 * @ref veng::CommandManager::begin "begin" only for the slot currently being recorded.
 *
 * @ingroup managers
 */

#ifndef VENG_COMMANDMANAGER_HPP
#define VENG_COMMANDMANAGER_HPP

#include <cstddef>
#include <expected>
#include <map>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>
#include <veng/context/Context.hpp>
#include <veng/managers/QueueKind.hpp>
#include <vulkan/vulkan.hpp>

namespace veng
{
/**
 * @brief Manages per-(queue, frame-slot, thread) Vulkan command pools and the command
 *        buffers they allocate.
 *
 * Construction allocates nothing — pools are created lazily on the first @ref begin call,
 * so the constructor cannot fail and requires no factory or `std::expected`.
 *
 * @ingroup managers
 * @see QueueKind
 * @see FrameExecutor
 */
class CommandManager
{
	 public:
	/**
	 * @brief Construct a manager that borrows the device from @p context.
	 * @param context The engine context; must outlive this manager.
	 */
	explicit CommandManager(const Context& context) noexcept;

	CommandManager(const CommandManager&)			 = delete;
	CommandManager& operator=(const CommandManager&) = delete;
	CommandManager(CommandManager&& other) noexcept;
	CommandManager& operator=(CommandManager&& other) noexcept;
	~CommandManager();

	/**
	 * @brief Return a begun (one-time-submit) primary command buffer for @p queue and
	 *        @p frame_slot on the calling thread.
	 *
	 * Backed by a per-(queue, slot, thread) pool so concurrent callers never share a pool.
	 * Buffers are pooled and reused: @ref reset_frame "reset_frame(frame_slot)" rewinds the
	 * pool so the next frame re-records the same handles — calling `begin` every frame
	 * allocates a constant, not a growing, number of command buffers. Multiple `begin` calls
	 * within one frame (before a reset) each return a distinct buffer.
	 *
	 * @param queue      Which device queue the buffer will be submitted to.
	 * @param frame_slot The in-flight slot index for this frame.
	 * @return A begun command buffer, or the `vk::Result` error if pool/buffer creation fails.
	 */
	[[nodiscard]] std::expected<vk::CommandBuffer, vk::Result> begin(QueueKind queue, std::size_t frame_slot);

	/**
	 * @brief Recycle every command buffer in every pool belonging to @p frame_slot.
	 *
	 * @pre The frame that last occupied @p frame_slot has fully retired on the GPU
	 *      (its fence has been signalled or its timeline value reached).
	 * @param frame_slot The in-flight slot whose pools are to be reset.
	 */
	void reset_frame(std::size_t frame_slot);

	/**
	 * @brief Emit a synchronization2 image layout transition barrier covering mip level 0.
	 * @param cmd        The command buffer to record the barrier into.
	 * @param image      The image to transition.
	 * @param old_layout Layout the image is currently in.
	 * @param new_layout Layout to transition to.
	 * @param src_stage  Pipeline stage that last wrote the image.
	 * @param src_access Access type of the last write.
	 * @param dst_stage  Pipeline stage that will next read/write the image.
	 * @param dst_access Access type of the next read/write.
	 * @param aspect     Image aspect to cover (defaults to color).
	 */
	static void image_barrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout old_layout,
							  vk::ImageLayout new_layout, vk::PipelineStageFlags2 src_stage,
							  vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage,
							  vk::AccessFlags2	   dst_access,
							  vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor);

	/**
	 * @brief Emit a synchronization2 image layout transition barrier over a specific mip range.
	 *
	 * The blit-down mip-chain generator transitions individual levels — for example,
	 * transitioning level @p i to `TRANSFER_SRC` after writing it and before using it as the
	 * source for level @p i+1 — so it needs to target a single, non-zero mip level.
	 *
	 * @param cmd             The command buffer to record the barrier into.
	 * @param image           The image to transition.
	 * @param old_layout      Layout the image range is currently in.
	 * @param new_layout      Layout to transition to.
	 * @param src_stage       Pipeline stage that last wrote the range.
	 * @param src_access      Access type of the last write.
	 * @param dst_stage       Pipeline stage that will next read/write the range.
	 * @param dst_access      Access type of the next read/write.
	 * @param base_mip_level  First mip level to cover.
	 * @param level_count     Number of mip levels to cover.
	 * @param aspect          Image aspect to cover (defaults to color).
	 */
	static void image_barrier_range(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout old_layout,
									vk::ImageLayout new_layout, vk::PipelineStageFlags2 src_stage,
									vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage,
									vk::AccessFlags2 dst_access, std::uint32_t base_mip_level,
									std::uint32_t		 level_count,
									vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor);

	 private:
	using PoolKey = std::tuple<std::size_t, std::thread::id, std::uint8_t>; // (slot, thread, queue)

	/// One Vulkan pool plus its reusable command buffers. `used` is the high-water index
	/// into `buffers` for the current frame; `reset_frame` resets it to 0 (and the pool),
	/// so handed-out handles are re-recorded next frame rather than re-allocated.
	struct Pool
	{
		vk::CommandPool				   pool;
		std::vector<vk::CommandBuffer> buffers;
		std::size_t					   used = 0; ///< High-water index; rewound to 0 on reset.
	};

	void destroy() noexcept;

	const Context*			m_context;
	vk::Device				m_device;
	std::mutex				m_mutex;
	std::map<PoolKey, Pool> m_pools;
};
} // namespace veng

#endif // VENG_COMMANDMANAGER_HPP
