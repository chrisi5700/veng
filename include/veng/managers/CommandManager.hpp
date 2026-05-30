//
// Created by chris on 5/25/26.
//
// L2 manager — command-buffer pools (design.md §L2.4). One pool per (queue, frame
// slot, recording thread) so multi-threaded node recording never shares a pool (which
// Vulkan requires be externally synchronized). RAII, move-only.
//
// Frames-in-flight contract: `reset_frame(slot)` recycles every pool for that slot and
// MUST be called only once that slot's frame has fully retired on the GPU (its fence/
// timeline value reached). Recording threads call `begin` only for the slot currently
// being recorded.
//

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
class CommandManager
{
	 public:
	// Construction allocates nothing (pools are created lazily on first `begin`), so
	// it cannot fail — no factory/std::expected needed.
	explicit CommandManager(const Context& context) noexcept;

	CommandManager(const CommandManager&)			 = delete;
	CommandManager& operator=(const CommandManager&) = delete;
	CommandManager(CommandManager&& other) noexcept;
	CommandManager& operator=(CommandManager&& other) noexcept;
	~CommandManager();

	/// A begun (one-time-submit) primary command buffer for `queue`/`frame_slot` on the
	/// calling thread. Backed by a per-(queue, slot, thread) pool, so concurrent callers
	/// never share a pool. Buffers are pooled and reused: `reset_frame(frame_slot)` rewinds
	/// the pool so the next frame re-records the same handles — calling `begin` every frame
	/// allocates a constant, not a growing, number of command buffers. Multiple `begin`
	/// calls within one frame (before a reset) each return a distinct buffer.
	[[nodiscard]] std::expected<vk::CommandBuffer, vk::Result> begin(QueueKind queue, std::size_t frame_slot);

	/// Recycle every command buffer in every pool belonging to `frame_slot`. Call only
	/// when that slot's frame has retired on the GPU (see the contract above).
	void reset_frame(std::size_t frame_slot);

	/// synchronization2 image layout transition helper (one mip level, base 0).
	static void image_barrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout old_layout,
							  vk::ImageLayout new_layout, vk::PipelineStageFlags2 src_stage,
							  vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage,
							  vk::AccessFlags2	   dst_access,
							  vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor);

	/// synchronization2 transition for a specific mip range. The blit-down mip-chain generator
	/// transitions individual levels (e.g. level i to TRANSFER_SRC after writing it, before
	/// sampling it as the source for level i+1), so it needs to target a single, non-zero level.
	static void image_barrier_range(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout old_layout,
									vk::ImageLayout new_layout, vk::PipelineStageFlags2 src_stage,
									vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage,
									vk::AccessFlags2 dst_access, std::uint32_t base_mip_level,
									std::uint32_t		 level_count,
									vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor);

	 private:
	using PoolKey = std::tuple<std::size_t, std::thread::id, std::uint8_t>; // (slot, thread, queue)

	// One Vulkan pool plus its reusable command buffers. `used` is the high-water index
	// into `buffers` for the current frame; `reset_frame` resets it to 0 (and the pool),
	// so handed-out handles are re-recorded next frame rather than re-allocated.
	struct Pool
	{
		vk::CommandPool				   pool;
		std::vector<vk::CommandBuffer> buffers;
		std::size_t					   used = 0;
	};

	void destroy() noexcept;

	const Context*			m_context;
	vk::Device				m_device;
	std::mutex				m_mutex;
	std::map<PoolKey, Pool> m_pools;
};
} // namespace veng

#endif // VENG_COMMANDMANAGER_HPP
