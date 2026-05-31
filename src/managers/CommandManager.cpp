/**
 * @file
 * @author chris
 * @brief Implementation of @ref veng::CommandManager.
 * @ingroup managers
 */

#include <utility>
#include <veng/managers/CommandManager.hpp>

namespace veng
{
CommandManager::CommandManager(const Context& context) noexcept
	: m_context(&context)
	, m_device(context.device())
{
}

void CommandManager::destroy() noexcept
{
	if (!m_device)
	{
		return;
	}
	for (const auto& [key, slot] : m_pools)
	{
		m_device.destroyCommandPool(slot.pool);
	}
	m_pools.clear();
	m_device = nullptr;
}

CommandManager::CommandManager(CommandManager&& other) noexcept
	: m_context(other.m_context)
	, m_device(other.m_device)
{
	// The mutex itself is not moved (each instance has its own); transfer the pools
	// under the source's lock.
	const std::lock_guard lock(other.m_mutex);
	m_pools			= std::move(other.m_pools);
	other.m_context = nullptr;
	other.m_device	= nullptr;
}

CommandManager& CommandManager::operator=(CommandManager&& other) noexcept
{
	if (this != &other)
	{
		destroy();
		const std::scoped_lock lock(m_mutex, other.m_mutex);
		m_context		= other.m_context;
		m_device		= other.m_device;
		m_pools			= std::move(other.m_pools);
		other.m_context = nullptr;
		other.m_device	= nullptr;
	}
	return *this;
}

CommandManager::~CommandManager()
{
	destroy();
}

std::expected<vk::CommandBuffer, vk::Result> CommandManager::begin(QueueKind queue, std::size_t frame_slot)
{
	const PoolKey		  key{frame_slot, std::this_thread::get_id(), static_cast<std::uint8_t>(queue)};
	const std::lock_guard lock(m_mutex);

	auto it = m_pools.find(key);
	if (it == m_pools.end())
	{
		const auto pool = m_device.createCommandPool(vk::CommandPoolCreateInfo()
														 .setQueueFamilyIndex(queue_family_of(*m_context, queue))
														 .setFlags(vk::CommandPoolCreateFlagBits::eTransient));
		if (pool.result != vk::Result::eSuccess)
		{
			return std::unexpected(pool.result);
		}
		it = m_pools.emplace(key, Pool{.pool = pool.value, .buffers = {}, .used = 0}).first;
	}
	Pool& slot = it->second;

	// Reuse a buffer recorded on a previous frame (reset_frame rewinds `used`), allocating
	// a new one only when this frame asks for more buffers than any prior frame did. The
	// total allocated count is bounded by the per-frame high-water mark — no per-frame leak.
	if (slot.used == slot.buffers.size())
	{
		const auto buffers = m_device.allocateCommandBuffers(vk::CommandBufferAllocateInfo()
																 .setCommandPool(slot.pool)
																 .setLevel(vk::CommandBufferLevel::ePrimary)
																 .setCommandBufferCount(1));
		if (buffers.result != vk::Result::eSuccess)
		{
			return std::unexpected(buffers.result);
		}
		slot.buffers.push_back(buffers.value.front());
	}

	const vk::CommandBuffer cmd = slot.buffers[slot.used++];
	if (const vk::Result begun =
			cmd.begin(vk::CommandBufferBeginInfo().setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
		begun != vk::Result::eSuccess)
	{
		return std::unexpected(begun);
	}
	return cmd;
}

void CommandManager::reset_frame(std::size_t frame_slot)
{
	const std::lock_guard lock(m_mutex);
	for (auto& [key, slot] : m_pools)
	{
		if (std::get<0>(key) == frame_slot)
		{
			static_cast<void>(m_device.resetCommandPool(slot.pool));
			slot.used = 0; // rewind: next frame re-records the same buffers
		}
	}
}

void CommandManager::image_barrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout old_layout,
								   vk::ImageLayout new_layout, vk::PipelineStageFlags2 src_stage,
								   vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage,
								   vk::AccessFlags2 dst_access, vk::ImageAspectFlags aspect)
{
	image_barrier_range(cmd, image, old_layout, new_layout, src_stage, src_access, dst_stage, dst_access, 0, 1, aspect);
}

void CommandManager::image_barrier_range(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout old_layout,
										 vk::ImageLayout new_layout, vk::PipelineStageFlags2 src_stage,
										 vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage,
										 vk::AccessFlags2 dst_access, std::uint32_t base_mip_level,
										 std::uint32_t level_count, vk::ImageAspectFlags aspect)
{
	const auto barrier = vk::ImageMemoryBarrier2()
							 .setSrcStageMask(src_stage)
							 .setSrcAccessMask(src_access)
							 .setDstStageMask(dst_stage)
							 .setDstAccessMask(dst_access)
							 .setOldLayout(old_layout)
							 .setNewLayout(new_layout)
							 .setImage(image)
							 .setSubresourceRange(vk::ImageSubresourceRange()
													  .setAspectMask(aspect)
													  .setBaseMipLevel(base_mip_level)
													  .setLevelCount(level_count)
													  .setBaseArrayLayer(0)
													  .setLayerCount(1));
	cmd.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(barrier));
}
} // namespace veng
