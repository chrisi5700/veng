/**
 * @file
 * @author chris
 * @brief @ref veng::rhi::Device implementation — the texture/buffer handle slot-maps.
 * @ingroup rhi
 */

#include <expected>
#include <vector>
#include <veng/rhi/CommandEncoder.hpp>
#include <veng/rhi/Convert.hpp>
#include <veng/rhi/Device.hpp>

namespace veng::rhi
{
Device::Device(vk::Device device, vma::Allocator allocator, vk::Queue graphics_queue,
			   std::uint32_t graphics_family) noexcept
	: m_device(device)
	, m_allocator(allocator)
	, m_graphics_queue(graphics_queue)
{
	// A one-shot command pool + reusable buffer + fence so begin_commands()/submit() can record and
	// run a pass without the caller touching Vulkan. On failure these stay null and submit() reports it.
	const auto pool = m_device.createCommandPool(
		vk::CommandPoolCreateInfo()
			.setQueueFamilyIndex(graphics_family)
			.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer | vk::CommandPoolCreateFlagBits::eTransient));
	if (pool.result == vk::Result::eSuccess)
	{
		m_command_pool	= pool.value;
		const auto bufs = m_device.allocateCommandBuffers(vk::CommandBufferAllocateInfo()
															  .setCommandPool(m_command_pool)
															  .setLevel(vk::CommandBufferLevel::ePrimary)
															  .setCommandBufferCount(1));
		if (bufs.result == vk::Result::eSuccess)
		{
			m_command_buffer = bufs.value.front();
		}
	}
	if (const auto fence = m_device.createFence({}); fence.result == vk::Result::eSuccess)
	{
		m_fence = fence.value;
	}
}

Device::~Device()
{
	// Free everything the device owns: created textures/buffers (allocation marks ownership), samplers,
	// and the one-shot command objects. All before the borrowed vk::Device/allocator are destroyed.
	for (const Texture& tex : m_textures)
	{
		if (tex.allocation)
		{
			if (tex.view)
			{
				m_device.destroyImageView(tex.view);
			}
			m_allocator.destroyImage(tex.image, tex.allocation);
		}
	}
	for (const Buffer& buf : m_buffers)
	{
		if (buf.allocation)
		{
			m_allocator.destroyBuffer(buf.buffer, buf.allocation);
		}
	}
	for (const vk::Sampler smp : m_samplers)
	{
		if (smp)
		{
			m_device.destroySampler(smp);
		}
	}
	if (m_fence)
	{
		m_device.destroyFence(m_fence);
	}
	if (m_command_pool)
	{
		m_device.destroyCommandPool(m_command_pool); // frees m_command_buffer with it
	}
}

std::expected<TextureHandle, Error> Device::create_texture(const TextureDesc& desc)
{
	const auto				  image_info = vk::ImageCreateInfo()
											   .setImageType(vk::ImageType::e2D)
											   .setFormat(to_vk(desc.format))
											   .setExtent(vk::Extent3D{desc.extent.width, desc.extent.height, 1})
											   .setMipLevels(1)
											   .setArrayLayers(1)
											   .setSamples(to_vk(desc.samples))
											   .setTiling(vk::ImageTiling::eOptimal)
											   .setUsage(to_vk(desc.usage))
											   .setSharingMode(vk::SharingMode::eExclusive)
											   .setInitialLayout(vk::ImageLayout::eUndefined);
	vma::AllocationCreateInfo alloc_info{};
	alloc_info.usage = vma::MemoryUsage::eAutoPreferDevice;

	vk::Image		image;
	vma::Allocation allocation;
	if (const vk::Result result = m_allocator.createImage(&image_info, &alloc_info, &image, &allocation, nullptr);
		result != vk::Result::eSuccess)
	{
		return std::unexpected(to_error(result));
	}

	// A view only when the usage allows one (attachment/sampled); transfer-only textures get none.
	constexpr TextureUsageFlags VIEW_USAGES =
		TextureUsageFlags::SAMPLED | TextureUsageFlags::COLOR_ATTACHMENT | TextureUsageFlags::DEPTH_ATTACHMENT;
	vk::ImageView view{};
	if ((desc.usage & VIEW_USAGES) != TextureUsageFlags::NONE)
	{
		const bool depth	   = (desc.usage & TextureUsageFlags::DEPTH_ATTACHMENT) != TextureUsageFlags::NONE;
		const auto view_info   = vk::ImageViewCreateInfo()
									 .setImage(image)
									 .setViewType(vk::ImageViewType::e2D)
									 .setFormat(to_vk(desc.format))
									 .setSubresourceRange(vk::ImageSubresourceRange()
															  .setAspectMask(depth ? vk::ImageAspectFlagBits::eDepth
																				   : vk::ImageAspectFlagBits::eColor)
															  .setLevelCount(1)
															  .setLayerCount(1));
		const auto view_result = m_device.createImageView(view_info);
		if (view_result.result != vk::Result::eSuccess)
		{
			m_allocator.destroyImage(image, allocation);
			return std::unexpected(to_error(view_result.result));
		}
		view = view_result.value;
	}

	const Texture slot{.image = image, .view = view, .allocation = allocation};
	return claim_texture(slot);
}

TextureHandle Device::claim_texture(const Texture& slot)
{
	// Recycle a freed id when one is available, else grow the table. The returned handle is stamped
	// with the slot's current generation (bumped at each release), so a recycled id never compares
	// equal to a handle that still names the resource previously at that slot.
	if (!m_free_textures.empty())
	{
		const std::uint32_t id = m_free_textures.back();
		m_free_textures.pop_back();
		m_textures[id] = slot;
		return TextureHandle{.id = id, .generation = m_texture_generations[id]};
	}
	m_textures.push_back(slot);
	m_texture_generations.push_back(0);
	return TextureHandle{.id = static_cast<std::uint32_t>(m_textures.size() - 1), .generation = 0};
}

void Device::destroy_texture(TextureHandle handle) noexcept
{
	if (handle.id < m_textures.size())
	{
		const Texture& tex = m_textures[handle.id];
		if (tex.view)
		{
			m_device.destroyImageView(tex.view);
		}
		if (tex.allocation)
		{
			m_allocator.destroyImage(tex.image, tex.allocation);
		}
		m_textures[handle.id] = Texture{};
		++m_texture_generations[handle.id]; // any handle still naming the freed resource is now stale
		m_free_textures.push_back(handle.id);
	}
}

std::expected<BufferHandle, Error> Device::create_buffer(const BufferDesc& desc)
{
	const auto				  buffer_info = vk::BufferCreateInfo()
												.setSize(desc.size)
												.setUsage(to_vk(desc.usage))
												.setSharingMode(vk::SharingMode::eExclusive);
	vma::AllocationCreateInfo alloc_info{};
	if (desc.memory == MemoryAccess::HOST_VISIBLE)
	{
		alloc_info.usage = vma::MemoryUsage::eAuto;
		alloc_info.flags = vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessRandom;
	}
	else
	{
		alloc_info.usage = vma::MemoryUsage::eAutoPreferDevice;
	}

	vk::Buffer			buffer;
	vma::Allocation		allocation;
	vma::AllocationInfo info{};
	if (const vk::Result result = m_allocator.createBuffer(&buffer_info, &alloc_info, &buffer, &allocation, &info);
		result != vk::Result::eSuccess)
	{
		return std::unexpected(to_error(result));
	}

	const Buffer slot{.buffer = buffer, .allocation = allocation, .mapped = info.pMappedData};
	return claim_buffer(slot);
}

BufferHandle Device::claim_buffer(const Buffer& slot)
{
	// Mirrors claim_texture: recycle a freed id (stamping the returned handle with the slot's current
	// generation) or grow the table, so a recycled buffer id never aliases a still-live handle.
	if (!m_free_buffers.empty())
	{
		const std::uint32_t id = m_free_buffers.back();
		m_free_buffers.pop_back();
		m_buffers[id] = slot;
		return BufferHandle{.id = id, .generation = m_buffer_generations[id]};
	}
	m_buffers.push_back(slot);
	m_buffer_generations.push_back(0);
	return BufferHandle{.id = static_cast<std::uint32_t>(m_buffers.size() - 1), .generation = 0};
}

void Device::destroy_buffer(BufferHandle handle) noexcept
{
	if (handle.id < m_buffers.size())
	{
		const Buffer& buf = m_buffers[handle.id];
		if (buf.allocation)
		{
			m_allocator.destroyBuffer(buf.buffer, buf.allocation);
		}
		m_buffers[handle.id] = Buffer{};
		++m_buffer_generations[handle.id]; // any handle still naming the freed buffer is now stale
		m_free_buffers.push_back(handle.id);
	}
}

void* Device::mapped(BufferHandle handle) const noexcept
{
	return (handle.id < m_buffers.size() && m_buffer_generations[handle.id] == handle.generation)
			   ? m_buffers[handle.id].mapped
			   : nullptr;
}

CommandEncoder Device::begin_commands()
{
	if (m_command_buffer)
	{
		(void)m_command_buffer.reset();
		(void)m_command_buffer.begin(
			vk::CommandBufferBeginInfo().setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
	}
	return CommandEncoder(m_command_buffer, *this);
}

std::expected<void, Error> Device::submit(CommandEncoder& enc)
{
	const vk::CommandBuffer cmd = enc.vk();
	if (!cmd || !m_fence)
	{
		return std::unexpected(to_error(vk::Result::eErrorInitializationFailed));
	}
	if (const vk::Result result = cmd.end(); result != vk::Result::eSuccess)
	{
		return std::unexpected(to_error(result));
	}
	if (const vk::Result result = m_graphics_queue.submit(vk::SubmitInfo().setCommandBuffers(cmd), m_fence);
		result != vk::Result::eSuccess)
	{
		return std::unexpected(to_error(result));
	}
	if (const vk::Result result = m_device.waitForFences(m_fence, vk::True, UINT64_MAX); result != vk::Result::eSuccess)
	{
		return std::unexpected(to_error(result));
	}
	if (const vk::Result result = m_device.resetFences(m_fence); result != vk::Result::eSuccess)
	{
		return std::unexpected(to_error(result));
	}
	return {};
}

std::expected<SamplerHandle, Error> Device::create_sampler(const SamplerDesc& desc)
{
	// The one place a SamplerDesc becomes Vulkan: one address mode applied to U/V/W.
	const auto info = vk::SamplerCreateInfo()
						  .setMagFilter(to_vk(desc.mag_filter))
						  .setMinFilter(to_vk(desc.min_filter))
						  .setMipmapMode(to_vk(desc.mipmap_mode))
						  .setAddressModeU(to_vk(desc.address_mode))
						  .setAddressModeV(to_vk(desc.address_mode))
						  .setAddressModeW(to_vk(desc.address_mode))
						  .setAnisotropyEnable(static_cast<vk::Bool32>(desc.anisotropy))
						  .setMaxAnisotropy(desc.max_anisotropy)
						  .setMinLod(desc.min_lod)
						  .setMaxLod(desc.max_lod);
	const auto created = m_device.createSampler(info);
	if (created.result != vk::Result::eSuccess)
	{
		return std::unexpected(to_error(created.result));
	}
	if (!m_free_samplers.empty())
	{
		const std::uint32_t id = m_free_samplers.back();
		m_free_samplers.pop_back();
		m_samplers[id] = created.value;
		return SamplerHandle{.id = id};
	}
	m_samplers.push_back(created.value);
	return SamplerHandle{.id = static_cast<std::uint32_t>(m_samplers.size() - 1)};
}

void Device::release_sampler(SamplerHandle handle) noexcept
{
	if (handle.id < m_samplers.size())
	{
		if (m_samplers[handle.id])
		{
			m_device.destroySampler(m_samplers[handle.id]);
		}
		m_samplers[handle.id] = vk::Sampler{};
		m_free_samplers.push_back(handle.id);
	}
}

vk::Sampler Device::sampler(SamplerHandle handle) const noexcept
{
	return handle.id < m_samplers.size() ? m_samplers[handle.id] : vk::Sampler{};
}
TextureHandle Device::register_texture(vk::Image image, vk::ImageView view)
{
	return claim_texture(Texture{.image = image, .view = view});
}

void Device::rebind_texture(TextureHandle handle, vk::Image image, vk::ImageView view) noexcept
{
	// Generation-checked: a stale handle must not clobber a slot that has since been recycled.
	if (handle.id < m_textures.size() && m_texture_generations[handle.id] == handle.generation)
	{
		m_textures[handle.id] = Texture{.image = image, .view = view};
	}
}

void Device::release_texture(TextureHandle handle) noexcept
{
	if (handle.id < m_textures.size())
	{
		m_textures[handle.id] = Texture{};
		++m_texture_generations[handle.id]; // any handle still naming the freed resource is now stale
		m_free_textures.push_back(handle.id);
	}
}

vk::Image Device::image(TextureHandle handle) const noexcept
{
	return (handle.id < m_textures.size() && m_texture_generations[handle.id] == handle.generation)
			   ? m_textures[handle.id].image
			   : vk::Image{};
}

vk::ImageView Device::view(TextureHandle handle) const noexcept
{
	// A generation mismatch means this handle outlived its resource (the slot was freed/recycled):
	// resolve to a null view so a stale binding fails closed instead of aliasing a different texture.
	return (handle.id < m_textures.size() && m_texture_generations[handle.id] == handle.generation)
			   ? m_textures[handle.id].view
			   : vk::ImageView{};
}

BufferHandle Device::register_buffer(vk::Buffer buffer)
{
	return claim_buffer(Buffer{.buffer = buffer});
}

void Device::release_buffer(BufferHandle handle) noexcept
{
	if (handle.id < m_buffers.size())
	{
		m_buffers[handle.id] = Buffer{};
		++m_buffer_generations[handle.id]; // any handle still naming the freed buffer is now stale
		m_free_buffers.push_back(handle.id);
	}
}

vk::Buffer Device::buffer(BufferHandle handle) const noexcept
{
	// Generation mismatch => the handle outlived its buffer (slot freed/recycled): resolve to null so
	// a stale binding fails closed rather than aliasing a different buffer.
	return (handle.id < m_buffers.size() && m_buffer_generations[handle.id] == handle.generation)
			   ? m_buffers[handle.id].buffer
			   : vk::Buffer{};
}

SemaphoreHandle Device::register_semaphore(vk::Semaphore semaphore)
{
	if (!m_free_semaphores.empty())
	{
		const std::uint32_t id = m_free_semaphores.back();
		m_free_semaphores.pop_back();
		m_semaphores[id] = semaphore;
		return SemaphoreHandle{.id = id};
	}
	m_semaphores.push_back(semaphore);
	return SemaphoreHandle{.id = static_cast<std::uint32_t>(m_semaphores.size() - 1)};
}

void Device::release_semaphore(SemaphoreHandle handle) noexcept
{
	if (handle.id < m_semaphores.size())
	{
		m_semaphores[handle.id] = vk::Semaphore{};
		m_free_semaphores.push_back(handle.id);
	}
}

vk::Semaphore Device::semaphore(SemaphoreHandle handle) const noexcept
{
	return handle.id < m_semaphores.size() ? m_semaphores[handle.id] : vk::Semaphore{};
}

PipelineHandle Device::register_pipeline(vk::Pipeline pipeline, vk::PipelineLayout layout)
{
	if (!m_free_pipelines.empty())
	{
		const std::uint32_t id = m_free_pipelines.back();
		m_free_pipelines.pop_back();
		m_pipelines[id] = Pipeline{.pipeline = pipeline, .layout = layout};
		return PipelineHandle{.id = id};
	}
	m_pipelines.push_back(Pipeline{.pipeline = pipeline, .layout = layout});
	return PipelineHandle{.id = static_cast<std::uint32_t>(m_pipelines.size() - 1)};
}

void Device::release_pipeline(PipelineHandle handle) noexcept
{
	if (handle.id < m_pipelines.size())
	{
		m_pipelines[handle.id] = Pipeline{};
		m_free_pipelines.push_back(handle.id);
	}
}

vk::Pipeline Device::pipeline(PipelineHandle handle) const noexcept
{
	return handle.id < m_pipelines.size() ? m_pipelines[handle.id].pipeline : vk::Pipeline{};
}

vk::PipelineLayout Device::pipeline_layout(PipelineHandle handle) const noexcept
{
	return handle.id < m_pipelines.size() ? m_pipelines[handle.id].layout : vk::PipelineLayout{};
}

void Device::update_bind_group(const BindGroup& group, std::span<const BindGroupEntry> entries) const
{
	// Build the vk write list from the RHI entries, resolving each handle to its vk object. The
	// buffer/image infos are stored in side vectors reserved up front so the .back() addresses the
	// WriteDescriptorSet proxies point at stay stable.
	std::vector<vk::DescriptorBufferInfo> buffer_infos;
	std::vector<vk::DescriptorImageInfo>  image_infos;
	std::vector<vk::WriteDescriptorSet>	  writes;
	buffer_infos.reserve(entries.size());
	image_infos.reserve(entries.size());
	writes.reserve(entries.size());

	for (const BindGroupEntry& entry : entries)
	{
		auto write = vk::WriteDescriptorSet()
						 .setDstSet(group.set)
						 .setDstBinding(entry.binding)
						 .setDstArrayElement(0)
						 .setDescriptorType(to_vk(entry.type));
		switch (entry.type)
		{
			case BindingType::UNIFORM_BUFFER:
			case BindingType::STORAGE_BUFFER:
				buffer_infos.push_back(vk::DescriptorBufferInfo()
										   .setBuffer(buffer(entry.buffer))
										   .setOffset(0)
										   .setRange(entry.buffer_size));
				write.setBufferInfo(buffer_infos.back());
				break;
			case BindingType::SAMPLED_IMAGE:
				image_infos.push_back(vk::DescriptorImageInfo()
										  .setImageView(view(entry.texture))
										  .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal));
				write.setImageInfo(image_infos.back());
				break;
			case BindingType::SAMPLER:
				image_infos.push_back(vk::DescriptorImageInfo().setSampler(sampler(entry.sampler)));
				write.setImageInfo(image_infos.back());
				break;
		}
		writes.push_back(write);
	}

	m_device.updateDescriptorSets(writes, {});
}
} // namespace veng::rhi
