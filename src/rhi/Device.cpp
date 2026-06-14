/**
 * @file
 * @author chris
 * @brief @ref veng::rhi::Device implementation — the texture/buffer handle slot-maps.
 * @ingroup rhi
 */

#include <expected>
#include <vector>
#include <veng/rhi/Convert.hpp>
#include <veng/rhi/Device.hpp>

namespace veng::rhi
{
Device::~Device()
{
	for (const vk::Sampler smp : m_samplers)
	{
		if (smp)
		{
			m_device.destroySampler(smp);
		}
	}
}

std::expected<SamplerHandle, vk::Result> Device::create_sampler(const vk::SamplerCreateInfo& info)
{
	const auto created = m_device.createSampler(info);
	if (created.result != vk::Result::eSuccess)
	{
		return std::unexpected(created.result);
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
	if (!m_free_textures.empty())
	{
		const std::uint32_t id = m_free_textures.back();
		m_free_textures.pop_back();
		m_textures[id] = Texture{.image = image, .view = view};
		return TextureHandle{.id = id};
	}
	m_textures.push_back(Texture{.image = image, .view = view});
	return TextureHandle{.id = static_cast<std::uint32_t>(m_textures.size() - 1)};
}

void Device::rebind_texture(TextureHandle handle, vk::Image image, vk::ImageView view) noexcept
{
	if (handle.id < m_textures.size())
	{
		m_textures[handle.id] = Texture{.image = image, .view = view};
	}
}

void Device::release_texture(TextureHandle handle) noexcept
{
	if (handle.id < m_textures.size())
	{
		m_textures[handle.id] = Texture{};
		m_free_textures.push_back(handle.id);
	}
}

vk::Image Device::image(TextureHandle handle) const noexcept
{
	return handle.id < m_textures.size() ? m_textures[handle.id].image : vk::Image{};
}

vk::ImageView Device::view(TextureHandle handle) const noexcept
{
	return handle.id < m_textures.size() ? m_textures[handle.id].view : vk::ImageView{};
}

BufferHandle Device::register_buffer(vk::Buffer buffer)
{
	if (!m_free_buffers.empty())
	{
		const std::uint32_t id = m_free_buffers.back();
		m_free_buffers.pop_back();
		m_buffers[id] = buffer;
		return BufferHandle{.id = id};
	}
	m_buffers.push_back(buffer);
	return BufferHandle{.id = static_cast<std::uint32_t>(m_buffers.size() - 1)};
}

void Device::release_buffer(BufferHandle handle) noexcept
{
	if (handle.id < m_buffers.size())
	{
		m_buffers[handle.id] = vk::Buffer{};
		m_free_buffers.push_back(handle.id);
	}
}

vk::Buffer Device::buffer(BufferHandle handle) const noexcept
{
	return handle.id < m_buffers.size() ? m_buffers[handle.id] : vk::Buffer{};
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
