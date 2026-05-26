//
// Created by chris on 5/26/26.
//
// See UniformNode.hpp and design.md §L4.
//

#include <cstring>
#include <utility>
#include <veng/gpu/UniformRef.hpp>
#include <veng/nodes/UniformNode.hpp>

namespace veng::nodes
{
std::expected<bool, graph::ExecError> UniformNode::record(gpu::GpuExecContext& ctx)
{
	const void* bytes = m_read(ctx);
	if (bytes == nullptr)
	{
		return std::unexpected(graph::ExecError::MISSING_INPUT);
	}

	// Allocate the persistent host-visible uniform buffer once; thereafter just refresh its
	// contents in place (the handle stays stable, so the consumer's descriptor set is valid).
	if (!m_buffer.has_value())
	{
		auto buffer = Buffer::create(
			ctx.allocator(), m_size, vk::BufferUsageFlagBits::eUniformBuffer, vma::MemoryUsage::eAuto,
			vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessSequentialWrite);
		if (!buffer.has_value() || buffer->mapped() == nullptr)
		{
			return std::unexpected(graph::ExecError::NODE_FAILED);
		}
		m_buffer = std::move(buffer.value());
	}

	std::memcpy(m_buffer->mapped(), bytes, m_size);

	if (auto* out = dynamic_cast<graph::ValueData<gpu::UniformRef>*>(ctx.data(m_output)); out != nullptr)
	{
		(void)out->produce(gpu::UniformRef{.buffer = m_buffer->buffer(), .size = m_size, .name = m_name});
	}
	return true;
}
} // namespace veng::nodes
