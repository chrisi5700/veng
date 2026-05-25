//
// Created by chris on 1/23/26.
//
// See ComputePipeline.hpp and design.md §L2.1.
//

#include <ranges>
#include <vector>
#include <veng/pipelines/ComputePipeline.hpp>

namespace v = std::views;
namespace r = std::ranges;

namespace veng
{
ComputePipeline::ComputePipeline(vk::Device device, vk::DescriptorSetLayout descriptor_set_layout,
								 vk::PipelineLayout pipeline_layout, vk::Pipeline pipeline) noexcept
	: m_device(device)
	, m_descriptor_set_layout(descriptor_set_layout)
	, m_pipeline_layout(pipeline_layout)
	, m_pipeline(pipeline)
{
}

void ComputePipeline::destroy() noexcept
{
	if (!m_device)
	{
		return;
	}
	if (m_pipeline)
	{
		m_device.destroyPipeline(m_pipeline);
	}
	if (m_pipeline_layout)
	{
		m_device.destroyPipelineLayout(m_pipeline_layout);
	}
	if (m_descriptor_set_layout)
	{
		m_device.destroyDescriptorSetLayout(m_descriptor_set_layout);
	}
	m_device				= nullptr;
	m_descriptor_set_layout = nullptr;
	m_pipeline_layout		= nullptr;
	m_pipeline				= nullptr;
}

ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
	: m_device(other.m_device)
	, m_descriptor_set_layout(other.m_descriptor_set_layout)
	, m_pipeline_layout(other.m_pipeline_layout)
	, m_pipeline(other.m_pipeline)
{
	other.m_device				  = nullptr;
	other.m_descriptor_set_layout = nullptr;
	other.m_pipeline_layout		  = nullptr;
	other.m_pipeline			  = nullptr;
}

ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept
{
	if (this != &other)
	{
		destroy();
		m_device					  = other.m_device;
		m_descriptor_set_layout		  = other.m_descriptor_set_layout;
		m_pipeline_layout			  = other.m_pipeline_layout;
		m_pipeline					  = other.m_pipeline;
		other.m_device				  = nullptr;
		other.m_descriptor_set_layout = nullptr;
		other.m_pipeline_layout		  = nullptr;
		other.m_pipeline			  = nullptr;
	}
	return *this;
}

ComputePipeline::~ComputePipeline()
{
	destroy();
}

std::expected<ComputePipeline, PipelineError> ComputePipelineBuilder::build(const Context&	  context,
																			vk::PipelineCache cache) const
{
	if (m_shader->get_details().stage() != vk::ShaderStageFlagBits::eCompute)
	{
		return std::unexpected(PipelineError::WRONG_STAGE);
	}

	const vk::Device device = context.device();

	// Descriptor-set layout straight from reflection (design.md §L2.1): no
	// hand-written bindings, so the layout can never drift from the shader.
	const auto bindings = m_shader->get_descriptor_infos() |
						  v::transform(
							  [](const DescriptorInfo& info)
							  {
								  return vk::DescriptorSetLayoutBinding()
									  .setBinding(static_cast<std::uint32_t>(info.binding))
									  .setDescriptorType(info.type)
									  .setDescriptorCount(static_cast<std::uint32_t>(info.descriptor_count))
									  .setStageFlags(vk::ShaderStageFlagBits::eCompute);
							  }) |
						  r::to<std::vector>();

	const auto descriptor_set_layout =
		device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo().setBindings(bindings));
	if (descriptor_set_layout.result != vk::Result::eSuccess)
	{
		return std::unexpected(PipelineError::DESCRIPTOR_SET_LAYOUT_CREATION_FAILED);
	}

	// Push-constant range from reflection, if the shader declares one.
	std::vector<vk::PushConstantRange> push_ranges;
	if (const auto& push_constant = m_shader->get_push_constant_info(); push_constant.has_value())
	{
		push_ranges.push_back(vk::PushConstantRange()
								  .setStageFlags(vk::ShaderStageFlagBits::eCompute)
								  .setOffset(static_cast<std::uint32_t>(push_constant->offset))
								  .setSize(static_cast<std::uint32_t>(push_constant->size)));
	}

	const auto pipeline_layout = device.createPipelineLayout(
		vk::PipelineLayoutCreateInfo().setSetLayouts(descriptor_set_layout.value).setPushConstantRanges(push_ranges));
	if (pipeline_layout.result != vk::Result::eSuccess)
	{
		device.destroyDescriptorSetLayout(descriptor_set_layout.value);
		return std::unexpected(PipelineError::PIPELINE_LAYOUT_CREATION_FAILED);
	}

	const auto stage_info = m_shader->create_pipeline_shader_stage_create_info();
	const auto pipeline	  = device.createComputePipeline(
		cache, vk::ComputePipelineCreateInfo().setStage(stage_info).setLayout(pipeline_layout.value));
	if (pipeline.result != vk::Result::eSuccess)
	{
		device.destroyPipelineLayout(pipeline_layout.value);
		device.destroyDescriptorSetLayout(descriptor_set_layout.value);
		return std::unexpected(PipelineError::PIPELINE_CREATION_FAILED);
	}

	return ComputePipeline(device, descriptor_set_layout.value, pipeline_layout.value, pipeline.value);
}
} // namespace veng
