//
// Created by chris on 5/25/26.
//
// See GraphicsPipeline.hpp and design.md §L2.1.
//

#include <array>
#include <map>
#include <variant>
#include <veng/pipelines/GraphicsPipeline.hpp>

namespace veng
{
GraphicsPipeline::GraphicsPipeline(vk::Device device, vk::DescriptorSetLayout descriptor_set_layout,
								   vk::PipelineLayout pipeline_layout, vk::Pipeline pipeline) noexcept
	: m_device(device)
	, m_descriptor_set_layout(descriptor_set_layout)
	, m_pipeline_layout(pipeline_layout)
	, m_pipeline(pipeline)
{
}

void GraphicsPipeline::destroy() noexcept
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

GraphicsPipeline::GraphicsPipeline(GraphicsPipeline&& other) noexcept
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

GraphicsPipeline& GraphicsPipeline::operator=(GraphicsPipeline&& other) noexcept
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

GraphicsPipeline::~GraphicsPipeline()
{
	destroy();
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::color_formats(std::span<const vk::Format> formats)
{
	m_color_formats.assign(formats.begin(), formats.end());
	return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::depth_format(vk::Format format)
{
	m_depth_format = format;
	return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::topology(vk::PrimitiveTopology topology)
{
	m_topology = topology;
	return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::rasterization(vk::PolygonMode polygon, vk::CullModeFlags cull,
																vk::FrontFace front)
{
	m_polygon	 = polygon;
	m_cull		 = cull;
	m_front_face = front;
	return *this;
}

std::expected<GraphicsPipeline, PipelineError> GraphicsPipelineBuilder::build(const Context&	context,
																			  vk::PipelineCache cache) const
{
	if (m_vertex->get_details().stage() != vk::ShaderStageFlagBits::eVertex ||
		m_fragment->get_details().stage() != vk::ShaderStageFlagBits::eFragment)
	{
		return std::unexpected(PipelineError::WRONG_STAGE);
	}
	// Reflection-checked interface compatibility (vertex outputs -> fragment inputs).
	if (!m_vertex->get_details().matches(m_fragment->get_details()))
	{
		return std::unexpected(PipelineError::STAGE_INCOMPATIBLE);
	}
	if (m_color_formats.empty())
	{
		return std::unexpected(PipelineError::MISSING_COLOR_FORMATS);
	}

	const vk::Device device = context.device();

	// Descriptor-set layout merged across both stages by binding, OR-ing stage flags
	// (single set 0 for v1, consistent with ComputePipeline). Ordered map keeps the
	// bindings sorted, which Vulkan does not require but keeps the layout deterministic.
	std::map<std::uint32_t, vk::DescriptorSetLayoutBinding> merged;
	const auto												collect = [&merged](const Shader& shader)
	{
		for (const DescriptorInfo& info : shader.get_descriptor_infos())
		{
			const auto binding = static_cast<std::uint32_t>(info.binding);
			if (auto it = merged.find(binding); it != merged.end())
			{
				it->second.stageFlags |= info.stage;
			}
			else
			{
				merged.emplace(binding, vk::DescriptorSetLayoutBinding()
											.setBinding(binding)
											.setDescriptorType(info.type)
											.setDescriptorCount(static_cast<std::uint32_t>(info.descriptor_count))
											.setStageFlags(info.stage));
			}
		}
	};
	collect(*m_vertex);
	collect(*m_fragment);

	std::vector<vk::DescriptorSetLayoutBinding> bindings;
	bindings.reserve(merged.size());
	for (const auto& [binding, layout_binding] : merged)
	{
		bindings.push_back(layout_binding);
	}

	const auto descriptor_set_layout =
		device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo().setBindings(bindings));
	if (descriptor_set_layout.result != vk::Result::eSuccess)
	{
		return std::unexpected(PipelineError::DESCRIPTOR_SET_LAYOUT_CREATION_FAILED);
	}

	std::vector<vk::PushConstantRange> push_ranges;
	const auto						   collect_push = [&push_ranges](const Shader& shader)
	{
		if (const auto& push = shader.get_push_constant_info(); push.has_value())
		{
			push_ranges.push_back(vk::PushConstantRange()
									  .setStageFlags(push->stage)
									  .setOffset(static_cast<std::uint32_t>(push->offset))
									  .setSize(static_cast<std::uint32_t>(push->size)));
		}
	};
	collect_push(*m_vertex);
	collect_push(*m_fragment);

	const auto pipeline_layout = device.createPipelineLayout(
		vk::PipelineLayoutCreateInfo().setSetLayouts(descriptor_set_layout.value).setPushConstantRanges(push_ranges));
	if (pipeline_layout.result != vk::Result::eSuccess)
	{
		device.destroyDescriptorSetLayout(descriptor_set_layout.value);
		return std::unexpected(PipelineError::PIPELINE_LAYOUT_CREATION_FAILED);
	}

	const std::array stages{m_vertex->create_pipeline_shader_stage_create_info(),
							m_fragment->create_pipeline_shader_stage_create_info()};

	// Vertex input straight from the vertex shader's reflection.
	std::vector<vk::VertexInputBindingDescription>	 vertex_bindings;
	std::vector<vk::VertexInputAttributeDescription> vertex_attributes;
	if (const auto* vertex_details = std::get_if<VertexDetails>(&m_vertex->get_details()))
	{
		for (const VertexBinding& binding : vertex_details->bindings)
		{
			vertex_bindings.push_back(binding.to_binding_description(vk::VertexInputRate::eVertex));
		}
		for (const VertexAttribute& attribute : vertex_details->inputs)
		{
			vertex_attributes.push_back(attribute.to_attribute_description());
		}
	}
	const auto vertex_input = vk::PipelineVertexInputStateCreateInfo()
								  .setVertexBindingDescriptions(vertex_bindings)
								  .setVertexAttributeDescriptions(vertex_attributes);

	const auto input_assembly = vk::PipelineInputAssemblyStateCreateInfo().setTopology(m_topology);
	const auto viewport_state = vk::PipelineViewportStateCreateInfo().setViewportCount(1).setScissorCount(1);
	const auto rasterization  = vk::PipelineRasterizationStateCreateInfo()
									.setPolygonMode(m_polygon)
									.setCullMode(m_cull)
									.setFrontFace(m_front_face)
									.setLineWidth(1.0F);
	const auto multisample =
		vk::PipelineMultisampleStateCreateInfo().setRasterizationSamples(vk::SampleCountFlagBits::e1);

	const bool has_depth	 = m_depth_format != vk::Format::eUndefined;
	const auto depth_stencil = vk::PipelineDepthStencilStateCreateInfo()
								   .setDepthTestEnable(static_cast<vk::Bool32>(has_depth))
								   .setDepthWriteEnable(static_cast<vk::Bool32>(has_depth))
								   .setDepthCompareOp(vk::CompareOp::eLess);

	// One non-blended, write-all attachment per color format.
	const std::vector<vk::PipelineColorBlendAttachmentState> blend_attachments(
		m_color_formats.size(), vk::PipelineColorBlendAttachmentState().setColorWriteMask(
									vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
									vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA));
	const auto color_blend = vk::PipelineColorBlendStateCreateInfo().setAttachments(blend_attachments);

	const std::array dynamic_states{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
	const auto		 dynamic_state = vk::PipelineDynamicStateCreateInfo().setDynamicStates(dynamic_states);

	// Dynamic rendering: attachment formats instead of a VkRenderPass (Vulkan 1.3).
	auto rendering = vk::PipelineRenderingCreateInfo().setColorAttachmentFormats(m_color_formats);
	if (has_depth)
	{
		rendering.setDepthAttachmentFormat(m_depth_format);
	}

	const auto info = vk::GraphicsPipelineCreateInfo()
						  .setPNext(&rendering)
						  .setStages(stages)
						  .setPVertexInputState(&vertex_input)
						  .setPInputAssemblyState(&input_assembly)
						  .setPViewportState(&viewport_state)
						  .setPRasterizationState(&rasterization)
						  .setPMultisampleState(&multisample)
						  .setPDepthStencilState(&depth_stencil)
						  .setPColorBlendState(&color_blend)
						  .setPDynamicState(&dynamic_state)
						  .setLayout(pipeline_layout.value);

	const auto pipeline = device.createGraphicsPipeline(cache, info);
	if (pipeline.result != vk::Result::eSuccess)
	{
		device.destroyPipelineLayout(pipeline_layout.value);
		device.destroyDescriptorSetLayout(descriptor_set_layout.value);
		return std::unexpected(PipelineError::PIPELINE_CREATION_FAILED);
	}

	return GraphicsPipeline(device, descriptor_set_layout.value, pipeline_layout.value, pipeline.value);
}
} // namespace veng
