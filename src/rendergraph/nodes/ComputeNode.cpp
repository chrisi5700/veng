//
// Created by chris on 1/23/26.
//
#include <algorithm>
#include <ranges>
#include <veng/rendergraph//nodes/ComputeNode.hpp>


namespace v = std::views;
namespace r = std::ranges;

ComputeNode ComputeNode::create(veng::Shader shader, veng::Context& context)
{
	assert(shader.get_details().stage() == vk::ShaderStageFlagBits::eCompute);

	auto descriptor_set_layout_bindings = shader.get_descriptor_infos() | v::transform([](const veng::DescriptorInfo& info)
	{
		return vk::DescriptorSetLayoutBinding()
		.setBinding(info.binding)
		.setDescriptorType(info.type)
		.setDescriptorCount(info.descriptor_count)
		.setStageFlags(vk::ShaderStageFlagBits::eCompute);
	}) | r::to<std::vector>();

	auto descriptor_set_layout = context.device().createDescriptorSetLayout(
		vk::DescriptorSetLayoutCreateInfo().setBindings(descriptor_set_layout_bindings));
	assert(descriptor_set_layout.result == vk::Result::eSuccess);


	// TODO Create the descriptor pool with appropriate sizes for each descriptor set
	// TODO allocate descriptor sets with descriptor set layout
	//




	auto pipeline_layout = context.device().createPipelineLayout(
		vk::PipelineLayoutCreateInfo().setSetLayouts(descriptor_set_layout.value));
	assert(pipeline_layout.result == vk::Result::eSuccess);

	auto stage_info = vk::PipelineShaderStageCreateInfo().setStage(vk::ShaderStageFlagBits::eCompute)
	.setModule(shader.get_shader_module())
	.setPName(shader.get_entry_point().c_str());
	auto pipeline_info = vk::ComputePipelineCreateInfo()
		.setStage(stage_info)
		.setLayout(pipeline_layout.value);
	auto pipeline = context.device().createComputePipeline(nullptr, pipeline_info);

}