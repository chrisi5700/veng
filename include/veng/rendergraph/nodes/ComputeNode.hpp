//
// Created by chris on 1/23/26.
//

#ifndef VENG_COMPUTEPIPELINE_HPP
#define VENG_COMPUTEPIPELINE_HPP
#include "veng/shader/Shader.hpp"
class ComputeNode
{
	veng::Shader m_compute_shader;
	vk::DescriptorPool m_descriptor_pool;
	vk::DescriptorSetLayout m_descriptor_set_layout;
	vk::DescriptorSet m_descriptor_set;
	vk::PipelineLayout m_pipeline_layout;
	vk::Pipeline m_pipeline;


	vk::CommandPool m_compute_command_pool;
	vk::CommandBuffer m_compute_command_buffer;
	vk::Fence m_compute_fence;
	vk::Queue m_queue;
	static ComputeNode create(veng::Shader shader, veng::Context& context);
};
#endif // VENG_COMPUTEPIPELINE_HPP
