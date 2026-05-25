//
// Created by chris on 1/23/26.
//
// L2 construction layer — compute pipeline (design.md §L2.1). A reflection-driven
// builder turns a compute `Shader` into an RAII, move-only `ComputePipeline`,
// reporting failure via std::expected rather than asserts.
//

#ifndef VENG_COMPUTEPIPELINE_HPP
#define VENG_COMPUTEPIPELINE_HPP

#include <expected>
#include <vulkan/vulkan.hpp>

#include "veng/context/Context.hpp"
#include "veng/pipelines/PipelineError.hpp"
#include "veng/shader/Shader.hpp"

namespace veng
{
/// RAII owner of a compute pipeline plus the layout and descriptor-set layout it
/// was built with (design.md §L2.1). Move-only; constructed via
/// `ComputePipelineBuilder`. Descriptor *allocation* is a separate concern
/// (DescriptorAllocator, §L2.2) and intentionally not owned here.
class ComputePipeline
{
	 public:
	ComputePipeline(const ComputePipeline&)			   = delete;
	ComputePipeline& operator=(const ComputePipeline&) = delete;
	ComputePipeline(ComputePipeline&& other) noexcept;
	ComputePipeline& operator=(ComputePipeline&& other) noexcept;
	~ComputePipeline();

	[[nodiscard]] vk::Pipeline			  pipeline() const noexcept { return m_pipeline; }
	[[nodiscard]] vk::PipelineLayout	  layout() const noexcept { return m_pipeline_layout; }
	[[nodiscard]] vk::DescriptorSetLayout descriptor_set_layout() const noexcept { return m_descriptor_set_layout; }

	 private:
	friend class ComputePipelineBuilder;
	ComputePipeline(vk::Device device, vk::DescriptorSetLayout descriptor_set_layout,
					vk::PipelineLayout pipeline_layout, vk::Pipeline pipeline) noexcept;
	void destroy() noexcept;

	vk::Device				m_device;
	vk::DescriptorSetLayout m_descriptor_set_layout;
	vk::PipelineLayout		m_pipeline_layout;
	vk::Pipeline			m_pipeline;
};

/// Reflection-driven builder (design.md §L2.1): derives the descriptor-set layout
/// and push-constant ranges from the shader's reflection, removing the create-info
/// boilerplate. A shared `vk::PipelineCache` may be passed to reuse compilation
/// work across rebuilds.
class ComputePipelineBuilder
{
	 public:
	explicit ComputePipelineBuilder(const Shader& compute) noexcept
		: m_shader(&compute)
	{
	}

	[[nodiscard]] std::expected<ComputePipeline, PipelineError> build(const Context&	context,
																	  vk::PipelineCache cache = {}) const;

	 private:
	const Shader* m_shader;
};
} // namespace veng

#endif // VENG_COMPUTEPIPELINE_HPP
