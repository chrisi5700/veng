/**
 * @file
 * @author chris
 * @brief Reflection-driven compute pipeline builder that turns a compute @ref veng::Shader into an
 *        RAII, move-only @ref veng::ComputePipeline, reporting failure via `std::expected`.
 *
 * This is the L2 construction layer for compute pipelines. The builder derives the
 * descriptor-set layout and push-constant ranges directly from the shader's SPIR-V
 * reflection, so there is no hand-written create-info boilerplate and the layout can
 * never drift from the shader source. Descriptor allocation is a separate concern
 * handled by @ref veng::DescriptorAllocator.
 *
 * @ingroup pipelines
 */

#ifndef VENG_COMPUTEPIPELINE_HPP
#define VENG_COMPUTEPIPELINE_HPP

#include <expected>
#include <vulkan/vulkan.hpp>

#include "veng/context/Context.hpp"
#include "veng/pipelines/PipelineError.hpp"
#include "veng/shader/Shader.hpp"

namespace veng
{
/**
 * @brief RAII owner of a compute pipeline plus the layout and descriptor-set layout it
 *        was built with.
 *
 * Move-only; constructed exclusively via @ref veng::ComputePipelineBuilder. Descriptor
 * allocation is a separate concern (@ref veng::DescriptorAllocator) and is intentionally
 * not owned here.
 *
 * @ingroup pipelines
 * @see ComputePipelineBuilder
 * @see DescriptorAllocator
 */
class ComputePipeline
{
	 public:
	ComputePipeline(const ComputePipeline&)			   = delete;
	ComputePipeline& operator=(const ComputePipeline&) = delete;
	ComputePipeline(ComputePipeline&& other) noexcept;
	ComputePipeline& operator=(ComputePipeline&& other) noexcept;
	~ComputePipeline();

	/** @brief The underlying `vkPipeline` handle. */
	[[nodiscard]] vk::Pipeline pipeline() const noexcept { return m_pipeline; }
	/** @brief The pipeline layout (push-constant ranges + descriptor set layout). */
	[[nodiscard]] vk::PipelineLayout layout() const noexcept { return m_pipeline_layout; }
	/** @brief The descriptor-set layout derived from SPIR-V reflection. */
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

/**
 * @brief Reflection-driven builder that constructs a @ref veng::ComputePipeline from a single
 *        compute @ref veng::Shader.
 *
 * The descriptor-set layout and push-constant ranges are derived from the shader's SPIR-V
 * reflection, removing all create-info boilerplate.
 *
 * @ingroup pipelines
 * @see ComputePipeline
 * @see PipelineError
 */
class ComputePipelineBuilder
{
	 public:
	/**
	 * @brief Construct a builder for the given compute shader.
	 * @param compute A loaded and reflected compute-stage @ref veng::Shader.
	 */
	explicit ComputePipelineBuilder(const Shader& compute) noexcept
		: m_shader(&compute)
	{
	}

	/**
	 * @brief Build the compute pipeline, creating the descriptor-set layout, pipeline
	 *        layout, and pipeline from reflection data.
	 * @param context The engine context that owns the `vk::Device`.
	 * @return The fully constructed @ref veng::ComputePipeline, or a @ref veng::PipelineError on failure.
	 * @retval PipelineError::WRONG_STAGE The shader is not a compute stage.
	 * @retval PipelineError::DESCRIPTOR_SET_LAYOUT_CREATION_FAILED `vkCreateDescriptorSetLayout` failed.
	 * @retval PipelineError::PIPELINE_LAYOUT_CREATION_FAILED `vkCreatePipelineLayout` failed.
	 * @retval PipelineError::PIPELINE_CREATION_FAILED `vkCreateComputePipelines` failed.
	 */
	[[nodiscard]] std::expected<ComputePipeline, PipelineError> build(const Context& context) const;

	 private:
	const Shader* m_shader;
};
} // namespace veng

#endif // VENG_COMPUTEPIPELINE_HPP
