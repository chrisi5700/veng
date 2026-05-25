//
// Created by chris on 5/25/26.
//
// L2 construction layer — shared pipeline-builder error type (design.md §L2.1, §9).
// Used by both the compute and graphics pipeline builders.
//

#ifndef VENG_PIPELINEERROR_HPP
#define VENG_PIPELINEERROR_HPP

#include <cstdint>
#include <string_view>

namespace veng
{
enum class PipelineError : std::uint8_t
{
	WRONG_STAGE,						   // a shader is not the stage the builder expects
	STAGE_INCOMPATIBLE,					   // adjacent stage interfaces do not match (reflection)
	MISSING_COLOR_FORMATS,				   // a graphics pipeline was built with no color formats
	DESCRIPTOR_SET_LAYOUT_CREATION_FAILED, // vkCreateDescriptorSetLayout failed
	PIPELINE_LAYOUT_CREATION_FAILED,	   // vkCreatePipelineLayout failed
	PIPELINE_CREATION_FAILED,			   // vkCreate{Compute,Graphics}Pipelines failed
};

[[nodiscard]] constexpr std::string_view to_string(PipelineError error) noexcept
{
	switch (error)
	{
		case PipelineError::WRONG_STAGE: return "shader is not the expected stage";
		case PipelineError::STAGE_INCOMPATIBLE: return "adjacent shader stage interfaces do not match";
		case PipelineError::MISSING_COLOR_FORMATS: return "graphics pipeline requires at least one color format";
		case PipelineError::DESCRIPTOR_SET_LAYOUT_CREATION_FAILED: return "failed to create descriptor set layout";
		case PipelineError::PIPELINE_LAYOUT_CREATION_FAILED: return "failed to create pipeline layout";
		case PipelineError::PIPELINE_CREATION_FAILED: return "failed to create pipeline";
	}
	return "unknown pipeline error";
}
} // namespace veng

#endif // VENG_PIPELINEERROR_HPP
