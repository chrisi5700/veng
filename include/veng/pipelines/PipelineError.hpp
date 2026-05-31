/**
 * @file
 * @author chris
 * @brief Shared pipeline-builder error type used by both the compute and graphics pipeline builders.
 *
 * Both @ref veng::ComputePipelineBuilder and @ref veng::GraphicsPipelineBuilder return
 * `std::expected<…, PipelineError>` on failure. Having a single error enum here avoids
 * duplicating the same set of failure codes in each pipeline header.
 *
 * @ingroup pipelines
 */

#ifndef VENG_PIPELINEERROR_HPP
#define VENG_PIPELINEERROR_HPP

#include <cstdint>
#include <string_view>

namespace veng
{
/**
 * @brief Error codes returned by @ref veng::ComputePipelineBuilder::build and
 *        @ref veng::GraphicsPipelineBuilder::build.
 * @ingroup pipelines
 */
enum class PipelineError : std::uint8_t
{
	WRONG_STAGE,						   ///< A shader is not the stage the builder expects.
	STAGE_INCOMPATIBLE,					   ///< Adjacent stage interfaces do not match (reflection check).
	MISSING_COLOR_FORMATS,				   ///< A graphics pipeline was built with no color formats.
	DESCRIPTOR_SET_LAYOUT_CREATION_FAILED, ///< `vkCreateDescriptorSetLayout` failed.
	PIPELINE_LAYOUT_CREATION_FAILED,	   ///< `vkCreatePipelineLayout` failed.
	PIPELINE_CREATION_FAILED,			   ///< `vkCreate{Compute,Graphics}Pipelines` failed.
};

/**
 * @brief Stringify a @ref veng::PipelineError for logging and error reporting.
 * @param error The error code to render.
 * @return A human-readable description of the error.
 */
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
