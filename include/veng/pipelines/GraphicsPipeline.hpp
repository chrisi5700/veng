/**
 * @file
 * @author chris
 * @brief Reflection-driven graphics pipeline builder that turns a vertex + fragment
 *        @ref veng::Shader pair into an RAII, move-only @ref veng::GraphicsPipeline.
 *
 * This is the L2 construction layer for graphics pipelines. Pipelines target Vulkan 1.3
 * dynamic rendering: they are built with attachment formats rather than a `VkRenderPass`.
 * The builder derives vertex input descriptions, the descriptor-set layout (merged across
 * both stages), and push-constant ranges from SPIR-V reflection. Rasterizer, depth, and
 * blend state take sensible defaults that are overridable via the fluent API.
 *
 * @ingroup pipelines
 */

#ifndef VENG_GRAPHICSPIPELINE_HPP
#define VENG_GRAPHICSPIPELINE_HPP

#include <expected>
#include <span>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "veng/context/Context.hpp"
#include "veng/pipelines/PipelineError.hpp"
#include "veng/shader/Shader.hpp"

namespace veng
{
/**
 * @brief RAII owner of a graphics pipeline plus the layout and descriptor-set layout it
 *        was built with.
 *
 * Move-only; constructed exclusively via @ref veng::GraphicsPipelineBuilder.
 *
 * @ingroup pipelines
 * @see GraphicsPipelineBuilder
 * @see DescriptorAllocator
 */
class GraphicsPipeline
{
	 public:
	GraphicsPipeline(const GraphicsPipeline&)			 = delete;
	GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
	GraphicsPipeline(GraphicsPipeline&& other) noexcept;
	GraphicsPipeline& operator=(GraphicsPipeline&& other) noexcept;
	~GraphicsPipeline();

	/** @brief The underlying `vkPipeline` handle. */
	[[nodiscard]] vk::Pipeline pipeline() const noexcept { return m_pipeline; }
	/** @brief The pipeline layout (push-constant ranges + descriptor set layout). */
	[[nodiscard]] vk::PipelineLayout layout() const noexcept { return m_pipeline_layout; }
	/** @brief The descriptor-set layout merged from vertex and fragment stage reflections. */
	[[nodiscard]] vk::DescriptorSetLayout descriptor_set_layout() const noexcept { return m_descriptor_set_layout; }

	 private:
	friend class GraphicsPipelineBuilder;
	GraphicsPipeline(vk::Device device, vk::DescriptorSetLayout descriptor_set_layout,
					 vk::PipelineLayout pipeline_layout, vk::Pipeline pipeline) noexcept;
	void destroy() noexcept;

	vk::Device				m_device;
	vk::DescriptorSetLayout m_descriptor_set_layout;
	vk::PipelineLayout		m_pipeline_layout;
	vk::Pipeline			m_pipeline;
};

/**
 * @brief Fluent, reflection-driven builder that constructs a @ref veng::GraphicsPipeline from a
 *        vertex + fragment @ref veng::Shader pair.
 *
 * Vertex input descriptions, the descriptor-set layout (merged across both stages), and
 * push-constant ranges are derived from SPIR-V reflection. Rasterizer, depth, and blend
 * state take sensible defaults that are overridable via the fluent setter methods. Viewport
 * and scissor are always dynamic state. At least one color format must be set before
 * calling `build`.
 *
 * @ingroup pipelines
 * @see GraphicsPipeline
 * @see PipelineError
 */
class GraphicsPipelineBuilder
{
	 public:
	/**
	 * @brief Construct a builder for the given vertex + fragment shader pair.
	 * @param vertex   A loaded and reflected vertex-stage @ref veng::Shader.
	 * @param fragment A loaded and reflected fragment-stage @ref veng::Shader.
	 */
	GraphicsPipelineBuilder(const Shader& vertex, const Shader& fragment) noexcept
		: m_vertex(&vertex)
		, m_fragment(&fragment)
	{
	}

	/**
	 * @brief Set the color attachment formats (required; at least one must be provided).
	 * @param formats The `vk::Format` of each color attachment, in attachment order.
	 * @return `*this` for chaining.
	 */
	GraphicsPipelineBuilder& color_formats(std::span<const vk::Format> formats);

	/**
	 * @brief Set the depth attachment format (default: `eUndefined` — no depth).
	 * @param format The `vk::Format` of the depth attachment.
	 * @return `*this` for chaining.
	 */
	GraphicsPipelineBuilder& depth_format(vk::Format format);

	/**
	 * @brief Enable or disable depth writes (default: `true`).
	 * @param enabled Whether depth writes are enabled.
	 * @return `*this` for chaining.
	 */
	GraphicsPipelineBuilder& depth_write(bool enabled);

	/**
	 * @brief Set the primitive topology (default: `eTriangleList`).
	 * @param topology The desired primitive topology.
	 * @return `*this` for chaining.
	 */
	GraphicsPipelineBuilder& topology(vk::PrimitiveTopology topology);

	/**
	 * @brief Override rasterizer state (defaults: fill / back-cull / CCW).
	 * @param polygon The polygon rasterization mode.
	 * @param cull    The cull mode flags.
	 * @param front   The front-face winding order.
	 * @return `*this` for chaining.
	 */
	GraphicsPipelineBuilder& rasterization(vk::PolygonMode polygon, vk::CullModeFlags cull, vk::FrontFace front);

	/**
	 * @brief Enable or disable straight-alpha blending on every color attachment
	 *        (`src*srcA + dst*(1-srcA)`).
	 *
	 * Off by default (opaque, write-all). Enable this for transparent passes.
	 *
	 * @param enabled Whether alpha blending is enabled.
	 * @return `*this` for chaining.
	 */
	GraphicsPipelineBuilder& blend(bool enabled);

	/**
	 * @brief Build the graphics pipeline from the configured state.
	 * @param context The engine context that owns the `vk::Device`.
	 * @param cache   An optional pipeline cache for reusing compiled shaders.
	 * @return The fully constructed @ref veng::GraphicsPipeline, or a @ref veng::PipelineError on failure.
	 * @retval PipelineError::WRONG_STAGE    Either shader is not the expected stage.
	 * @retval PipelineError::STAGE_INCOMPATIBLE Vertex outputs and fragment inputs do not match.
	 * @retval PipelineError::MISSING_COLOR_FORMATS No color formats were set.
	 * @retval PipelineError::DESCRIPTOR_SET_LAYOUT_CREATION_FAILED `vkCreateDescriptorSetLayout` failed.
	 * @retval PipelineError::PIPELINE_LAYOUT_CREATION_FAILED `vkCreatePipelineLayout` failed.
	 * @retval PipelineError::PIPELINE_CREATION_FAILED `vkCreateGraphicsPipelines` failed.
	 */
	[[nodiscard]] std::expected<GraphicsPipeline, PipelineError> build(const Context&	 context,
																	   vk::PipelineCache cache = {}) const;

	 private:
	const Shader*			m_vertex;
	const Shader*			m_fragment;
	std::vector<vk::Format> m_color_formats;
	vk::Format				m_depth_format = vk::Format::eUndefined;
	bool					m_depth_write  = true;
	vk::PrimitiveTopology	m_topology	   = vk::PrimitiveTopology::eTriangleList;
	vk::PolygonMode			m_polygon	   = vk::PolygonMode::eFill;
	vk::CullModeFlags		m_cull		   = vk::CullModeFlagBits::eBack;
	vk::FrontFace			m_front_face   = vk::FrontFace::eCounterClockwise;
	bool					m_blend		   = false;
};
} // namespace veng

#endif // VENG_GRAPHICSPIPELINE_HPP
