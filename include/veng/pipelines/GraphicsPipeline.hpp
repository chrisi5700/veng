//
// Created by chris on 5/25/26.
//
// L2 construction layer — graphics pipeline (design.md §L2.1). A reflection-driven
// fluent builder turns a vertex + fragment `Shader` pair into an RAII, move-only
// `GraphicsPipeline`. Targets Vulkan 1.3 dynamic rendering: pipelines are built with
// attachment *formats*, not a VkRenderPass.
//

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
/// RAII owner of a graphics pipeline plus the layout and descriptor-set layout it was
/// built with. Move-only; constructed via `GraphicsPipelineBuilder`.
class GraphicsPipeline
{
	 public:
	GraphicsPipeline(const GraphicsPipeline&)			 = delete;
	GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
	GraphicsPipeline(GraphicsPipeline&& other) noexcept;
	GraphicsPipeline& operator=(GraphicsPipeline&& other) noexcept;
	~GraphicsPipeline();

	[[nodiscard]] vk::Pipeline			  pipeline() const noexcept { return m_pipeline; }
	[[nodiscard]] vk::PipelineLayout	  layout() const noexcept { return m_pipeline_layout; }
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

/// Fluent, reflection-driven builder (design.md §L2.1). Vertex input, the descriptor
/// set layout (merged across stages), and push-constant ranges are derived from the
/// shaders' reflection; raster/depth/blend take sensible defaults overridable here.
/// Viewport and scissor are dynamic state. At least one color format is required.
class GraphicsPipelineBuilder
{
	 public:
	GraphicsPipelineBuilder(const Shader& vertex, const Shader& fragment) noexcept
		: m_vertex(&vertex)
		, m_fragment(&fragment)
	{
	}

	GraphicsPipelineBuilder& color_formats(std::span<const vk::Format> formats);
	GraphicsPipelineBuilder& depth_format(vk::Format format);
	GraphicsPipelineBuilder& depth_write(bool enabled);
	GraphicsPipelineBuilder& topology(vk::PrimitiveTopology topology);
	GraphicsPipelineBuilder& rasterization(vk::PolygonMode polygon, vk::CullModeFlags cull, vk::FrontFace front);

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
};
} // namespace veng

#endif // VENG_GRAPHICSPIPELINE_HPP
