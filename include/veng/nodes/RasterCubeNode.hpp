//
// Created by chris on 5/25/26.
//
// L4 node — a depth-tested, rotating cube drawn into a persistent scene-color target
// sized from a ScreenSize source (design.md §L4). Its rotation is a reactive `float`
// angle input: when the angle changes the node re-renders; when it holds, the node is
// cached and the PresentNode re-presents the last frame unchanged. This is the
// demand-driven caching thesis (design.md §1) exercised with real per-frame GPU work.
//

#ifndef VENG_RASTERCUBENODE_HPP
#define VENG_RASTERCUBENODE_HPP

#include <array>
#include <expected>
#include <optional>
#include <span>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/pipelines/GraphicsPipeline.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <veng/resources/Image.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::nodes
{
class RasterCubeNode final : public gpu::GpuNode
{
	 public:
	/// `pipeline` must have been built for (`color_format`, `depth_format`). Inputs are
	/// a `ValueData<vk::Extent2D>` ScreenSize source and a `ValueData<float>` rotation
	/// angle; `output` is the scene `ImageData` (a ref to the rendered target, left in
	/// TRANSFER_SRC) the BlitNode reads.
	RasterCubeNode(GraphicsPipeline pipeline, vk::Format color_format, vk::Format depth_format,
				   graph::DataHandle screen_size, graph::DataHandle angle, graph::DataHandle output) noexcept;

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return m_inputs; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	/// The persistent scene-color target, or nullptr before the first render (test lens).
	[[nodiscard]] const Image* scene() const noexcept { return m_scene.has_value() ? &m_scene.value() : nullptr; }

	 protected:
	[[nodiscard]] std::expected<bool, graph::ExecError> record(gpu::GpuExecContext& ctx) override;

	 private:
	GraphicsPipeline				 m_pipeline;
	vk::Format						 m_color_format;
	vk::Format						 m_depth_format;
	std::array<graph::DataHandle, 2> m_inputs; // [screen_size, angle]
	graph::DataHandle				 m_output;
	std::optional<Image>			 m_scene;
	std::optional<Image>			 m_depth;
	vk::Extent2D					 m_extent{};
};
} // namespace veng::nodes

#endif // VENG_RASTERCUBENODE_HPP
