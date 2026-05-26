//
// Created by chris on 5/25/26.
//
// L4 node — a GPU raster node that draws a triangle into a persistent scene-color
// target sized from a ScreenSize source (design.md §L4). The first node that actually
// renders. It owns the target Image (re)created from the demanded ScreenSize, so a
// resize is an ordinary invalidation cascade. Leaves the target in TRANSFER_SRC layout
// ready for the present/blit node (or a readback).
//

#ifndef VENG_RASTERTRIANGLENODE_HPP
#define VENG_RASTERTRIANGLENODE_HPP

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
class RasterTriangleNode final : public gpu::GpuNode
{
	 public:
	/// `pipeline` must have been built for `color_format` (one color attachment).
	/// `screen_size` is a `ValueData<vk::Extent2D>` source; `output` is the scene
	/// `ImageData` (a ref to the rendered target, left in TRANSFER_SRC) the BlitNode reads.
	RasterTriangleNode(GraphicsPipeline pipeline, vk::Format color_format, graph::DataHandle screen_size,
					   graph::DataHandle output) noexcept;

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return {&m_screen_size, 1}; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	/// The persistent scene-color target, or nullptr before the first render (test lens).
	[[nodiscard]] const Image* scene() const noexcept { return m_scene.has_value() ? &m_scene.value() : nullptr; }

	 protected:
	[[nodiscard]] std::expected<bool, graph::ExecError> record(gpu::GpuExecContext& ctx) override;

	 private:
	GraphicsPipeline	 m_pipeline;
	vk::Format			 m_color_format;
	graph::DataHandle	 m_screen_size;
	graph::DataHandle	 m_output;
	std::optional<Image> m_scene;
	vk::Extent2D		 m_scene_extent{};
};
} // namespace veng::nodes

#endif // VENG_RASTERTRIANGLENODE_HPP
