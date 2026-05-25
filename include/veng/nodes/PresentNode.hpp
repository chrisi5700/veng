//
// Created by chris on 5/25/26.
//
// L4 node — present/blit (design.md §L4 swapchain & present). Copies the cached
// scene-color target into the frame's acquired image. It depends on BOTH the scene
// token AND the swapchain source; only the swapchain source is dirtied each frame, so
// this node re-runs every frame (microseconds) while the scene render stays cached.
//
// The load-bearing wiring rule: the scene subgraph must NOT depend on the swapchain
// source — only present does. (design.md §L4.)
//

#ifndef VENG_PRESENTNODE_HPP
#define VENG_PRESENTNODE_HPP

#include <array>
#include <cstddef>
#include <expected>
#include <span>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/nodes/RasterTriangleNode.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <veng/resources/Image.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::nodes
{
class PresentNode final : public gpu::GpuNode
{
	 public:
	/// `scene` produces the cached scene-color target (read fresh each frame so a
	/// resize is picked up). `target` is the frame's acquired image to present into
	/// (offscreen stand-in for a swapchain image; non-owning). `scene_token` and
	/// `swapchain_source` are this node's two inputs; `output` is the present token.
	PresentNode(const RasterTriangleNode& scene, Image& target, graph::DataHandle scene_token,
				graph::DataHandle swapchain_source, graph::DataHandle output) noexcept;

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return m_inputs; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	/// How many times this node has actually recorded — the lens for the caching proof.
	[[nodiscard]] std::size_t record_count() const noexcept { return m_record_count; }

	 protected:
	[[nodiscard]] std::expected<bool, graph::ExecError> record(gpu::GpuExecContext& ctx) override;

	 private:
	const RasterTriangleNode*		 m_scene;
	Image*							 m_target;
	std::array<graph::DataHandle, 2> m_inputs; // [scene_token, swapchain_source]
	graph::DataHandle				 m_output;
	std::size_t						 m_record_count = 0;
};
} // namespace veng::nodes

#endif // VENG_PRESENTNODE_HPP
