//
// Created by chris on 5/26/26.
//
// L4 node — a *generic* raster node (design.md §L4). It is self-contained: you give it
// the names of a vertex + fragment shader, the attachment formats, a screen-size input
// that sizes its owned color (+ optional depth) target, and zero or more push-constant
// value edges. It builds its own pipeline internally (the first time it records, when it
// has a device) — graph construction never touches the L2 pipeline layer — then draws and
// publishes the rendered target as an `ImageData`. "The cube" is just this node wired to
// the cube shaders plus an `mvp` push-constant edge fed by a pure transform of the angle:
// no cube-specific node. A resize reallocates the target (an ordinary invalidation); a
// changed push-constant re-renders, an unchanged one is cached.
//
// Geometry is either buffer-backed — `set_mesh(handle)` binds a `gpu::MeshRef` produced by
// a MeshNode and the node draws it (indexed when the ref has an index buffer) — or, with no
// mesh bound, a `vertex_count` draw of vertices fabricated in the shader from SV_VertexID.
// `add_uniform(handle)` binds a `gpu::UniformRef` (from a UniformNode) into the node's own
// descriptor set, matched to the shader's reflected binding by name.
//

#ifndef VENG_GRAPHICSNODE_HPP
#define VENG_GRAPHICSNODE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>
#include <veng/descriptors/DescriptorAllocator.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/pipelines/GraphicsPipeline.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <veng/resources/Image.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::nodes
{
class GraphicsNode final : public gpu::GpuNode
{
	 public:
	/// `vertex_shader`/`fragment_shader` are shader names resolved from SHADER_DIR; the
	/// node loads them and builds its pipeline for (`color_format`[, `depth_format`]) on
	/// first record. Pass `eUndefined` for `depth_format` to render without depth.
	/// `screen_size` is a `ValueData<vk::Extent2D>` source that sizes the owned target;
	/// `output` is the scene `ImageData` (left in TRANSFER_SRC). `vertex_count` is the draw
	/// count (geometry comes from the vertex shader for now).
	GraphicsNode(std::string vertex_shader, std::string fragment_shader, vk::Format color_format,
				 vk::Format depth_format, std::uint32_t vertex_count, graph::DataHandle screen_size,
				 graph::DataHandle output) noexcept;

	/// Bind a push-constant value edge. `T` is the value type on `handle`; its bytes are
	/// pushed at `offset` for `stage` each time the node records. The edge is a reactive
	/// input — changing it re-renders the node. Returns *this for chaining; call before
	/// adding the node to the graph (it extends the input set).
	template <class T>
	GraphicsNode& push_constant(graph::DataHandle handle, vk::ShaderStageFlags stage = vk::ShaderStageFlagBits::eVertex,
								std::uint32_t offset = 0)
	{
		m_push_constants.push_back(
			PushBinding{.stage	= stage,
						.offset = offset,
						.size	= static_cast<std::uint32_t>(sizeof(T)),
						.read	= [handle](graph::ExecContext& ctx) -> const void*
						{
							const auto* slot = dynamic_cast<graph::ValueData<T>*>(ctx.data(handle));
							return slot != nullptr ? static_cast<const void*>(&slot->value()) : nullptr;
						}});
		m_inputs.push_back(handle); // a push-constant edge is a dirtiness input
		return *this;
	}

	/// Bind a mesh edge: the node draws the buffer-backed geometry on `handle` (a
	/// `ValueData<gpu::MeshRef>`, produced by a MeshNode) instead of fabricating vertices
	/// from SV_VertexID. Indexed if the ref carries an index buffer, else a non-indexed draw
	/// of the ref's vertex count — either way overriding the constructor's `vertex_count`.
	/// The vertex layout must match the vertex shader's reflected input. A reactive input
	/// (re-uploading the mesh re-renders). Returns *this for chaining; call before adding the
	/// node to the graph (it extends the input set).
	GraphicsNode& set_mesh(graph::DataHandle handle)
	{
		m_mesh = handle;
		m_inputs.push_back(handle);
		return *this;
	}

	/// Bind a uniform edge: `handle` (a `ValueData<gpu::UniformRef>`, produced by a
	/// UniformNode) is written into this node's descriptor set at the binding whose reflected
	/// name matches the ref's `name`. A reactive input — changing the uniform value re-renders
	/// the node. Returns *this for chaining; call before adding the node to the graph (it
	/// extends the input set).
	GraphicsNode& add_uniform(graph::DataHandle handle)
	{
		m_uniforms.push_back(handle);
		m_inputs.push_back(handle);
		return *this;
	}

	/// Set the color the target is cleared to each frame (RGBA, default opaque black).
	/// Returns *this for chaining; call before adding the node to the graph.
	GraphicsNode& clear_color(std::array<float, 4> rgba) noexcept
	{
		m_clear_color = rgba;
		return *this;
	}

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return m_inputs; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	/// The persistent color target, or nullptr before the first render (test lens).
	[[nodiscard]] const Image* scene() const noexcept { return m_color.has_value() ? &m_color.value() : nullptr; }

	 protected:
	[[nodiscard]] std::expected<bool, graph::ExecError> record(gpu::GpuExecContext& ctx) override;

	 private:
	struct PushBinding
	{
		vk::ShaderStageFlags							stage;
		std::uint32_t									offset;
		std::uint32_t									size;
		std::function<const void*(graph::ExecContext&)> read;
	};

	std::string						 m_vertex_shader;
	std::string						 m_fragment_shader;
	vk::Format						 m_color_format;
	vk::Format						 m_depth_format; // eUndefined => no depth attachment
	std::uint32_t					 m_vertex_count;
	graph::DataHandle				 m_output;
	std::vector<graph::DataHandle>	 m_inputs; // [screen_size, push-constant + mesh + uniform edges...]
	std::vector<PushBinding>		 m_push_constants;
	std::optional<graph::DataHandle> m_mesh;	 // buffer-backed geometry, else SV_VertexID draw
	std::vector<graph::DataHandle>	 m_uniforms; // descriptor-bound uniform edges
	std::array<float, 4>			 m_clear_color{0.0F, 0.0F, 0.0F, 1.0F};
	std::optional<GraphicsPipeline>	 m_pipeline; // built lazily on first record

	// Uniform descriptor state, populated on first record (alongside the pipeline). The set is
	// written once and reused: the uniform buffers are persistent, so only their contents
	// change across frames — m_bound_buffers detects the rare handle change that needs a rewrite.
	std::map<std::string, std::uint32_t> m_binding_by_name; // reflected descriptor name -> binding
	std::optional<DescriptorAllocator>	 m_descriptors;
	vk::DescriptorSet					 m_descriptor_set;
	std::vector<vk::Buffer>				 m_bound_buffers;

	std::optional<Image> m_color;
	std::optional<Image> m_depth;
	vk::Extent2D		 m_extent{};
};
} // namespace veng::nodes

#endif // VENG_GRAPHICSNODE_HPP
