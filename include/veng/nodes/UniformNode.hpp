//
// Created by chris on 5/26/26.
//
// L4 upload node (design.md §L4) — the uniform counterpart to MeshNode. You give it a
// *value edge* (a `ValueData<T>` source/transform) and the reflected binding name; each
// time that value changes it uploads the new bytes into its persistent uniform buffer and
// publishes a `gpu::UniformRef`. A `GraphicsNode::add_uniform(handle)` consumes the ref,
// matches the name to the shader's reflected descriptor, and binds it. This is the user's
// `UniformNode{light_pos, "light_pos"}`.
//
// Unlike MeshNode (geometry baked in, uploaded once) this is genuinely reactive: the value
// is an *input* edge, so changing it re-runs the upload and re-renders every dependent
// GraphicsNode. The buffer is host-visible + persistently mapped and reused across
// uploads, so its handle is stable — the dependent node writes its descriptor set once and
// just re-binds it as the contents change.
//

#ifndef VENG_UNIFORMNODE_HPP
#define VENG_UNIFORMNODE_HPP

#include <cstddef>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <veng/resources/Buffer.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::nodes
{
class UniformNode final : public gpu::GpuNode
{
	 public:
	/// Upload the value on `value` (a typed edge, `TypedHandle<T>`) into a uniform buffer and
	/// publish a `gpu::UniformRef` on `output` (a `ValueData<gpu::UniformRef>`). `binding_name`
	/// must match the reflected descriptor name in the consuming shader (the `ConstantBuffer`
	/// variable name). `T`'s memory layout must match the shader's uniform block (std140); for
	/// a single `float4`/`glm::vec4` that is a plain 16-byte copy. `T` is deduced from `value`.
	template <class T>
	UniformNode(graph::TypedHandle<T> value, std::string binding_name, graph::DataHandle output) noexcept
		: m_value(value.handle)
		, m_output(output)
		, m_name(std::move(binding_name))
		, m_size(sizeof(T))
		, m_read(
			  [handle = value.handle](graph::ExecContext& ctx) -> const void*
			  {
				  const auto* slot = dynamic_cast<graph::ValueData<T>*>(ctx.data(handle));
				  return slot != nullptr ? static_cast<const void*>(&slot->value()) : nullptr;
			  })
	{
	}

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return {&m_value, 1}; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	 protected:
	[[nodiscard]] std::expected<bool, graph::ExecError> record(gpu::GpuExecContext& ctx) override;

	 private:
	graph::DataHandle								m_value; // the reactive value edge
	graph::DataHandle								m_output;
	std::string										m_name;
	vk::DeviceSize									m_size;
	std::function<const void*(graph::ExecContext&)> m_read;
	bool											m_declared	= false; // m_buffer_id declared in the pool?
	BufferId										m_buffer_id = 0;	 // pool-owned, N-buffered uniform buffer
};
} // namespace veng::nodes

#endif // VENG_UNIFORMNODE_HPP
