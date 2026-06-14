/**
 * @file
 * @author chris
 * @brief L4 reactive uniform-buffer upload node: uploads a typed value each time it changes
 *        and publishes a `gpu::UniformRef` for @ref veng::nodes::GraphicsNode to bind.
 *
 * You give it a value edge (a `ValueData<T>` source or transform) and the reflected binding
 * name; each time that value changes it uploads the new bytes into its pool-owned persistent
 * uniform buffer and publishes a `gpu::UniformRef`. A
 * @ref veng::nodes::GraphicsNode::add_uniform "add_uniform(handle)" call consumes the ref, matches the name
 * to the shader's reflected descriptor, and binds it.
 *
 * Unlike @ref veng::nodes::MeshNode (geometry baked in, uploaded once) this is genuinely reactive: the
 * value is an input edge, so changing it re-runs the upload and re-renders every dependent
 * @ref veng::nodes::GraphicsNode. The buffer is host-visible and persistently mapped and reused across
 * uploads, so its handle is stable — the dependent node writes its descriptor set once and
 * just re-binds it as the contents change.
 *
 * @ingroup graph_nodes
 * @see GraphicsNode::add_uniform
 * @see MeshNode
 */

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
#include <veng/gpu/VersionedOutput.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <veng/resources/Buffer.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::nodes
{
/**
 * @brief L4 reactive uniform-buffer upload node: publishes a `gpu::UniformRef` that
 *        @ref veng::nodes::GraphicsNode binds to the named shader `ConstantBuffer`.
 * @ingroup graph_nodes
 * @see GraphicsNode::add_uniform
 */
class UniformNode final : public gpu::GpuNode
{
	 public:
	/**
	 * @brief Construct a uniform node from a typed value edge and a reflected binding name.
	 *
	 * Uploads the value on `value` (a `TypedHandle<T>`) into a uniform buffer and publishes a
	 * `gpu::UniformRef` on `output` (a `ValueData<gpu::UniformRef>`). `binding_name` must match
	 * the reflected descriptor name in the consuming shader (the `ConstantBuffer` variable
	 * name). `T`'s memory layout must match the shader's uniform block (std140); for a single
	 * `float4` / `glm::vec4` that is a plain 16-byte copy. `T` is deduced from `value`.
	 *
	 * @tparam T           The uniform value type (deduced from `value`).
	 * @param value        Reactive `TypedHandle<T>` edge whose value is uploaded each change.
	 * @param binding_name Reflected `ConstantBuffer` name in the consuming shader.
	 * @param output       `ValueData<gpu::UniformRef>` edge this node publishes its ref on.
	 */
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
	graph::DataHandle								m_value; ///< The reactive value edge.
	graph::DataHandle								m_output;
	std::string										m_name;
	std::uint64_t									m_size;
	std::function<const void*(graph::ExecContext&)> m_read;
	bool				 m_declared	 = false; ///< `true` once `m_buffer_id` has been declared in the pool.
	BufferId			 m_buffer_id = 0;	  ///< Pool-owned, N-buffered uniform buffer.
	gpu::VersionedOutput m_versioned;		  ///< Owns the per-upload version bump for the published `UniformRef`.
};
} // namespace veng::nodes

#endif // VENG_UNIFORMNODE_HPP
