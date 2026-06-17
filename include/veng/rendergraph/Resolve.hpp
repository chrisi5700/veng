/**
 * @file
 * @author chris
 * @brief Typed edge resolution for the execute pass — the one `dynamic_cast` a node would otherwise
 *        spell out by hand for every input and output edge it reads.
 *
 * A node's `execute`/`record` resolves each @ref veng::graph::DataHandle it touches to the concrete
 * @ref veng::graph::ValueData "ValueData<T>" slot behind it, through the @ref veng::graph::ExecContext.
 * That resolution is always the same checked downcast `dynamic_cast<ValueData<T>*>(ctx.data(handle))`;
 * @ref veng::graph::resolve names it once so node code reads in graph vocabulary and never repeats the
 * cast — nor mistypes the `ValueData<>` it casts to, which the compiler cannot catch in the raw form.
 *
 * @ingroup rendergraph
 */

#ifndef VENG_RESOLVE_HPP
#define VENG_RESOLVE_HPP

#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/nodes/Node.hpp>

namespace veng::graph
{
/**
 * @brief Resolve @p handle to its concrete `ValueData<T>` slot for the execute pass.
 *
 * The single checked downcast a node performs for each edge it reads or writes: it looks the handle
 * up in @p ctx and casts the resulting @ref veng::graph::Data to the expected `ValueData<T>`. Returns
 * `nullptr` if the handle is invalid or names a slot of a different type — the caller turns that into
 * its own error (typically @ref veng::graph::ExecError::MISSING_INPUT), exactly as the hand-written
 * cast did. Bind the result to `const auto*` for a read or `auto*` when you will `produce` into it.
 *
 * @tparam T     The value type the slot is expected to carry.
 * @param ctx    The execution context resolving handles for this frame.
 * @param handle The edge to resolve.
 * @return The typed slot, or `nullptr` if @p handle is invalid or of a different type.
 */
template <class T>
[[nodiscard]] ValueData<T>* resolve(const ExecContext& ctx, DataHandle handle) noexcept
{
	return dynamic_cast<ValueData<T>*>(ctx.data(handle));
}
} // namespace veng::graph

#endif // VENG_RESOLVE_HPP
