/**
 * @file
 * @author chris
 * @brief Screen-coverage LOD metric: picks a discrete LOD level from a bounding sphere's projected
 *        screen size, with hysteresis to avoid flicker at the thresholds.
 *
 * Produces the `uint` level edge a @ref veng::nodes::MeshSelectorNode consumes. The metric is the
 * sphere's projected diameter as a fraction of viewport height — `coverage = radius · proj[1][1] /
 * view_depth` — which folds distance, object size and field-of-view into one number (the standard
 * Unity/UE "screen size" criterion). It subsumes plain distance and is resolution-independent.
 *
 * Thresholds are descending fractions of viewport height: with N LOD meshes pass N−1 thresholds, so
 * coverage above `thresholds[0]` is the finest level (0) and coverage below the last threshold is the
 * coarsest. A hysteresis margin widens the band around the *current* level's boundaries so a camera
 * dithering on a threshold doesn't ping-pong; the node reads its own previous output to apply it.
 *
 * Selection is pure CPU and recomputes only when the camera/viewport/bounds change — GPU-side
 * per-cluster selection is deferred to a future mesh-shader path (see notes).
 *
 * @ingroup graph_nodes
 * @see MeshSelectorNode
 */

#ifndef VENG_COVERAGELODNODE_HPP
#define VENG_COVERAGELODNODE_HPP

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>
#include <veng/rendergraph/nodes/Node.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>

namespace veng::nodes
{
/**
 * @brief Maps a world-space bounding sphere's screen coverage to a LOD level (with hysteresis).
 * @ingroup graph_nodes
 */
class CoverageLodNode final : public graph::Node
{
	 public:
	/**
	 * @brief Wire the metric from the camera matrices, viewport and object bounds.
	 *
	 * @param view       `ValueData<glm::mat4>` world→view matrix edge (gives the sphere's view depth).
	 * @param proj       `ValueData<glm::mat4>` projection matrix edge (its [1][1] is the vertical focal length).
	 * @param sphere     `ValueData<glm::vec4>` world-space bounding sphere: xyz = centre, w = radius.
	 * @param output     `ValueData<std::uint32_t>` edge this node publishes the chosen level on.
	 * @param thresholds Descending coverage thresholds (fractions of viewport height); size = levels − 1.
	 * @param hysteresis Fractional dead-band around the current level's boundaries (e.g. 0.1 = ±10%).
	 */
	CoverageLodNode(graph::DataHandle view, graph::DataHandle proj, graph::DataHandle sphere, graph::DataHandle output,
					std::vector<float> thresholds, float hysteresis = 0.1F);

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return m_inputs; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	[[nodiscard]] std::expected<bool, graph::ExecError> execute(graph::ExecContext& ctx) override;

	 private:
	std::array<graph::DataHandle, 3> m_inputs; ///< view, proj, sphere (in that order).
	graph::DataHandle				 m_output;
	std::vector<float>				 m_thresholds; ///< Descending; one fewer than the LOD level count.
	float							 m_hysteresis;
};
} // namespace veng::nodes

#endif // VENG_COVERAGELODNODE_HPP
