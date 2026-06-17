/**
 * @file
 * @author chris
 * @brief @ref veng::nodes::CoverageLodNode implementation.
 * @ingroup graph_nodes
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <glm/glm.hpp>
#include <utility>
#include <veng/nodes/CoverageLodNode.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/Resolve.hpp>

namespace veng::nodes
{
CoverageLodNode::CoverageLodNode(graph::DataHandle view, graph::DataHandle proj, graph::DataHandle sphere,
								 graph::DataHandle output, std::vector<float> thresholds, float hysteresis)
	: m_inputs{view, proj, sphere}
	, m_output(output)
	, m_thresholds(std::move(thresholds))
	, m_hysteresis(hysteresis)
{
}

std::expected<bool, graph::ExecError> CoverageLodNode::execute(graph::ExecContext& ctx)
{
	auto* view	 = graph::resolve<glm::mat4>(ctx, m_inputs[0]);
	auto* proj	 = graph::resolve<glm::mat4>(ctx, m_inputs[1]);
	auto* sphere = graph::resolve<glm::vec4>(ctx, m_inputs[2]);
	auto* out	 = graph::resolve<std::uint32_t>(ctx, m_output);
	if (view == nullptr || proj == nullptr || sphere == nullptr || out == nullptr)
	{
		return std::unexpected(graph::ExecError::MISSING_INPUT);
	}

	// Project the sphere: coverage = its diameter as a fraction of viewport height. The view-space
	// depth is -z (camera looks down -Z); proj[1][1] is the vertical focal length (1/tan(fovy/2)).
	const glm::vec4 centre	   = sphere->value();
	const float		radius	   = centre.w;
	const glm::vec4 view_pos   = view->value() * glm::vec4(glm::vec3(centre), 1.0F);
	const float		view_depth = -view_pos.z;
	// Behind / on the camera plane ⇒ treat as maximally close so we pick the finest level. abs() on the
	// focal term keeps it correct under a flipped-Y (Vulkan) clip convention where proj[1][1] < 0.
	const float coverage = view_depth > 1e-4F ? (radius * std::abs(proj->value()[1][1])) / view_depth : 1e9F;

	// Level without hysteresis: how many descending thresholds the coverage falls below.
	std::uint32_t raw = 0;
	for (std::size_t i = 0; i < m_thresholds.size(); ++i)
	{
		if (coverage < m_thresholds[i])
		{
			raw = static_cast<std::uint32_t>(i + 1);
		}
	}

	// Hysteresis: only leave the current level once coverage clears the relevant boundary by the
	// margin — the finer boundary (thresholds[current-1]) to go finer, the coarser one to go coarser.
	std::uint32_t current = std::min(out->value(), static_cast<std::uint32_t>(m_thresholds.size()));
	std::uint32_t result  = current;
	if (raw < current && current > 0 && coverage > m_thresholds[current - 1] * (1.0F + m_hysteresis))
	{
		result = raw; // grown enough to commit to the finer level
	}
	else if (raw > current && current < m_thresholds.size() && coverage < m_thresholds[current] * (1.0F - m_hysteresis))
	{
		result = raw; // shrunk enough to commit to the coarser level
	}
	return out->produce(result);
}
} // namespace veng::nodes
