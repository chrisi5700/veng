//
// Orbit camera that owns its `view_proj` graph source. The example wires the source into a
// `GraphicsNode::push_constant<glm::mat4>` and the camera's `tick()` publishes a fresh value
// whenever the user drags / scrolls — a stationary camera contributes no reactive dirtiness,
// so an OnDemand loop idles. The graph mutation goes through the source handle; the camera
// does not touch any descriptor / pipeline state itself.
//

#ifndef VENG_EXAMPLE_ORBIT_CAMERA_HPP
#define VENG_EXAMPLE_ORBIT_CAMERA_HPP

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <vulkan/vulkan.hpp>

#include "Window.hpp"

namespace example
{
class OrbitCamera
{
	 public:
	/// Register the `view_proj` source on `graph` and publish the initial matrix sized for
	/// `initial_extent`. The example treats `view_proj()` as a regular reactive source from
	/// here on.
	OrbitCamera(veng::graph::Graph& graph, vk::Extent2D initial_extent, glm::vec3 target = {0.0F, 0.0F, 0.0F},
				float distance = 6.0F, float yaw = 0.6F, float pitch = 0.45F) noexcept
		: m_graph(&graph)
		, m_target(target)
		, m_distance(distance)
		, m_yaw(yaw)
		, m_pitch(pitch)
		, m_view_proj_src(graph.add_source<glm::mat4>(glm::mat4(1.0F)))
		, m_eye_pos_src(graph.add_source<glm::vec4>(glm::vec4(0.0F)))
	{
		publish(initial_extent);
	}

	/// The view-projection source handle (typical push-constant target).
	[[nodiscard]] veng::graph::TypedHandle<glm::mat4> view_proj() const noexcept { return m_view_proj_src; }

	/// The world-space eye position as a `vec4` source (w = 1). Use it when the fragment
	/// shader needs the viewer position — Phong specular, world-space fog, parallax, ...
	[[nodiscard]] veng::graph::TypedHandle<glm::vec4> eye_pos() const noexcept { return m_eye_pos_src; }

	/// Poll input + publish a fresh view_proj if the camera moved. Returns true if it did.
	/// AppLoop calls this each iteration under the graph mutex.
	bool tick(Window& window, vk::Extent2D extent) noexcept
	{
		if (!consume_input(window))
		{
			return false;
		}
		publish(extent);
		return true;
	}

	/// Republish at a new framebuffer extent (called on resize), without consuming input.
	void publish(vk::Extent2D extent) noexcept
	{
		m_graph->set(m_view_proj_src, compute_view_proj(extent));
		m_graph->set(m_eye_pos_src, glm::vec4(compute_eye(), 1.0F));
	}

	 private:
	[[nodiscard]] bool consume_input(Window& window) noexcept
	{
		const Window::CursorPos cursor = window.cursor_pos();
		bool					moved  = false;

		if (window.mouse_down(0 /* GLFW_MOUSE_BUTTON_LEFT */) && m_have_last_cursor)
		{
			const double dx = cursor.x - m_last_cursor.x;
			const double dy = cursor.y - m_last_cursor.y;
			if (dx != 0.0 || dy != 0.0)
			{
				constexpr float DRAG_SCALE = 0.005F;
				m_yaw -= static_cast<float>(dx) * DRAG_SCALE;
				m_pitch = std::clamp(m_pitch + static_cast<float>(dy) * DRAG_SCALE, -1.55F, 1.55F);
				moved	= true;
			}
		}
		m_last_cursor	   = cursor;
		m_have_last_cursor = true;

		const double scroll = window.consume_scroll_y();
		if (scroll != 0.0)
		{
			constexpr float ZOOM_FACTOR = 0.9F;
			m_distance = std::clamp(m_distance * std::pow(ZOOM_FACTOR, static_cast<float>(scroll)), 0.5F, 200.0F);
			moved	   = true;
		}
		return moved;
	}

	[[nodiscard]] glm::vec3 compute_eye() const noexcept
	{
		return m_target + glm::vec3{std::cos(m_pitch) * std::sin(m_yaw), std::sin(m_pitch),
									std::cos(m_pitch) * std::cos(m_yaw)} *
							  m_distance;
	}

	[[nodiscard]] glm::mat4 compute_view_proj(vk::Extent2D extent) const noexcept
	{
		const float aspect =
			extent.height > 0 ? static_cast<float>(extent.width) / static_cast<float>(extent.height) : 1.0F;
		glm::mat4 proj = glm::perspective(glm::radians(55.0F), aspect, 0.1F, 1000.0F);
		proj[1][1] *= -1.0F; // Vulkan clip-space y-down
		return proj * glm::lookAt(compute_eye(), m_target, glm::vec3{0.0F, 1.0F, 0.0F});
	}

	veng::graph::Graph*					m_graph;
	glm::vec3							m_target;
	float								m_distance;
	float								m_yaw;
	float								m_pitch;
	veng::graph::TypedHandle<glm::mat4> m_view_proj_src;
	veng::graph::TypedHandle<glm::vec4> m_eye_pos_src;
	Window::CursorPos					m_last_cursor{};
	bool								m_have_last_cursor = false;
};
} // namespace example

#endif // VENG_EXAMPLE_ORBIT_CAMERA_HPP
