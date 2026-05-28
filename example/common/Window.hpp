//
// A minimal GLFW window for the renderer demo: owns the GLFWwindow, exposes the
// Vulkan instance extensions it needs, and creates a VkSurfaceKHR from an instance.
// Window/event handling lives on the main thread (a GLFW requirement); the surface is
// owned by the Context, not by this window.
//

#ifndef VENG_DEMO_WINDOW_HPP
#define VENG_DEMO_WINDOW_HPP

#include <span>
#include <vector>
#include <vulkan/vulkan.hpp>

struct GLFWwindow;

namespace example
{
class Window
{
	 public:
	Window(const char* title, int width, int height);
	~Window();
	Window(const Window&)			 = delete;
	Window& operator=(const Window&) = delete;
	Window(Window&&)				 = delete;
	Window& operator=(Window&&)		 = delete;

	[[nodiscard]] bool		   should_close() const;
	void					   poll() const;
	[[nodiscard]] vk::Extent2D framebuffer_extent() const;

	/// True while `key` (a GLFW_KEY_* code) is held. Edge detection is the caller's job.
	[[nodiscard]] bool key_down(int key) const;

	/// Current cursor position in screen-space pixels (origin top-left). For orbit/fly camera
	/// drags the caller stores last frame's value and computes the delta itself.
	struct CursorPos
	{
		double x = 0.0;
		double y = 0.0;
	};
	[[nodiscard]] CursorPos cursor_pos() const;

	/// True while `button` (a GLFW_MOUSE_BUTTON_* code) is held.
	[[nodiscard]] bool mouse_down(int button) const;

	/// Wheel scroll accumulated since the last call, then cleared (consumed-once). A drag
	/// poll loop reads this each frame; positive y is "scroll up / zoom in" on every platform
	/// GLFW supports. Owned via a callback installed at construction.
	[[nodiscard]] double consume_scroll_y();

	/// Instance extensions needed for a window surface (VK_KHR_surface + the platform
	/// surface ext). Falls back to a known-good list if GLFW's own query comes up empty.
	[[nodiscard]] std::span<const char* const> required_extensions() const { return m_required; }

	/// Create a surface from the instance, or VK_NULL_HANDLE on failure.
	[[nodiscard]] VkSurfaceKHR create_surface(VkInstance instance) const;

	 private:
	GLFWwindow*				 m_window = nullptr;
	std::vector<const char*> m_required;
	// Accumulated by a GLFW scroll callback (called inside `poll()` -> glfwPollEvents on the
	// main thread), drained by `consume_scroll_y`. Mutable so the const consumer can clear it.
	mutable double m_scroll_y = 0.0;
};
} // namespace example

#endif // VENG_DEMO_WINDOW_HPP
