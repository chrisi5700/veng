//
// See Window.hpp.
//

#include "Window.hpp"

#include <cstdint>
#include <print>
#include <stdexcept>
#include <string>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace demo
{
namespace
{
std::string g_last_glfw_error;

void glfw_error_callback(int code, const char* description)
{
	g_last_glfw_error =
		std::string("GLFW error ") + std::to_string(code) + ": " + (description != nullptr ? description : "(null)");
	std::println(stderr, "{}", g_last_glfw_error);
}

const char* platform_name(int platform)
{
	switch (platform)
	{
		case GLFW_PLATFORM_WAYLAND: return "Wayland";
		case GLFW_PLATFORM_X11: return "X11";
		case GLFW_PLATFORM_NULL: return "Null";
		default: return "unknown";
	}
}
} // namespace

Window::Window(const char* title, int width, int height)
{
	glfwSetErrorCallback(glfw_error_callback);
	if (glfwInit() != GLFW_TRUE)
	{
		throw std::runtime_error("glfwInit failed: " + g_last_glfw_error);
	}
	std::println(stderr, "[window] GLFW platform: {}, Vulkan supported: {}", platform_name(glfwGetPlatform()),
				 glfwVulkanSupported() == GLFW_TRUE);
	if (glfwVulkanSupported() != GLFW_TRUE)
	{
		glfwTerminate();
		throw std::runtime_error("GLFW reports no Vulkan support: " + g_last_glfw_error);
	}
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // no OpenGL context — Vulkan only
	m_window = glfwCreateWindow(width, height, title, nullptr, nullptr);
	if (m_window == nullptr)
	{
		glfwTerminate();
		throw std::runtime_error("glfwCreateWindow failed");
	}

	std::uint32_t	   count	  = 0;
	const char* const* extensions = glfwGetRequiredInstanceExtensions(&count);
	if (extensions != nullptr && count > 0)
	{
		m_required.assign(extensions, extensions + count);
	}
	else
	{
		// GLFW's query came up empty even though the loader exposes the surface
		// extensions (seen on XWayland). Enable the ones we know exist; glfwCreateWindow-
		// Surface then creates the surface (falling back to xlib internally if needed).
		const int platform = glfwGetPlatform();
		m_required.push_back("VK_KHR_surface");
		if (platform == GLFW_PLATFORM_WAYLAND)
		{
			m_required.push_back("VK_KHR_wayland_surface");
		}
		else
		{
			m_required.push_back("VK_KHR_xcb_surface");
			m_required.push_back("VK_KHR_xlib_surface");
		}
		std::println(stderr, "[window] glfwGetRequiredInstanceExtensions empty; using {} fallback list",
					 platform_name(platform));
	}
	std::println(stderr, "[window] instance extensions ({}):", m_required.size());
	for (const char* extension : m_required)
	{
		std::println(stderr, "  {}", extension);
	}
}

Window::~Window()
{
	if (m_window != nullptr)
	{
		glfwDestroyWindow(m_window);
	}
	glfwTerminate();
}

bool Window::should_close() const
{
	return glfwWindowShouldClose(m_window) == GLFW_TRUE;
}

void Window::poll() const
{
	glfwPollEvents();
}

vk::Extent2D Window::framebuffer_extent() const
{
	int width  = 0;
	int height = 0;
	glfwGetFramebufferSize(m_window, &width, &height);
	return vk::Extent2D{static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)};
}

bool Window::key_down(int key) const
{
	return glfwGetKey(m_window, key) == GLFW_PRESS;
}

VkSurfaceKHR Window::create_surface(VkInstance instance) const
{
	VkSurfaceKHR surface = VK_NULL_HANDLE;
	if (glfwCreateWindowSurface(instance, m_window, nullptr, &surface) != VK_SUCCESS)
	{
		return VK_NULL_HANDLE;
	}
	return surface;
}
} // namespace demo
