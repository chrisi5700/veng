//
// Created by chris on 1/22/26.
//

#ifndef VENG_CONTEXT_HPP
#define VENG_CONTEXT_HPP

#include <functional>
#include <span>
#include <string_view>
#include <vulkan-memory-allocator-hpp/vk_mem_alloc.hpp>
#include <vulkan/vulkan.hpp>

#include "ContextErrors.hpp"
#include "veng/common.hpp"
#include "veng/logging/Logger.hpp"

namespace veng
{
struct QueueFamilyIndices
{
	uint32_t graphics;
	uint32_t compute;

	[[nodiscard]] bool has_dedicated_compute() const { return compute != graphics; }
};

class Context
{
	 public:
	/// Create the Vulkan context. With no extra arguments this is a headless context
	/// (all existing tests). For windowed rendering, pass the window's required instance
	/// extensions and a `surface_factory` that creates a VkSurfaceKHR from the instance;
	/// the Context then owns that surface and verifies the graphics queue can present.
	static std::expected<Context, ContextCreationError> create(
		std::string_view title, std::span<const char* const> instance_extensions = {},
		const std::function<VkSurfaceKHR(VkInstance)>& surface_factory = {});
	~Context();

	Context(const Context&)			   = delete;
	Context& operator=(const Context&) = delete;
	Context(Context&& other) noexcept;
	Context& operator=(Context&& other) noexcept;

	[[nodiscard]] vk::Instance		 instance() const { return m_instance; }
	[[nodiscard]] vk::PhysicalDevice physical_device() const { return m_physical_device; }
	[[nodiscard]] QueueFamilyIndices queue_indices() const { return m_queue_indices; }
	[[nodiscard]] vk::Device		 device() const { return m_device; }
	[[nodiscard]] vk::Queue			 graphics_queue() const { return m_graphics_queue; }
	[[nodiscard]] vk::Queue			 compute_queue() const { return m_compute_queue; }
	[[nodiscard]] vma::Allocator	 allocator() const { return m_allocator; }
	/// The window surface, or a null handle for a headless context.
	[[nodiscard]] vk::SurfaceKHR surface() const { return m_surface; }

	/// One-shot recording on the graphics queue: allocates a transient command buffer, calls
	/// `record(cmd)`, ends + submits with a private fence, waits for completion, frees. The
	/// engine's "do this on the GPU now" primitive — replaces ad-hoc pool/fence/submit/wait
	/// sequences in user code (e.g. resource uploads, constant-image initialization).
	[[nodiscard]] vk::Result immediate_submit(const std::function<void(vk::CommandBuffer)>& record) const;

	 private:
	Context(vk::Instance m_instance, vk::DebugUtilsMessengerEXT m_debug_messenger, vk::PhysicalDevice m_physical_device,
			QueueFamilyIndices m_queue_indices, vk::Device m_device, vk::Queue m_graphics_queue,
			vk::Queue m_compute_queue, vma::Allocator m_allocator, vk::SurfaceKHR m_surface)
		: m_instance(m_instance)
		, m_debug_messenger(m_debug_messenger)
		, m_physical_device(m_physical_device)
		, m_queue_indices(m_queue_indices)
		, m_device(m_device)
		, m_graphics_queue(m_graphics_queue)
		, m_compute_queue(m_compute_queue)
		, m_allocator(m_allocator)
		, m_surface(m_surface)
	{
		Logger::instance().info("VulkanContext VK_HEADER_VERSION: {}", VK_HEADER_VERSION);
		Logger::instance().info("VulkanContext initialized");
	}
	vk::Instance			   m_instance;
	vk::DebugUtilsMessengerEXT m_debug_messenger;
	vk::PhysicalDevice		   m_physical_device;
	QueueFamilyIndices		   m_queue_indices;
	vk::Device				   m_device;
	vk::Queue				   m_graphics_queue;
	vk::Queue				   m_compute_queue;
	vma::Allocator			   m_allocator;
	vk::SurfaceKHR			   m_surface; // null for headless; owned, destroyed before the instance
};
} // namespace veng
#endif // VENG_CONTEXT_HPP
