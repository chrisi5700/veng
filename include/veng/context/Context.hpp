//
// Created by chris on 1/22/26.
//

#ifndef VENG_CONTEXT_HPP
#define VENG_CONTEXT_HPP

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
	static std::expected<Context, ContextCreationError> create(std::string_view title);
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

	 private:
	Context(vk::Instance m_instance, vk::DebugUtilsMessengerEXT m_debug_messenger, vk::PhysicalDevice m_physical_device,
			QueueFamilyIndices m_queue_indices, vk::Device m_device, vk::Queue m_graphics_queue,
			vk::Queue m_compute_queue, vma::Allocator m_allocator)
		: m_instance(m_instance)
		, m_debug_messenger(m_debug_messenger)
		, m_physical_device(m_physical_device)
		, m_queue_indices(m_queue_indices)
		, m_device(m_device)
		, m_graphics_queue(m_graphics_queue)
		, m_compute_queue(m_compute_queue)
		, m_allocator(m_allocator)
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
};
} // namespace veng
#endif // VENG_CONTEXT_HPP
