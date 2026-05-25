//
// Created by chris on 1/22/26.
//
// VulkanContext.cpp

#include <utility>
#include <veng/context/Context.hpp>
#include <veng/context/ContextErrors.hpp>
#include <veng/logging/Logger.hpp>
#include <VkBootstrap.h>
#include <vulkan/vulkan.hpp>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace veng
{

#ifdef NDEBUG
constexpr bool ENABLE_VALIDATION = false;
#else
constexpr bool ENABLE_VALIDATION = true;
#endif

VKAPI_ATTR vk::Bool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT			 severity,
												[[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT type,
												const VkDebugUtilsMessengerCallbackDataEXT*		 callback_data,
												[[maybe_unused]] void*							 user_data)
{
	auto& logger  = Logger::instance();
	auto  pattern = fmt::format("[IteratedFunction]{:<30}[%^%5l%$] %v", "[VulkanDebug]");
	logger.set_pattern(pattern);
	switch (severity)
	{
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT: logger.trace("{}", callback_data->pMessage); break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT: logger.debug("{}", callback_data->pMessage); break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT: logger.warn("{}", callback_data->pMessage); break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT: logger.error("{}", callback_data->pMessage); break;
		default: logger.info("{}", callback_data->pMessage); break;
	}

	return vk::False;
}

namespace
{
// vk-bootstrap reports failures as a std::error_code plus an optional VkResult.
// Flatten that down to the vk::Result our error types carry.
template <class T>
vk::Result vk_result_of(const vkb::Result<T>& result)
{
	auto raw = result.vk_result();
	return raw != VK_SUCCESS ? static_cast<vk::Result>(raw) : vk::Result::eErrorInitializationFailed;
}

void destroy_debug_messenger(vk::Instance instance, vk::DebugUtilsMessengerEXT messenger)
{
	if (!messenger)
		return;

	instance.destroyDebugUtilsMessengerEXT(messenger, nullptr);
	Logger::instance().trace("Destroyed debug messenger");
}
} // namespace

std::expected<Context, ContextCreationError> Context::create(std::string_view title)
{
	// vk-bootstrap owns the handles until we hand them to Context; these guards
	// tear them back down if we bail out part-way through.
	struct InstanceGuard
	{
		vkb::Instance instance;
		bool		  armed = true;
		~InstanceGuard()
		{
			if (armed)
				vkb::destroy_instance(instance);
		}
	};
	struct DeviceGuard
	{
		vkb::Device device;
		bool		armed = true;
		~DeviceGuard()
		{
			if (armed)
				vkb::destroy_device(device);
		}
	};

	// We drive presentation through VK_KHR_surface/swapchain but never let
	// vk-bootstrap auto-require the platform windowing extensions (xcb/xlib/
	// wayland): no window is created here, and headless loaders may not expose
	// them. The surface extension is enabled explicitly so the swapchain device
	// extension stays valid once a window is attached later.
	// vk-bootstrap takes a const char*; string_view::data() is not guaranteed
	// NUL-terminated, so copy into a local std::string first (M1).
	const std::string	 app_name{title};
	vkb::InstanceBuilder builder;
	auto				 instance_ret = builder.set_app_name(app_name.c_str())
											.set_engine_name("No Engine")
											.require_api_version(1, 3, 0)
											.set_headless()
											.enable_extension(VK_KHR_SURFACE_EXTENSION_NAME)
											.request_validation_layers(ENABLE_VALIDATION)
											.set_debug_callback(debug_callback)
											.build();
	if (!instance_ret)
	{
		Logger::instance().error("Failed to create Vulkan instance: {}", instance_ret.error().message());
		return std::unexpected(InstanceCreationError{vk_result_of(instance_ret)});
	}
	InstanceGuard instance_guard{instance_ret.value()};
	const auto&	  vkb_instance = instance_guard.instance;

	// Drive vulkan.hpp's dynamic dispatcher from the loader vk-bootstrap resolved.
	VULKAN_HPP_DEFAULT_DISPATCHER.init(vkb_instance.fp_vkGetInstanceProcAddr);
	VULKAN_HPP_DEFAULT_DISPATCHER.init(vk::Instance(vkb_instance.instance));
	Logger::instance().debug("Created Vulkan instance");

	// We request the swapchain extension and defer the surface so a window can be
	// attached later, without requiring one to exist during device selection.
	vk::PhysicalDeviceFeatures features{};
	features.tessellationShader = vk::True;
	features.geometryShader		= vk::True;

	vk::PhysicalDeviceVulkan11Features features11{};
	features11.shaderDrawParameters = vk::True;

	// Vulkan 1.3 core features the design mandates (design.md §L0): dynamic rendering
	// (pipelines built with formats instead of a VkRenderPass — required for
	// GraphicsPipelineBuilder), synchronization2 (barrier helpers), and timeline
	// semaphores (the GPU-side revision clock, §2.7/§5).
	vk::PhysicalDeviceVulkan13Features features13{};
	features13.dynamicRendering = vk::True;
	features13.synchronization2 = vk::True;

	vk::PhysicalDeviceVulkan12Features features12{};
	features12.timelineSemaphore = vk::True;

	vkb::PhysicalDeviceSelector selector{vkb_instance};
	auto						physical_ret = selector.set_minimum_version(1, 3)
												   .defer_surface_initialization()
												   .add_required_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME)
												   .set_required_features(features)
												   .set_required_features_11(features11)
												   .set_required_features_12(features12)
												   .set_required_features_13(features13)
												   .select();
	if (!physical_ret)
	{
		Logger::instance().error("Failed to select physical device: {}", physical_ret.error().message());
		return std::unexpected(PhysicalDeviceCreationError{vk_result_of(physical_ret)});
	}
	const auto& vkb_physical = physical_ret.value();
	Logger::instance().info("Selected GPU: {}", vkb_physical.name);

	auto device_ret = vkb::DeviceBuilder{vkb_physical}.build();
	if (!device_ret)
	{
		Logger::instance().error("Failed to create device: {}", device_ret.error().message());
		return std::unexpected(DeviceCreationError{vk_result_of(device_ret)});
	}
	DeviceGuard device_guard{device_ret.value()};
	const auto& vkb_device = device_guard.device;

	VULKAN_HPP_DEFAULT_DISPATCHER.init(vk::Device(vkb_device.device));
	Logger::instance().debug("Created logical device");

	// vk-bootstrap picks a dedicated compute queue when one exists, falling back
	// to a shared graphics/compute family otherwise.
	auto graphics_queue = vkb_device.get_queue(vkb::QueueType::graphics);
	auto compute_queue	= vkb_device.get_queue(vkb::QueueType::compute);
	auto graphics_index = vkb_device.get_queue_index(vkb::QueueType::graphics);
	auto compute_index	= vkb_device.get_queue_index(vkb::QueueType::compute);
	if (!graphics_queue || !compute_queue || !graphics_index || !compute_index)
	{
		Logger::instance().error("Failed to retrieve graphics/compute queues");
		return std::unexpected(NoQueueFamilyError{});
	}
	QueueFamilyIndices indices{graphics_index.value(), compute_index.value()};
	Logger::instance().debug("Queue families - graphics: {}, compute: {}", indices.graphics, indices.compute);

	// VMA fetches its entry points from the same dynamic dispatcher.
	vma::VulkanFunctions	 functions = vma::functionsFromDispatcher(VULKAN_HPP_DEFAULT_DISPATCHER);
	vma::AllocatorCreateInfo allocator_info{};
	allocator_info.physicalDevice	= vk::PhysicalDevice(vkb_physical.physical_device);
	allocator_info.device			= vk::Device(vkb_device.device);
	allocator_info.instance			= vk::Instance(vkb_instance.instance);
	allocator_info.vulkanApiVersion = VK_API_VERSION_1_3;
	allocator_info.pVulkanFunctions = &functions;

	vma::Allocator allocator;
	if (auto result = vma::createAllocator(&allocator_info, &allocator); result != vk::Result::eSuccess)
	{
		Logger::instance().error("Failed to create VMA allocator: {}", vk::to_string(result));
		return std::unexpected(AllocatorCreationError{result});
	}
	Logger::instance().debug("Created VMA allocator");

	// Ownership transfers to Context, which destroys everything in its destructor.
	instance_guard.armed = false;
	device_guard.armed	 = false;
	return Context{vk::Instance(vkb_instance.instance),
				   vk::DebugUtilsMessengerEXT(vkb_instance.debug_messenger),
				   vk::PhysicalDevice(vkb_physical.physical_device),
				   indices,
				   vk::Device(vkb_device.device),
				   vk::Queue(graphics_queue.value()),
				   vk::Queue(compute_queue.value()),
				   allocator};
}

Context::~Context()
{
	if (m_allocator)
	{
		m_allocator.destroy();
		Logger::instance().trace("Destroyed VMA allocator");
	}

	if (m_device)
	{
		m_device.destroy();
		Logger::instance().trace("Destroyed logical device");
	}

	destroy_debug_messenger(m_instance, m_debug_messenger);

	if (m_instance)
	{
		m_instance.destroy();
		Logger::instance().trace("Destroyed instance");
	}
}
Context::Context(Context&& other) noexcept
	: m_instance(std::exchange(other.m_instance, nullptr))
	, m_debug_messenger(std::exchange(other.m_debug_messenger, nullptr))
	, m_physical_device(other.m_physical_device)
	, m_queue_indices(other.m_queue_indices)
	, m_device(std::exchange(other.m_device, nullptr))
	, m_graphics_queue(other.m_graphics_queue)
	, m_compute_queue(other.m_compute_queue)
	, m_allocator(std::exchange(other.m_allocator, nullptr))
{
}
Context& Context::operator=(Context&& other) noexcept
{
	if (this == &other)
		return *this;

	// Clean up
	if (m_allocator)
	{
		m_allocator.destroy();
	}
	if (m_device)
	{
		m_device.destroy();
	}
	destroy_debug_messenger(m_instance, m_debug_messenger);
	if (m_instance)
	{
		m_instance.destroy();
	}

	// Steal
	m_instance		  = std::exchange(other.m_instance, nullptr);
	m_debug_messenger = std::exchange(other.m_debug_messenger, nullptr);
	m_physical_device = other.m_physical_device;
	m_queue_indices	  = other.m_queue_indices;
	m_device		  = std::exchange(other.m_device, nullptr);
	m_graphics_queue  = other.m_graphics_queue;
	m_compute_queue	  = other.m_compute_queue;
	m_allocator		  = std::exchange(other.m_allocator, nullptr);
	return *this;
}

} // namespace veng
