//
// Created by chris on 1/22/26.
//
// VulkanContext.cpp

#include <GLFW/glfw3.h>
#include <optional>
#include <set>
#include <veng/context/Context.hpp>
#include <veng/context/ContextErrors.hpp>
#include <veng/logging/Logger.hpp>
#include <vulkan/vulkan.hpp>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace veng
{

#ifdef NDEBUG
constexpr bool ENABLE_VALIDATION = false;
#else
constexpr bool ENABLE_VALIDATION = true;
#endif

constexpr std::array VALIDATION_LAYERS = {"VK_LAYER_KHRONOS_validation"};

vk::Bool32 debug_callback(vk::DebugUtilsMessageSeverityFlagBitsEXT			 severity,
						  [[maybe_unused]] vk::DebugUtilsMessageTypeFlagsEXT type,
						  const vk::DebugUtilsMessengerCallbackDataEXT* callback_data, [[maybe_unused]] void* user_data)
{
	auto& logger  = Logger::instance();
	auto  pattern = fmt::format("[IteratedFunction]{:<30}[%^%5l%$] %v", "[VulkanDebug]");
	logger.set_pattern(pattern);
	switch (static_cast<VkDebugUtilsMessageSeverityFlagBitsEXT>(severity))
	{
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT: logger.trace("{}", callback_data->pMessage); break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT: logger.debug("{}", callback_data->pMessage); break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT: logger.warn("{}", callback_data->pMessage); break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT: logger.error("{}", callback_data->pMessage); break;
		default: logger.info("{}", callback_data->pMessage); break;
	}

	return vk::False;
}

bool check_validation_layer_support()
{
	auto available_res = vk::enumerateInstanceLayerProperties();
	if (available_res.result != vk::Result::eSuccess)
	{
		Logger::instance().warn("Could not query InstanceLayerProperties {}", to_string(available_res.result));
	}
	auto available = std::move(available_res.value);
	for (const char* layer_name : VALIDATION_LAYERS)
	{
		bool found = false;
		for (const auto& layer : available)
		{
			if (std::strcmp(layer_name, layer.layerName) == 0)
			{
				found = true;
				break;
			}
		}
		if (!found)
		{
			Logger::instance().warn("Validation layer {} not available", layer_name);
			return false;
		}
	}
	return true;
}

vk::DebugUtilsMessengerCreateInfoEXT make_debug_messenger_create_info()
{
	return vk::DebugUtilsMessengerCreateInfoEXT()
		.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
							vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
		.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
						vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
						vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance)
		.setPfnUserCallback(debug_callback);
}

std::expected<vk::Instance, InstanceCreationError> create_instance(std::string_view title)
{
	static vk::detail::DynamicLoader dl;
	auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
	VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
	auto app_info = vk::ApplicationInfo()
						.setPApplicationName(title.data())
						.setApplicationVersion(VK_MAKE_VERSION(1, 0, 0))
						.setPEngineName("No Engine")
						.setEngineVersion(VK_MAKE_VERSION(1, 0, 0))
						.setApiVersion(VK_API_VERSION_1_3);

	uint32_t	 glfw_extension_count = 0;
	const char** glfw_extensions	  = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

	std::vector<const char*> extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

	if constexpr (ENABLE_VALIDATION)
	{
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}
	extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);

	Logger::instance().debug("Instance extensions:");
	for (const auto* ext : extensions)
	{
		Logger::instance().debug("  {}", ext);
	}

	auto create_info = vk::InstanceCreateInfo().setPApplicationInfo(&app_info).setPEnabledExtensionNames(extensions);

	vk::DebugUtilsMessengerCreateInfoEXT debug_create_info;
	if constexpr (ENABLE_VALIDATION)
	{
		if (check_validation_layer_support())
		{
			create_info.setPEnabledLayerNames(VALIDATION_LAYERS);

			debug_create_info = make_debug_messenger_create_info();
			create_info.setPNext(&debug_create_info);

			Logger::instance().info("Validation layers enabled");
		}
	}

	auto instance_res = vk::createInstance(create_info);
	if (instance_res.result != vk::Result::eSuccess)
	{
		Logger::instance().error("Failed to create Vulkan instance {}", to_string(instance_res.result));
		return std::unexpected(InstanceCreationError{instance_res.result});
	}
	auto instance = instance_res.value;
	VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
	Logger::instance().debug("Created Vulkan instance");
	return instance;
}

std::expected<vk::DebugUtilsMessengerEXT, DebugUtilsMessengerEXTCreationError> create_debug_messenger(
	vk::Instance instance)
{
	if constexpr (!ENABLE_VALIDATION)
	{
		return nullptr;
	}

	auto create_info = make_debug_messenger_create_info();

	auto debug_msngr_res = instance.createDebugUtilsMessengerEXT(create_info, nullptr);
	if (debug_msngr_res.result != vk::Result::eSuccess)
	{
		Logger::instance().error("Failed to create debug messenger {}", to_string(debug_msngr_res.result));
		return std::unexpected(DebugUtilsMessengerEXTCreationError{debug_msngr_res.result});
	}
	Logger::instance().debug("Created debug messenger");
	return debug_msngr_res.value;
}

void destroy_debug_messenger(vk::Instance instance, vk::DebugUtilsMessengerEXT messenger)
{
	if (!messenger)
		return;

	instance.destroyDebugUtilsMessengerEXT(messenger, nullptr);
	Logger::instance().trace("Destroyed debug messenger");
}

std::expected<vk::PhysicalDevice, PhysicalDeviceCreationError> select_physical_device(vk::Instance instance)
{
	auto devices_res = instance.enumeratePhysicalDevices();
	if (devices_res.result != vk::Result::eSuccess)
	{
		Logger::instance().error("Failed to enumerate physical devices {}", to_string(devices_res.result));
		return std::unexpected(PhysicalDeviceCreationError{devices_res.result});
	}
	auto devices = std::move(devices_res.value);

	for (const auto& dev : devices)
	{
		auto props = dev.getProperties();
		if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
		{
			Logger::instance().info("Selected discrete GPU: {}", props.deviceName.data());
			return dev;
		}
	}

	for (const auto& dev : devices)
	{
		auto props = dev.getProperties();
		if (props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu)
		{
			Logger::instance().info("Selected integrated GPU: {}", props.deviceName.data());
			return dev;
		}
	}

	return std::unexpected(PhysicalDeviceCreationError{vk::Result::eErrorUnknown});
}

std::expected<QueueFamilyIndices, NoQueueFamilyError> find_queue_families(vk::PhysicalDevice physical_device)
{
	auto queue_families = physical_device.getQueueFamilyProperties();

	std::optional<uint32_t> graphics;
	std::optional<uint32_t> compute;

	for (uint32_t i = 0; i < queue_families.size(); i++)
	{
		const auto& family = queue_families[i];

		if (family.queueFlags & vk::QueueFlagBits::eGraphics)
		{
			graphics = i;
		}

		if (family.queueFlags & vk::QueueFlagBits::eCompute)
		{
			if (!compute.has_value() || !(family.queueFlags & vk::QueueFlagBits::eGraphics))
			{
				compute = i;
			}
		}
	}

	if (!graphics || !compute)
	{
		return std::unexpected(NoQueueFamilyError{});
	}

	Logger::instance().debug("Queue families - graphics: {}, compute: {}", *graphics, *compute);

	return QueueFamilyIndices{*graphics, *compute};
}

std::expected<vk::Device, DeviceCreationError> create_logical_device(vk::PhysicalDevice		   physical_device,
																	 const QueueFamilyIndices& indices)
{
	std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
	std::set<uint32_t>					   unique_families = {indices.graphics, indices.compute};

	float queue_priority = 1.0f;
	for (uint32_t family : unique_families)
	{
		auto queue_create_info =
			vk::DeviceQueueCreateInfo().setQueueFamilyIndex(family).setQueueCount(1).setPQueuePriorities(
				&queue_priority);
		queue_create_infos.push_back(queue_create_info);
	}

	std::vector<const char*> extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

	// Base features
	vk::PhysicalDeviceFeatures features{};
	features.tessellationShader = VK_TRUE;
	features.geometryShader		= VK_TRUE;

	// Vulkan 1.1 features
	vk::PhysicalDeviceVulkan11Features vulkan11_features{};
	vulkan11_features.shaderDrawParameters = VK_TRUE;

	// Vulkan 1.2 features
	vk::PhysicalDeviceVulkan12Features vulkan12_features{};
	vulkan12_features.pNext = &vulkan11_features;

	// Features2 container
	vk::PhysicalDeviceFeatures2 features2{};
	features2.features = features;
	features2.pNext	   = &vulkan12_features;

	auto create_info = vk::DeviceCreateInfo()
						   .setQueueCreateInfos(queue_create_infos)
						   .setPEnabledExtensionNames(extensions)
						   .setPNext(&features2);

	auto device_res = physical_device.createDevice(create_info);
	if (device_res.result != vk::Result::eSuccess)
	{
		Logger::instance().error("Failed to create device {}", to_string(device_res.result));
		return std::unexpected(DeviceCreationError{device_res.result});
	}
	Logger::instance().debug("Created logical device");
	return device_res.value;
}

std::expected<Context, ContextCreationError> Context::create(std::string_view title)
{
	struct RAIIInstance
	{
		vk::Instance instance;
		vk::Instance steal() { return std::exchange(instance, nullptr); }
		~RAIIInstance()
		{
			if (instance)
			{
				instance.destroy();
			}
		}
	};
	struct RAIIDebugMSG
	{
		vk::Instance instance;
		vk::DebugUtilsMessengerEXT messenger;
		vk::DebugUtilsMessengerEXT steal()
		{
			return std::exchange(messenger, nullptr);
		}
		~RAIIDebugMSG()
		{
			if (messenger)
				destroy_debug_messenger(instance, messenger);
		}
	};
	auto instance_res = create_instance(title);
	if (not instance_res)
	{
		return std::unexpected(instance_res.error());
	}
	RAIIInstance instance{*instance_res};
	auto		 debug_msg_res = create_debug_messenger(instance.instance);
	if (not debug_msg_res)
	{
		return std::unexpected(debug_msg_res.error());
	}
	RAIIDebugMSG debug_msg{instance.instance, *debug_msg_res};
	auto physical_device_res = select_physical_device(instance.instance);
	if (not physical_device_res)
	{
		return std::unexpected(physical_device_res.error());
	}
	auto queue_family_res = find_queue_families(*physical_device_res);
	if (not queue_family_res)
	{
		return std::unexpected(queue_family_res.error());
	}
	auto device_res = create_logical_device(*physical_device_res, *queue_family_res);
	if (not device_res)
	{
		return std::unexpected(device_res.error());
	}
	auto graphics_queue = device_res->getQueue(queue_family_res->graphics, 0);
	auto compute_queue	= device_res->getQueue(queue_family_res->compute, 0);
	return Context{instance.steal(), debug_msg.steal(), *physical_device_res, *queue_family_res,
				   *device_res,		 graphics_queue, compute_queue};
}
Context::~Context()
{

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
{
}
Context& Context::operator=(Context&& other) noexcept
{
	if (this == &other)
		return *this;

	// Clean up
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
	return *this;
}

} // namespace veng