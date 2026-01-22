//
// Created by chris on 1/22/26.
//

#ifndef VENG_CONTEXTERRORS_HPP
#define VENG_CONTEXTERRORS_HPP

#include <veng/common.hpp>
#include <vulkan/vulkan.hpp>

struct InstanceCreationError
{
	vk::Result result;
	explicit(false) operator vk::Result() const { return result; }
};

struct DebugUtilsMessengerEXTCreationError
{
	vk::Result result;
	explicit(false) operator vk::Result() const { return result; }
};

struct PhysicalDeviceCreationError
{
	vk::Result result;
	explicit(false) operator vk::Result() const { return result; }
};


struct NoQueueFamilyError {};
inline std::string to_string(const NoQueueFamilyError&)
{
	return "Could not find Queue Families";
}

struct DeviceCreationError
{
	vk::Result result;
	explicit(false) operator vk::Result() const { return result; }
};


using ContextCreationError = ResultVariant<InstanceCreationError, DebugUtilsMessengerEXTCreationError, PhysicalDeviceCreationError, NoQueueFamilyError, DeviceCreationError>;


#endif // VENG_CONTEXTERRORS_HPP
