/**
 * @file
 * @author chris
 * @brief Typed error structs for each step of @ref veng::Context creation, collected into the
 *        @ref ContextCreationError variant.
 *
 * Each struct wraps a `vk::Result` and provides an implicit conversion to it so the
 * @ref ResultType concept is satisfied via the generic `to_string(vk::Result)` overload in
 * `common.hpp`.
 *
 * @ingroup context
 */

#ifndef VENG_CONTEXTERRORS_HPP
#define VENG_CONTEXTERRORS_HPP

#include <veng/common.hpp>
#include <vulkan/vulkan.hpp>

/** @brief Failure creating the `vk::Instance`. @ingroup context */
struct InstanceCreationError
{
	vk::Result		result;
	explicit(false) operator vk::Result() const { return result; }
};

/** @brief Failure creating the debug-utils messenger (validation layers only). @ingroup context */
struct DebugUtilsMessengerEXTCreationError
{
	vk::Result		result;
	explicit(false) operator vk::Result() const { return result; }
};

/** @brief Failure selecting a suitable physical device. @ingroup context */
struct PhysicalDeviceCreationError
{
	vk::Result		result;
	explicit(false) operator vk::Result() const { return result; }
};

/**
 * @brief Raised when no queue family supports the required graphics and compute operations.
 * @ingroup context
 */
struct NoQueueFamilyError
{
};
/** @return A human-readable description of the error. */
inline std::string to_string(const NoQueueFamilyError&)
{
	return "Could not find Queue Families";
}

/** @brief Failure creating the `vk::Device`. @ingroup context */
struct DeviceCreationError
{
	vk::Result		result;
	explicit(false) operator vk::Result() const { return result; }
};

/** @brief Failure creating the VMA allocator. @ingroup context */
struct AllocatorCreationError
{
	vk::Result		result;
	explicit(false) operator vk::Result() const { return result; }
};

/**
 * @brief Raised when a window surface could not be created from the provided factory, or
 *        the selected graphics queue family cannot present to it.
 * @ingroup context
 */
struct SurfaceCreationError
{
	vk::Result		result;
	explicit(false) operator vk::Result() const { return result; }
};

/**
 * @brief Variant over all typed errors that @ref veng::Context::create can return.
 *
 * Any constituent error can be extracted with `std::get` or `std::visit`; the
 * @ref ResultType concept means it can be stringified uniformly via `to_string`.
 *
 * @ingroup context
 * @see Context::create
 */
using ContextCreationError =
	ResultVariant<InstanceCreationError, DebugUtilsMessengerEXTCreationError, PhysicalDeviceCreationError,
				  NoQueueFamilyError, DeviceCreationError, AllocatorCreationError, SurfaceCreationError>;

#endif // VENG_CONTEXTERRORS_HPP
