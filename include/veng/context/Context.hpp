/**
 * @file
 * @author chris
 * @brief Engine-wide Vulkan context: instance, physical device, logical device, and VMA
 *        allocator, all created together and destroyed in order.
 *
 * @ref veng::Context is the root Vulkan object. Every other manager borrows from it by const
 * reference. Construction is done through the @ref veng::Context::create factory which returns
 * `std::expected<Context, ContextCreationError>` so failures are typed and composable.
 *
 * Without extra arguments the context is headless (suitable for unit tests). For windowed
 * rendering, pass the platform instance extensions and a surface factory; the @ref veng::Context
 * then owns the surface and verifies the graphics queue can present to it.
 *
 * @ingroup context
 */

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
/**
 * @brief Graphics and compute queue-family indices selected during device creation.
 * @ingroup context
 */
struct QueueFamilyIndices
{
	uint32_t graphics; ///< Queue family index used for graphics and presentation.
	uint32_t compute;  ///< Queue family index used for compute (may equal `graphics`).

	/** @return `true` when the compute queue belongs to a different family than graphics. */
	[[nodiscard]] bool has_dedicated_compute() const { return compute != graphics; }
};

/**
 * @brief Engine-wide Vulkan context: the root object every other subsystem borrows from.
 *
 * Owns the `vk::Instance`, `vk::Device`, `vma::Allocator`, and — for windowed contexts —
 * the `vk::SurfaceKHR`. Construction is performed through @ref create; move-only.
 *
 * @ingroup context
 * @see ContextCreationError
 * @see SwapchainManager
 * @see CommandManager
 */
class Context
{
	 public:
	/**
	 * @brief Create the Vulkan context.
	 *
	 * With no extra arguments this produces a headless context (suitable for all unit
	 * tests). For windowed rendering, pass the window's required instance extensions and a
	 * @p surface_factory that creates a `VkSurfaceKHR` from the instance; the @ref veng::Context
	 * then owns that surface and verifies the graphics queue can present to it.
	 *
	 * @param title                Application title passed to the Vulkan instance.
	 * @param instance_extensions  Platform surface extensions required by the window system
	 *                             (empty for headless).
	 * @param surface_factory      Callable that creates a `VkSurfaceKHR` from the instance
	 *                             (empty for headless).
	 * @return A fully initialised @ref veng::Context, or a @ref ContextCreationError variant
	 *         identifying which creation step failed.
	 */
	static std::expected<Context, ContextCreationError> create(
		std::string_view title, std::span<const char* const> instance_extensions = {},
		const std::function<VkSurfaceKHR(VkInstance)>& surface_factory = {});
	~Context();

	Context(const Context&)			   = delete;
	Context& operator=(const Context&) = delete;
	Context(Context&& other) noexcept;
	Context& operator=(Context&& other) noexcept;

	/** @return The Vulkan instance. */
	[[nodiscard]] vk::Instance instance() const { return m_instance; }
	/** @return The selected physical device. */
	[[nodiscard]] vk::PhysicalDevice physical_device() const { return m_physical_device; }
	/** @return The graphics and compute queue-family indices. */
	[[nodiscard]] QueueFamilyIndices queue_indices() const { return m_queue_indices; }
	/** @return The logical device. */
	[[nodiscard]] vk::Device device() const { return m_device; }
	/** @return The graphics queue (also used for presentation). */
	[[nodiscard]] vk::Queue graphics_queue() const { return m_graphics_queue; }
	/** @return The compute queue (may be the same as the graphics queue). */
	[[nodiscard]] vk::Queue compute_queue() const { return m_compute_queue; }
	/** @return The VMA allocator. */
	[[nodiscard]] vma::Allocator allocator() const { return m_allocator; }
	/** @return The window surface, or a null handle for a headless context. */
	[[nodiscard]] vk::SurfaceKHR surface() const { return m_surface; }

	/**
	 * @brief Execute GPU work synchronously on the graphics queue.
	 *
	 * Allocates a transient command buffer, calls @p record, then ends, submits with a
	 * private fence, waits for completion, and frees all resources. This is the engine's
	 * "do this on the GPU now" primitive — it replaces ad-hoc pool/fence/submit/wait
	 * sequences in user code (e.g. resource uploads, constant-image initialisation).
	 *
	 * @param record Callable that records GPU work into the provided command buffer.
	 * @return `vk::Result::eSuccess` on success, or the first failing `vk::Result`.
	 */
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
	vk::SurfaceKHR			   m_surface; ///< Null for headless; owned and destroyed before the instance.
};
} // namespace veng
#endif // VENG_CONTEXT_HPP
