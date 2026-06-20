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

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <veng/rhi/Device.hpp>
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
 * @note Lifetime: every other subsystem (@ref SwapchainManager, @ref CommandManager,
 *       @ref veng::ResourcePool, the managers and nodes) borrows the Context by reference and does
 *       NOT extend its lifetime. The Context must therefore outlive every object built from it;
 *       destroy them in reverse order of creation. The borrow is unchecked — destroying the Context
 *       first dangles those references.
 * @note Threading: a Context is not internally synchronized. Treat creation/destruction as
 *       single-threaded; concurrent GPU recording is supported through per-thread command pools
 *       (see @ref CommandManager), not by sharing one Context method across threads.
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
	 * @param shader_search_paths  Extra directories the Slang compiler searches for shader modules,
	 *                             in addition to veng's own shipped shaders (which are always
	 *                             searched first). This is how a downstream consumer points
	 *                             @ref veng::Shader::create_shader at its own `.slang` files.
	 * @return A fully initialised @ref veng::Context, or a @ref ContextCreationError variant
	 *         identifying which creation step failed.
	 */
	static std::expected<Context, ContextCreationError> create(
		std::string_view title, std::span<const char* const> instance_extensions = {},
		const std::function<VkSurfaceKHR(VkInstance)>& surface_factory	   = {},
		std::span<const std::string_view>			   shader_search_paths = {});

	/**
	 * @brief Adopt a Vulkan instance + device owned by a host framework (e.g. Qt's `QVulkanWindow`).
	 *
	 * Wraps externally-created Vulkan objects instead of creating its own: veng builds only the
	 * resources it owns — the VMA allocator and the @ref veng::rhi::Device registry — on the borrowed
	 * @p device, and its destructor frees *only those*, never the adopted instance/device (the host
	 * keeps owning those). The graphics queue doubles as the compute queue (the common case for an
	 * embedded surface owned by the host). The host is responsible for having requested every device
	 * feature/extension veng needs (dynamic rendering, synchronization2, …) when it created @p device.
	 *
	 * No surface is owned here: a host that drives its own presentation (Qt owns the swapchain)
	 * renders veng's frame into an offscreen target and blits it into the host's image — so no
	 * @ref veng::rhi::Swapchain is involved on the adopt path.
	 *
	 * @param instance             The host-owned `vk::Instance`.
	 * @param physical_device      The physical device the host selected.
	 * @param device               The host-owned logical `vk::Device` (not destroyed by veng).
	 * @param graphics_queue       A graphics-capable queue on @p device (also used for compute).
	 * @param graphics_family      The queue family index @p graphics_queue belongs to.
	 * @param shader_search_paths  Extra Slang shader search directories (see @ref create).
	 * @return A Context borrowing the host's instance/device, or a @ref ContextCreationError.
	 */
	static std::expected<Context, ContextCreationError> adopt(
		vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device, vk::Queue graphics_queue,
		std::uint32_t graphics_family, std::span<const std::string_view> shader_search_paths = {});
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
	/** @return The RHI handle registry (texture/buffer handle ↔ vk-object indirection). */
	[[nodiscard]] rhi::Device& rhi() const noexcept { return *m_rhi; }
	/** @return The extra Slang shader search directories this context was created with. */
	[[nodiscard]] std::span<const std::string> shader_search_paths() const noexcept { return m_shader_search_paths; }

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
			vk::Queue m_compute_queue, vma::Allocator m_allocator, vk::SurfaceKHR m_surface,
			std::vector<std::string> shader_search_paths, bool owns_instance)
		: m_instance(m_instance)
		, m_debug_messenger(m_debug_messenger)
		, m_physical_device(m_physical_device)
		, m_queue_indices(m_queue_indices)
		, m_device(m_device)
		, m_graphics_queue(m_graphics_queue)
		, m_compute_queue(m_compute_queue)
		, m_allocator(m_allocator)
		, m_surface(m_surface)
		, m_shader_search_paths(std::move(shader_search_paths))
		, m_owns_instance(owns_instance)
	{
		m_rhi = std::make_unique<rhi::Device>(m_device, m_allocator, m_graphics_queue, m_queue_indices.graphics);
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
	vk::SurfaceKHR			   m_surface;			  ///< Null for headless; owned and destroyed before the instance.
	std::vector<std::string>   m_shader_search_paths; ///< Extra Slang shader search dirs (see @ref create).
	/// Whether this context owns its instance/device (the @ref create path) or borrows them from a host
	/// (the @ref adopt path). Gates whether the destructor tears the instance/device/surface down — the
	/// VMA allocator and @ref veng::rhi::Device are veng-created either way and always freed.
	bool m_owns_instance = true;
	/// The RHI device (handle registry + resource/command factory). Declared last so it is destroyed
	/// FIRST — its owned vk objects (created textures/buffers, sampler, command pool, fence) are freed
	/// while the borrowed `vk::Device`/allocator above are still alive.
	std::unique_ptr<rhi::Device> m_rhi;
};
} // namespace veng
#endif // VENG_CONTEXT_HPP
