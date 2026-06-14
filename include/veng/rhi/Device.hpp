/**
 * @file
 * @author chris
 * @brief The RHI device: an opaque-handle registry AND a resource/command factory.
 *
 * Two roles in one type. As a **registry**, `Device` is the indirection that lets graph edges carry
 * opaque handles instead of raw `vk::Image`/`vk::Buffer`: a resource registers its vk objects and
 * gets a stable handle, and a consumer recording a command resolves the handle back to the vk
 * objects through the device. The engine's `Image`/`Buffer` wrappers and the swapchain register here
 * and own their own vk objects (the registry slot is non-owning for those).
 *
 * As a **factory/device**, it can also create the resources and run the commands itself, entirely in
 * RHI vocabulary: @ref create_texture / @ref create_buffer allocate device-owned resources, @ref
 * create_sampler owns samplers, and @ref begin_commands + @ref submit hand out a one-shot @ref
 * CommandEncoder and execute it. This is what lets a caller render without ever naming Vulkan (see
 * `example/rhi_triangle`). Device-created objects, the sampler pool, and the one-shot command pool +
 * fence are freed in the destructor.
 *
 * One `Device` lives inside the engine @ref veng::Context, destroyed before the `vk::Device`/
 * allocator it borrows. Handle resolution is read-only and lock-free; create/destroy happen at
 * resource setup/teardown, not in the record hot path.
 *
 * @ingroup rhi
 */

#ifndef VENG_RHI_DEVICE_HPP
#define VENG_RHI_DEVICE_HPP

#include <cstdint>
#include <expected>
#include <span>
#include <vector>
#include <veng/rhi/BindGroup.hpp>
#include <veng/rhi/Enums.hpp>
#include <veng/rhi/Error.hpp>
#include <veng/rhi/Handles.hpp>
#include <vulkan-memory-allocator-hpp/vk_mem_alloc.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::rhi
{
class CommandEncoder;

/// @brief Parameters for @ref Device::create_texture — a 2D image in RHI vocabulary.
struct TextureDesc
{
	Extent2D		  extent;					 ///< Width/height in texels.
	Format			  format;					 ///< Pixel format.
	TextureUsageFlags usage;					 ///< What the texture may be used for.
	SampleCount		  samples = SampleCount::X1; ///< MSAA sample count.
};

/// @brief Parameters for @ref Device::create_buffer — a GPU buffer in RHI vocabulary.
struct BufferDesc
{
	std::uint64_t	 size;							  ///< Size in bytes.
	BufferUsageFlags usage;							  ///< What the buffer may be bound as.
	MemoryAccess	 memory = MemoryAccess::GPU_ONLY; ///< Where it lives / whether it is host-mapped.
};

/**
 * @brief The RHI device: a handle registry AND the factory that creates GPU resources and records +
 *        submits commands — enough to render without naming Vulkan.
 *
 * Two roles. (1) Registry: a resource registers its vk objects and gets a stable opaque handle; a
 * consumer resolves a handle back to its vk objects when it records. The engine's `Image`/`Buffer`/
 * swapchain register here and own their own vk objects (non-owning slots). (2) Device: @ref
 * create_texture / @ref create_buffer allocate RHI-owned resources (freed on @ref destroy_texture /
 * @ref destroy_buffer or with the device); @ref create_sampler owns samplers; @ref begin_commands +
 * @ref submit give a one-shot @ref CommandEncoder and run it. A caller can therefore create targets,
 * record a pass, and submit entirely in RHI vocabulary.
 *
 * Non-copyable and non-movable: the engine @ref veng::Context holds exactly one via `unique_ptr`, so
 * only the pointer ever moves. It is destroyed before the `vk::Device`/allocator it borrows.
 *
 * @ingroup rhi
 */
class Device
{
	 public:
	/// @param device          The logical device used to create owned objects (samplers, views, commands).
	/// @param allocator       The VMA allocator backing @ref create_texture / @ref create_buffer.
	/// @param graphics_queue  The queue @ref submit submits to.
	/// @param graphics_family The graphics queue family the one-shot command pool is created for.
	Device(vk::Device device, vma::Allocator allocator, vk::Queue graphics_queue,
		   std::uint32_t graphics_family) noexcept;
	Device(const Device&)			 = delete;
	Device& operator=(const Device&) = delete;
	Device(Device&&)				 = delete;
	Device& operator=(Device&&)		 = delete;
	~Device();

	// --- device-level resource creation (RHI-owned; freed on destroy_* or with the device) --------

	/// @brief Allocate an RHI-owned texture (image + view, registered), returning its handle.
	[[nodiscard]] std::expected<TextureHandle, Error> create_texture(const TextureDesc& desc);
	/// @brief Free a texture created by @ref create_texture (image, view, memory) and its slot.
	void destroy_texture(TextureHandle handle) noexcept;
	/// @brief Allocate an RHI-owned buffer, returning its handle. HOST_VISIBLE buffers are mapped.
	[[nodiscard]] std::expected<BufferHandle, Error> create_buffer(const BufferDesc& desc);
	/// @brief Free a buffer created by @ref create_buffer and its slot.
	void destroy_buffer(BufferHandle handle) noexcept;
	/// @brief The persistently-mapped host pointer for a HOST_VISIBLE buffer (null otherwise).
	[[nodiscard]] void* mapped(BufferHandle handle) const noexcept;

	// --- commands ---------------------------------------------------------------------------------

	/// @brief Begin recording a one-shot command encoder (the device owns the command buffer).
	[[nodiscard]] CommandEncoder begin_commands();
	/// @brief End @p enc, submit it on the graphics queue, and block until it completes.
	[[nodiscard]] std::expected<void, Error> submit(CommandEncoder& enc);

	/// @brief Register an image + its default view, returning a stable handle to them.
	[[nodiscard]] TextureHandle register_texture(vk::Image image, vk::ImageView view);
	/// @brief Rebind an existing handle's slot to new vk objects (e.g. a swapchain image on rebuild).
	void rebind_texture(TextureHandle handle, vk::Image image, vk::ImageView view) noexcept;
	/// @brief Free a texture slot for reuse (called by the owning `Image`/swapchain on destroy).
	void release_texture(TextureHandle handle) noexcept;
	/// @brief Resolve a texture handle to its `vk::Image`, or a null handle if invalid/released.
	[[nodiscard]] vk::Image image(TextureHandle handle) const noexcept;
	/// @brief Resolve a texture handle to its `vk::ImageView`, or a null handle if invalid/released.
	[[nodiscard]] vk::ImageView view(TextureHandle handle) const noexcept;

	/// @brief Register a buffer, returning a stable handle to it.
	[[nodiscard]] BufferHandle register_buffer(vk::Buffer buffer);
	/// @brief Free a buffer slot for reuse (called by the owning `Buffer` on destroy).
	void release_buffer(BufferHandle handle) noexcept;
	/// @brief Resolve a buffer handle to its `vk::Buffer`, or a null handle if invalid/released.
	[[nodiscard]] vk::Buffer buffer(BufferHandle handle) const noexcept;

	/// @brief Register a semaphore (non-owning — the swapchain owns it), returning a stable handle.
	[[nodiscard]] SemaphoreHandle register_semaphore(vk::Semaphore semaphore);
	/// @brief Free a semaphore slot for reuse (called by the owning swapchain on rebuild/destroy).
	void release_semaphore(SemaphoreHandle handle) noexcept;
	/// @brief Resolve a semaphore handle to its `vk::Semaphore`, or a null handle if invalid/released.
	[[nodiscard]] vk::Semaphore semaphore(SemaphoreHandle handle) const noexcept;

	/// @brief Create a device-owned sampler from @p info, returning a handle (or an @ref Error on failure).
	[[nodiscard]] std::expected<SamplerHandle, Error> create_sampler(const vk::SamplerCreateInfo& info);
	/// @brief Destroy a sampler and free its slot (optional — all owned samplers are freed on device destroy).
	void release_sampler(SamplerHandle handle) noexcept;
	/// @brief Resolve a sampler handle to its `vk::Sampler`, or a null handle if invalid/released.
	[[nodiscard]] vk::Sampler sampler(SamplerHandle handle) const noexcept;

	/// @brief Register a pipeline + its layout, returning a stable handle (non-owning — `GraphicsPipeline` owns them).
	[[nodiscard]] PipelineHandle register_pipeline(vk::Pipeline pipeline, vk::PipelineLayout layout);
	/// @brief Free a pipeline slot for reuse (called by the owning `GraphicsPipeline` on destroy).
	void release_pipeline(PipelineHandle handle) noexcept;
	/// @brief Resolve a pipeline handle to its `vk::Pipeline`, or a null handle if invalid/released.
	[[nodiscard]] vk::Pipeline pipeline(PipelineHandle handle) const noexcept;
	/// @brief Resolve a pipeline handle to its `vk::PipelineLayout`, or a null handle if invalid/released.
	[[nodiscard]] vk::PipelineLayout pipeline_layout(PipelineHandle handle) const noexcept;

	/// @brief Write @p entries (resolving their handles) into @p group's descriptor set — the single
	///        `updateDescriptorSets` behind @ref veng::rhi::BindGroup. No caller names a vk write struct.
	void update_bind_group(const BindGroup& group, std::span<const BindGroupEntry> entries) const;

	 private:
	struct Texture
	{
		vk::Image		image;
		vk::ImageView	view;
		vma::Allocation allocation{}; ///< Non-null only for device-created textures (the device frees those).
	};
	struct Buffer
	{
		vk::Buffer		buffer;
		vma::Allocation allocation{}; ///< Non-null only for device-created buffers.
		void*			mapped{};	  ///< Host pointer for a HOST_VISIBLE device-created buffer.
	};
	struct Pipeline
	{
		vk::Pipeline	   pipeline;
		vk::PipelineLayout layout;
	};
	vk::Device				   m_device;		   ///< For owned-sampler/view/command create + destroy.
	vma::Allocator			   m_allocator{};	   ///< Backs create_texture / create_buffer.
	vk::Queue				   m_graphics_queue{}; ///< submit() target.
	vk::CommandPool			   m_command_pool{};   ///< One-shot pool for begin_commands().
	vk::CommandBuffer		   m_command_buffer{}; ///< The reusable one-shot buffer begin_commands() hands out.
	vk::Fence				   m_fence{};		   ///< submit() waits on this.
	std::vector<Texture>	   m_textures;
	std::vector<std::uint32_t> m_free_textures;
	std::vector<Buffer>		   m_buffers;
	std::vector<std::uint32_t> m_free_buffers;
	std::vector<vk::Semaphore> m_semaphores; ///< Non-owning; the swapchain owns the vk objects.
	std::vector<std::uint32_t> m_free_semaphores;
	std::vector<vk::Sampler>   m_samplers; ///< Device-owned; destroyed in the destructor.
	std::vector<std::uint32_t> m_free_samplers;
	std::vector<Pipeline>	   m_pipelines; ///< Non-owning; `GraphicsPipeline` owns the vk objects.
	std::vector<std::uint32_t> m_free_pipelines;
};
} // namespace veng::rhi

#endif // VENG_RHI_DEVICE_HPP
