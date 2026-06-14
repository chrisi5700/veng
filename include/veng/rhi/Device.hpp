/**
 * @file
 * @author chris
 * @brief The RHI resource registry: maps opaque @ref veng::rhi handles to their Vulkan objects.
 *
 * `Device` is the indirection that lets graph edges carry opaque handles instead of raw `vk::Image`
 * / `vk::Buffer`. A resource registers its vk objects here on creation and gets back a stable
 * handle; a consumer recording a command resolves the handle to the vk objects through the device.
 * The device does **not** own the textures/buffers — the `Image`/`Buffer` RAII wrappers (and the
 * swapchain) own them and register/release their slots over their own lifetime. It is therefore a
 * pure slot-map: registration hands out an id, release frees the slot for reuse, resolution is a
 * bounds-checked lookup that returns a null vk handle for an invalid/released id.
 *
 * One `Device` lives inside the engine @ref veng::Context. Resolution is read-only and lock-free;
 * registration/release happen at resource create/destroy time (frame setup + teardown), not in the
 * record hot path.
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
#include <veng/rhi/Handles.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::rhi
{
/**
 * @brief Handle→Vulkan-object registry for textures and buffers, and the owner of RHI samplers.
 *
 * Textures/buffers are non-owning: their vk objects are owned by the `Image`/`Buffer`/swapchain and
 * the device only maps handle ids to them. Samplers, by contrast, the device owns outright —
 * @ref create_sampler builds one and the device destroys all of them in its destructor, so a node
 * no longer captures a `vk::Device` to free its own sampler.
 *
 * Non-copyable and non-movable: the engine @ref veng::Context holds exactly one via `unique_ptr`,
 * so only the pointer ever moves, never this object (which keeps owned-sampler teardown trivial).
 *
 * @ingroup rhi
 */
class Device
{
	 public:
	/// @param device The logical device used to create and destroy owned samplers.
	explicit Device(vk::Device device) noexcept
		: m_device(device)
	{
	}
	Device(const Device&)			 = delete;
	Device& operator=(const Device&) = delete;
	Device(Device&&)				 = delete;
	Device& operator=(Device&&)		 = delete;
	~Device();

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

	/// @brief Create a device-owned sampler from @p info, returning a handle (or the `vk::Result` on failure).
	[[nodiscard]] std::expected<SamplerHandle, vk::Result> create_sampler(const vk::SamplerCreateInfo& info);
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
		vk::Image	  image;
		vk::ImageView view;
	};
	struct Pipeline
	{
		vk::Pipeline	   pipeline;
		vk::PipelineLayout layout;
	};
	vk::Device				   m_device; ///< For owned-sampler create/destroy.
	std::vector<Texture>	   m_textures;
	std::vector<std::uint32_t> m_free_textures;
	std::vector<vk::Buffer>	   m_buffers;
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
