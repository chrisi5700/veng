/**
 * @file
 * @author chris
 * @brief Versioned, N-buffered transient resource pool for per-frame render targets and buffers.
 *
 * L1.5 resource layer. The pool is the engine's sole owner of per-frame render targets and
 * transient buffers: nodes never own a `vk::Image` or `Buffer` for a render target anymore —
 * they declare a logical resource once (lazily, on first record) and ask the pool for a
 * physical copy each frame. The pool decides how many physical copies a logical resource
 * needs and recycles them safely.
 *
 * **Why a pool and not one image per node:**
 * With frames-in-flight > 1, frame N+1 is recorded on the CPU while frame N still executes
 * on the GPU. If a producer re-renders into the same physical image every frame it stomps
 * pixels the GPU is still reading. The pool gives the producer a *different* physical copy
 * whenever the previous one might still be in flight.
 *
 * **The subtle retained-consumer case:**
 * A copy is retained until every frame that *touched* it — write or read — has retired. Both
 * @ref veng::ResourcePool::acquire_image (producer) and @ref veng::ResourcePool::read_image (consumer)
 * stamp the copy with the current frame; a copy is reusable only once that stamp is older
 * than the in-flight window. So a copy that a consumer keeps sampling because its producer
 * is cached is never recycled out from under it.
 *
 * @ingroup resources
 */

#ifndef VENG_RESOURCEPOOL_HPP
#define VENG_RESOURCEPOOL_HPP

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <vector>
#include <veng/gpu/BufferRef.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/gpu/UniformRef.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/resources/Image.hpp>
#include <vulkan-memory-allocator-hpp/vk_mem_alloc.hpp>
#include <vulkan/vulkan.hpp>

namespace veng
{
class Context;

/**
 * @brief Opaque, stable handle to a logical image declared on the pool.
 *
 * A node declares a logical image once (lazily on first record) and keeps this id; the
 * physical @ref veng::Image copy behind it varies per frame. Ids are assigned by
 * @ref veng::ResourcePool::declare_image and remain valid for the pool's lifetime.
 *
 * @ingroup resources
 * @see ResourcePool::declare_image
 * @see ResourcePool::acquire_image
 */
using ImageId = std::uint32_t;

/**
 * @brief Opaque, stable handle to a logical buffer declared on the pool.
 *
 * Analogous to @ref ImageId for buffers. Ids are assigned by
 * @ref veng::ResourcePool::declare_buffer and remain valid for the pool's lifetime.
 *
 * @ingroup resources
 * @see ResourcePool::declare_buffer
 * @see ResourcePool::acquire_buffer
 */
using BufferId = std::uint32_t;

/**
 * @brief Versioned, N-buffered transient resource pool.
 *
 * Manages pools of physical @ref veng::Image and @ref veng::Buffer copies behind stable logical ids.
 * Callers interact with logical resources; the pool selects and recycles physical copies
 * based on the frames-in-flight retirement window.
 *
 * @note Threading: a pool is not internally synchronized and is owned by one frame loop. Drive it
 *       from a single thread: `begin_frame` then the frame's node records (which `acquire`/`consume`)
 *       run in sequence. The retirement window assumes the caller waited frame `N - frames_in_flight`'s
 *       in-flight fence before `begin_frame(N)` (see @ref begin_frame). A consumer that binds a pooled
 *       resource whose producer may be cached this frame must @ref consume it so its copy is retained.
 *
 * @ingroup resources
 * @see ImageId
 * @see BufferId
 * @see gpu::ImageRef
 * @see gpu::BufferRef
 */
class ResourcePool
{
	 public:
	/**
	 * @brief Construct a pool bound to the given device and allocator.
	 *
	 * `frames_in_flight` is the retirement window: a copy stamped at frame F is reusable
	 * once the current frame reaches `F + frames_in_flight` (the driver guarantees frame F
	 * has retired by then, having waited that slot's fence in `acquire`).
	 *
	 * @param device           The Vulkan logical device (used to destroy image views).
	 * @param allocator        The VMA allocator (used to allocate/free images and buffers).
	 * @param frames_in_flight Number of frames that may be in flight simultaneously.
	 */
	ResourcePool(vk::Device device, rhi::Device& rhi, vma::Allocator allocator, std::size_t frames_in_flight) noexcept;

	ResourcePool(const ResourcePool&)			 = delete;
	ResourcePool& operator=(const ResourcePool&) = delete;
	ResourcePool(ResourcePool&& other) noexcept;
	ResourcePool& operator=(ResourcePool&& other) noexcept;
	~ResourcePool();

	/**
	 * @brief Advance to a new frame.
	 *
	 * `frame_index` is monotonically increasing. The caller must have ensured that the
	 * frame `frame_index - frames_in_flight` has retired on the GPU before calling (i.e.
	 * the corresponding in-flight fence has been waited).
	 *
	 * @param frame_index The new frame's monotonic index.
	 */
	void begin_frame(std::uint64_t frame_index) noexcept { m_frame = static_cast<std::int64_t>(frame_index); }

	/**
	 * @brief Declare a transient image logical resource.
	 *
	 * The extent is supplied per @ref acquire_image call (it tracks a screen-size edge); a
	 * change reallocates all physical copies (do this only when the device is idle).
	 *
	 * @param format  The pixel format of the image.
	 * @param usage   Vulkan image-usage flags.
	 * @param aspect  View aspect mask for the auto-created image view (colour or depth).
	 * @param samples Sample count; `e1` for a normal target, higher for an MSAA attachment.
	 * @return A stable @ref ImageId valid for the lifetime of this pool.
	 */
	[[nodiscard]] ImageId declare_image(vk::Format format, vk::ImageUsageFlags usage,
										vk::ImageAspectFlags	aspect	= vk::ImageAspectFlagBits::eColor,
										vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1);

	/**
	 * @brief Declare a transient host-visible buffer logical resource.
	 *
	 * The buffer is persistently mapped; size is supplied per @ref acquire_buffer call.
	 *
	 * @param usage RHI buffer-usage flags (e.g. `UNIFORM`, `STORAGE`).
	 * @return A stable @ref BufferId valid for the lifetime of this pool.
	 */
	[[nodiscard]] BufferId declare_buffer(rhi::BufferUsageFlags usage);

	/**
	 * @brief Declare an immutable image initialized to a solid clear colour.
	 *
	 * Allocates a single physical @ref veng::Image, submits an immediate command on the
	 * @ref veng::Context graphics queue to transition it from `Undefined` to `TransferDst`,
	 * clears it, and transitions it to `SHADER_READ_ONLY_OPTIMAL` — all in one fence-waited
	 * submit. The copy is marked immutable and is never recycled. Returns a
	 * @ref veng::gpu::ImageRef ready to be fed as a graph source edge.
	 *
	 * This is the clean replacement for hand-rolled command-pool + fence + clear sequences
	 * in user code.
	 *
	 * @param ctx    Engine context used for the immediate submit.
	 * @param extent Width and height of the image in texels.
	 * @param format Pixel format.
	 * @param clear  RGBA clear colour applied once at initialization.
	 * @return A @ref veng::gpu::ImageRef on success, or the `vk::Result` error on failure.
	 */
	[[nodiscard]] std::expected<gpu::ImageRef, vk::Result> constant_image(const Context& ctx, vk::Extent2D extent,
																		  vk::Format		   format,
																		  std::array<float, 4> clear);

	/**
	 * @brief Producer side: obtain a physical copy to write into this frame.
	 *
	 * Reuses the first physical copy whose last-touch stamp (write or read) has aged past
	 * the in-flight window, or allocates a fresh one. If `extent` changed since the last
	 * acquire, all existing copies are dropped and a fresh one is allocated (the caller must
	 * ensure the device is idle before an extent change). Marks the chosen copy current and
	 * stamps it with the current frame index.
	 *
	 * @param id     Logical image id returned by @ref declare_image.
	 * @param extent Desired width and height; triggers a reallocation if changed.
	 * @return A raw pointer to the physical @ref veng::Image on success, or a `vk::Result` error.
	 * @note The returned pointer is stable until the next @ref acquire_image call for this id.
	 */
	[[nodiscard]] std::expected<Image*, vk::Result> acquire_image(ImageId id, vk::Extent2D extent);

	/**
	 * @brief Consumer side: obtain the physical copy holding the latest produced version.
	 *
	 * Returns `nullptr` if the image has never been produced. Stamps the current copy's
	 * last-use to the current frame so it is retained while this frame is in flight.
	 *
	 * @param id Logical image id returned by @ref declare_image.
	 * @return A raw pointer to the current physical @ref veng::Image, or `nullptr` if never produced.
	 */
	[[nodiscard]] Image* read_image(ImageId id) noexcept;

	/**
	 * @brief Retention-only consumer stamp: keep the current copy of `id` alive this frame.
	 *
	 * Marks the current copy of `id` as used this frame without needing its `Image*` (the
	 * consumer already has the view via the published @ref veng::gpu::ImageRef). A no-op for an
	 * id that has never been produced or is out of range. This is what keeps a copy whose
	 * producer is cached from being recycled out from under a reader that is still sampling
	 * it in an in-flight frame.
	 *
	 * @param id Logical image id returned by @ref declare_image.
	 */
	void touch(ImageId id) noexcept;

	/**
	 * @brief Convenience overload: retain the pooled copy backing `ref` while this frame is in flight.
	 *
	 * A no-op when `ref` is not pool-owned (the swapchain or a test target carries no pool
	 * id). This folds the `if (ref.pool_id != INVALID) touch(ref.pool_id)` guard that every
	 * sampling consumer used to repeat into the pool itself, so a consumer reading a
	 * @ref veng::gpu::ImageRef cannot forget the touch that keeps its copy from being recycled out
	 * from under an in-flight read.
	 *
	 * @param ref An @ref veng::gpu::ImageRef whose pool copy should be retained.
	 */
	void consume(const gpu::ImageRef& ref) noexcept
	{
		if (ref.pool_id != gpu::ImageRef::INVALID_POOL_ID)
		{
			touch(ref.pool_id);
		}
	}

	/**
	 * @brief Retention-only consumer stamp for a buffer: keep the current copy of `id` alive this frame.
	 *
	 * Buffer analogue of @ref touch(ImageId). Stamps the current copy of `id` as used this
	 * frame so a cached producer's buffer is retained while an in-flight consumer still
	 * references it. A no-op for an id that has never been produced or is out of range.
	 *
	 * @param id Logical buffer id returned by @ref declare_buffer.
	 */
	void touch_buffer(BufferId id) noexcept;

	/**
	 * @brief Convenience overload: retain the pooled copy backing `ref` while this frame is in flight.
	 *
	 * Buffer analogue of @ref consume(const gpu::ImageRef&). A no-op when `ref` is not
	 * pool-owned. A consumer binding an SSBO from a `StorageBufferNode` whose producer may
	 * be cached this frame must call this — otherwise the pool can recycle or destroy the
	 * copy out from under the descriptor set still referencing it.
	 *
	 * @param ref A @ref veng::gpu::BufferRef whose pool copy should be retained.
	 */
	void consume(const gpu::BufferRef& ref) noexcept
	{
		if (ref.pool_id != gpu::BufferRef::INVALID_POOL_ID)
		{
			touch_buffer(ref.pool_id);
		}
	}

	/**
	 * @brief Convenience overload: retain the pooled copy backing a uniform `ref` while in flight.
	 *
	 * Uniform analogue of @ref consume(const gpu::BufferRef&) — uniform buffers are N-buffered in the
	 * same buffer pool, so a `GraphicsNode::add_uniform` consumer whose `UniformNode` producer may be
	 * cached this frame must call this, or the pool can recycle the copy out from under the descriptor
	 * set still referencing it. A no-op when `ref` is not pool-owned.
	 *
	 * @param ref A @ref veng::gpu::UniformRef whose pool copy should be retained.
	 */
	void consume(const gpu::UniformRef& ref) noexcept
	{
		if (ref.pool_id != gpu::UniformRef::INVALID_POOL_ID)
		{
			touch_buffer(ref.pool_id);
		}
	}

	/**
	 * @brief Insert a pipeline barrier to transition the current copy of `id` to `new_layout`.
	 *
	 * This is the engine's auto-barrier primitive. Nodes declare what layout, stage, and
	 * access they need via `GpuNode::image_usages`, and the executor calls this for them —
	 * no node records its own layout transitions for pool-backed images anymore. The barrier
	 * uses the prior usage's stage and access as the source scope.
	 *
	 * A no-op when the image is already in `new_layout`, has never been produced, the id is
	 * out of range, or `id` has no current copy.
	 *
	 * @param id         Logical image id returned by @ref declare_image.
	 * @param cmd        The command buffer to record the barrier into.
	 * @param new_layout The layout the image must be in before the next access.
	 * @param new_stage  Pipeline stage at which the next access occurs.
	 * @param new_access Access kind (read/write) of the next access.
	 */
	void transition_image(ImageId id, vk::CommandBuffer cmd, vk::ImageLayout new_layout,
						  vk::PipelineStageFlags2 new_stage, vk::AccessFlags2 new_access) noexcept;

	/**
	 * @brief Producer side for a transient buffer: obtain a copy to write into this frame.
	 *
	 * Reuses a retired copy or allocates a fresh persistently-mapped host-visible buffer.
	 * When `size` differs from the last acquire, existing copies are moved to an internal
	 * retirement graveyard rather than destroyed immediately, because an in-flight frame's
	 * descriptor set may still reference them; `purge_retired_buffers` releases them once
	 * their last-use stamp has aged past the retirement window.
	 *
	 * Consumers read the buffer through the handle the producer publishes, so there is no
	 * `read_buffer`; the copy is retained for the in-flight window from its write stamp.
	 *
	 * @param id   Logical buffer id returned by @ref declare_buffer.
	 * @param size Desired size in bytes; triggers a resize/reallocation if changed.
	 * @return A raw pointer to the physical @ref veng::Buffer on success, or a `vk::Result` error.
	 */
	[[nodiscard]] std::expected<Buffer*, vk::Result> acquire_buffer(BufferId id, vk::DeviceSize size);

	/**
	 * @brief Return the number of physical image copies currently held for a logical image.
	 *
	 * Test lens on the reuse and retention behaviour.
	 *
	 * @param id Logical image id returned by @ref declare_image.
	 * @return The number of physical copies, including any in-flight ones.
	 */
	[[nodiscard]] std::size_t image_copy_count(ImageId id) const noexcept;

	/**
	 * @brief Return the number of physical buffer copies currently held for a logical buffer.
	 * @param id Logical buffer id returned by @ref declare_buffer.
	 * @return The number of physical copies, including any in-flight ones.
	 */
	[[nodiscard]] std::size_t buffer_copy_count(BufferId id) const noexcept;

	/**
	 * @brief Return the number of buffer copies awaiting retirement after a resize.
	 *
	 * Test lens: buffer copies set aside by a resize are not freed immediately (an in-flight
	 * frame may still reference them); they are released in @ref acquire_buffer once their
	 * last-use stamp has aged past the retirement window.
	 *
	 * @return The count of buffers in the resize graveyard.
	 */
	[[nodiscard]] std::size_t retiring_buffer_count() const noexcept { return m_retiring_buffers.size(); }

	 private:
	template <class T>
	struct Copy
	{
		explicit Copy(T&& moved) noexcept
			: resource(std::move(moved))
		{
		}
		T			 resource;
		std::int64_t last_use = -1; // frame index that last wrote or read this copy
	};
	// An image copy also tracks its recorded layout + the producer's (stage, access) — the
	// executor's auto-barrier path reads these to compute the right transition.
	struct ImageCopy
	{
		explicit ImageCopy(Image&& moved) noexcept
			: resource(std::move(moved))
		{
		}
		Image					resource;
		std::int64_t			last_use	   = -1;
		vk::ImageLayout			current_layout = vk::ImageLayout::eUndefined;
		vk::PipelineStageFlags2 last_stage	   = vk::PipelineStageFlagBits2::eTopOfPipe;
		vk::AccessFlags2		last_access	   = vk::AccessFlagBits2::eNone;
	};
	struct ImageResource
	{
		vk::Format				format{};
		vk::ImageUsageFlags		usage{};
		vk::ImageAspectFlags	aspect{};
		vk::SampleCountFlagBits sample_count = vk::SampleCountFlagBits::e1;
		vk::Extent2D			extent{};
		// unique_ptr so a returned Image* stays valid when the copy list grows.
		std::vector<std::unique_ptr<ImageCopy>> copies{};
		std::size_t								current		= NONE;
		bool									is_constant = false; // immutable: one copy, never recycled
	};
	struct BufferResource
	{
		vk::BufferUsageFlags					   usage{};
		vk::DeviceSize							   size = 0;
		std::vector<std::unique_ptr<Copy<Buffer>>> copies{};
		std::size_t								   current = NONE;
	};

	static constexpr std::size_t NONE = ~static_cast<std::size_t>(0);

	void					   destroy() noexcept;
	void					   purge_retired_buffers() noexcept;
	[[nodiscard]] std::int64_t retired_through() const noexcept { return m_frame - m_frames_in_flight; }

	vk::Device					m_device;
	rhi::Device*				m_rhi = nullptr; ///< Registry threaded into the Image/Buffer copies this pool creates.
	vma::Allocator				m_allocator;
	std::int64_t				m_frames_in_flight;
	std::int64_t				m_frame = 0;
	std::vector<ImageResource>	m_images;
	std::vector<BufferResource> m_buffers;
	// Buffer copies a resize moved aside: an in-flight frame may still reference them, so they are
	// freed only once their last_use has retired (purged each begin_frame). A buffer resize is not
	// gated by a device-idle the way an image resize is, so we cannot drop them outright.
	std::vector<std::unique_ptr<Copy<Buffer>>> m_retiring_buffers;
};
} // namespace veng

#endif // VENG_RESOURCEPOOL_HPP
