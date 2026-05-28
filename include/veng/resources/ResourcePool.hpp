//
// L1.5 — the versioned, N-buffered transient resource pool (design.md §L1; the redesign in
// [[pass-draw-redesign]]). It is the engine's owner of per-frame render targets and transient
// buffers: nodes never own a `vk::Image`/`Buffer` for a target anymore — they declare a logical
// resource once and ask the pool for a physical copy each frame. The pool decides how many
// physical copies a logical resource needs and recycles them safely.
//
// Why a pool and not one image per node: with frames-in-flight > 1, frame N+1 is recorded on the
// CPU while frame N still executes on the GPU. If a producer re-renders into the same physical
// image every frame, it stomps pixels the GPU is still reading. The pool gives the producer a
// *different* physical copy whenever the previous one might still be in flight.
//
// The subtle case (a cached producer feeding a still-running consumer across a frame boundary) is
// why a copy is retained until every frame that *touched* it — write OR read — has retired. Both
// `acquire_image` (producer) and `read_image` (consumer) stamp the copy with the current frame;
// a copy is reusable only once that stamp is older than the in-flight window. So a copy a consumer
// keeps sampling (because its producer is cached) is never recycled out from under it.
//

#ifndef VENG_RESOURCEPOOL_HPP
#define VENG_RESOURCEPOOL_HPP

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <vector>
#include <veng/gpu/ImageRef.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/resources/Image.hpp>
#include <vulkan-memory-allocator-hpp/vk_mem_alloc.hpp>
#include <vulkan/vulkan.hpp>

namespace veng
{
class Context;

/// Opaque, stable handle to a logical resource declared on the pool. A node declares once
/// (lazily, on first record) and keeps the id; the physical copy behind it varies per frame.
using ImageId  = std::uint32_t;
using BufferId = std::uint32_t;

class ResourcePool
{
	 public:
	/// `frames_in_flight` is the retirement window: a copy stamped at frame F is reusable once
	/// the current frame reaches F + frames_in_flight (the driver guarantees frame F has retired
	/// by then, having waited that slot's fence in `acquire`).
	ResourcePool(vk::Device device, vma::Allocator allocator, std::size_t frames_in_flight) noexcept;

	ResourcePool(const ResourcePool&)			 = delete;
	ResourcePool& operator=(const ResourcePool&) = delete;
	ResourcePool(ResourcePool&& other) noexcept;
	ResourcePool& operator=(ResourcePool&& other) noexcept;
	~ResourcePool();

	/// Advance to a new frame. `frame_index` is monotonic; the caller must have ensured the
	/// frame `frame_index - frames_in_flight` has retired on the GPU before calling.
	void begin_frame(std::uint64_t frame_index) noexcept { m_frame = static_cast<std::int64_t>(frame_index); }

	/// Declare a transient image. The extent is supplied per-`acquire_image` (it tracks a
	/// screen-size edge); a change reallocates the copies (do it only when the device is idle).
	[[nodiscard]] ImageId declare_image(vk::Format format, vk::ImageUsageFlags usage,
										vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor);

	/// Declare a transient host-visible buffer (persistently mapped). Size is supplied per
	/// `acquire_buffer`.
	[[nodiscard]] BufferId declare_buffer(vk::BufferUsageFlags usage);

	/// Declare an *immutable* image initialized to a clear color via an immediate submit on the
	/// Context's graphics queue, transitioned to `SHADER_READ_ONLY_OPTIMAL`, and never recycled.
	/// Returns a `gpu::ImageRef` ready to be fed as a graph source — this is the clean
	/// replacement for hand-rolled command-pool + fence + clear sequences in user code (the leak
	/// the old `make_black_texture` opened up).
	[[nodiscard]] std::expected<gpu::ImageRef, vk::Result> constant_image(const Context& ctx, vk::Extent2D extent,
																		  vk::Format		   format,
																		  std::array<float, 4> clear);

	/// Producer side: a physical copy to write this frame. Reuses a retired copy or allocates a
	/// fresh one (and reallocates all copies if `extent` changed since last time). Marks the copy
	/// current for `read_image` and stamps its last-use to the current frame.
	[[nodiscard]] std::expected<Image*, vk::Result> acquire_image(ImageId id, vk::Extent2D extent);

	/// Consumer side: the copy holding the latest produced version, or nullptr if never produced.
	/// Stamps its last-use to the current frame so it is retained while this frame is in flight.
	[[nodiscard]] Image* read_image(ImageId id) noexcept;

	/// Retention-only consumer stamp: marks the current copy of `id` used this frame, without
	/// needing its `Image*` (the consumer already has the view via the published `ImageRef`). A
	/// no-op for an id never produced or out of range. This is what keeps a copy a cached
	/// producer's output is still being sampled from being recycled out from under the reader.
	void touch(ImageId id) noexcept;

	/// Producer side for a uniform/transient buffer: a copy to write this frame (reused/allocated
	/// like images). Consumers read the buffer through the handle the producer publishes, so there
	/// is no buffer `read_*`; the copy is retained for the in-flight window from its write stamp.
	[[nodiscard]] std::expected<Buffer*, vk::Result> acquire_buffer(BufferId id, vk::DeviceSize size);

	/// The number of physical copies a logical image currently holds (test lens on the
	/// reuse/retention behaviour).
	[[nodiscard]] std::size_t image_copy_count(ImageId id) const noexcept;
	[[nodiscard]] std::size_t buffer_copy_count(BufferId id) const noexcept;

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
	struct ImageResource
	{
		vk::Format			 format{};
		vk::ImageUsageFlags	 usage{};
		vk::ImageAspectFlags aspect{};
		vk::Extent2D		 extent{};
		// unique_ptr so a returned Image* stays valid when the copy list grows.
		std::vector<std::unique_ptr<Copy<Image>>> copies{};
		std::size_t								  current	  = NONE;
		bool									  is_constant = false; // immutable: one copy, never recycled
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
	[[nodiscard]] std::int64_t retired_through() const noexcept { return m_frame - m_frames_in_flight; }

	vk::Device					m_device;
	vma::Allocator				m_allocator;
	std::int64_t				m_frames_in_flight;
	std::int64_t				m_frame = 0;
	std::vector<ImageResource>	m_images;
	std::vector<BufferResource> m_buffers;
};
} // namespace veng

#endif // VENG_RESOURCEPOOL_HPP
