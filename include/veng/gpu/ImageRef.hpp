/**
 * @file
 * @author chris
 * @brief The value that flows on an image edge of the reactive graph.
 *
 * `ImageRef` is a non-owning reference to a GPU image — handle, view, and the metadata a
 * consumer needs — never the pixels. A node that renders owns the backing `Image` and produces
 * an `ImageRef` to it; a downstream blit or present reads the ref and works the image on the
 * GPU. Passing a reference (not a copy) is what makes a pipeline such as
 * `GraphicsPipeline -> ImageData -> BlitNode -> SwapchainImageData -> PresentNode` mean
 * exactly what it reads as.
 *
 * Equality is value-based and includes a `version` the producing node bumps on every publish,
 * so two structurally identical refs published by consecutive producer runs still compare
 * unequal. This retires the old "deliberately non-comparable so re-produce always counts as
 * changed" hack — the property needed (a producer's re-run dirties consumers even when the
 * underlying handle was reused, e.g. the swapchain recycling images) is now explicit in the
 * type, not implicit in `ValueData<T>`'s non-comparable fallback.
 *
 * @ingroup gpu_handles
 */

#ifndef VENG_IMAGEREF_HPP
#define VENG_IMAGEREF_HPP

#include <cstdint>
#include <vulkan/vulkan.hpp>

namespace veng::gpu
{
/**
 * @brief Non-owning reference to a GPU image and its associated view/format/extent.
 *
 * Flows on image edges of the reactive graph. The producing node owns the backing `Image`;
 * consumers such as blit, present, or sampling draws bind the image through this ref.
 *
 * @ingroup gpu_handles
 * @see BufferRef
 * @see MeshRef
 * @see UniformRef
 * @see VersionedOutput
 */
struct ImageRef
{
	vk::Image	  image{};						   ///< The underlying Vulkan image.
	vk::ImageView view{};						   ///< Default view over the image.
	vk::Extent2D  extent{};						   ///< Pixel dimensions.
	vk::Format	  format = vk::Format::eUndefined; ///< Pixel format of the image.

	/// The @ref veng::ResourcePool resource id this ref's physical image belongs to, or `INVALID_POOL_ID`
	/// for an image the pool does not own (the swapchain, a test-owned target). A consumer that
	/// reads a pooled ref must call `pool.touch(pool_id)` so the copy it is sampling is retained
	/// while in flight (N-buffering).
	///
	/// Sync handles (acquire/render-finished semaphores and the slot's in-flight fence and the
	/// acquired image index) used to ride here too; they are now driver-owned and flow to sinks
	/// through @ref veng::gpu::SubmitContext instead, so the edge value stays purely a reference to the image
	/// with no per-frame queue plumbing leaking through graph edges.
	static constexpr std::uint32_t INVALID_POOL_ID = ~0U;
	std::uint32_t pool_id = INVALID_POOL_ID; ///< @ref veng::ResourcePool id, or `INVALID_POOL_ID` if unmanaged.

	/// Producer-bumped version. Incremented on every publish so two consecutive produces of the
	/// same underlying image handle (e.g. a recycled swapchain image) compare unequal and the
	/// change-cutoff in `ValueData<ImageRef>` fires correctly. Constant images never bump
	/// (one produce, one version).
	std::uint64_t version = 0;

	friend bool operator==(const ImageRef&, const ImageRef&) noexcept = default;
};
} // namespace veng::gpu

#endif // VENG_IMAGEREF_HPP
