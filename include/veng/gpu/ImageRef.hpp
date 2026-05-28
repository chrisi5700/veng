//
// Created by chris on 5/26/26.
//
// The value that flows on an image edge of the reactive graph (design.md §L4). It is a
// non-owning *reference* to a GPU image — handle + view + the metadata a consumer needs
// — never the pixels. A node that renders owns the backing `Image` and produces an
// `ImageRef` to it; a downstream blit/present reads the ref and works the image on the
// GPU. Passing a reference (not a copy) is what lets `GraphicsPipeline -> ImageData ->
// BlitNode -> SwapchainImageData -> PresentNode` mean exactly what it reads as.
//
// Equality is *value-based* and includes a `version` the producing node bumps on every
// `produce`, so two structurally identical refs published by consecutive producer runs still
// compare unequal. This retires the old "deliberately non-comparable so re-produce always
// counts as changed" hack — the property we needed (a producer's re-run dirties consumers
// even when the underlying handle was reused, e.g. the swapchain recycling images) is now
// explicit in the type, not implicit in `ValueData<T>`'s non-comparable fallback.
//

#ifndef VENG_IMAGEREF_HPP
#define VENG_IMAGEREF_HPP

#include <cstdint>
#include <vulkan/vulkan.hpp>

namespace veng::gpu
{
struct ImageRef
{
	vk::Image	  image{};
	vk::ImageView view{};
	vk::Extent2D  extent{};
	vk::Format	  format = vk::Format::eUndefined;

	// The ResourcePool resource this ref's physical image is a copy of, or INVALID for an
	// image the pool does not own (the swapchain, a test-owned target). A consumer that reads a
	// pooled ref must `pool.touch(pool_id)` so the copy it is sampling is retained while in
	// flight (N-buffering); see [[pass-draw-redesign]] and ResourcePool.
	//
	// Sync handles (the acquire / render-finished semaphores + the slot's in-flight fence + the
	// acquired image index) used to ride here too; they are now driver/`FrameExecutor`-owned
	// and flow to sinks through `gpu::SubmitContext` instead, so the edge value stays purely a
	// reference to *the image* — no per-frame queue plumbing leaks through graph edges.
	static constexpr std::uint32_t INVALID_POOL_ID = ~0U;
	std::uint32_t				   pool_id		   = INVALID_POOL_ID;

	// Producer-bumped version: incremented on every `produce` so two structurally identical
	// refs published by consecutive producer runs compare unequal (and `ValueData<ImageRef>`'s
	// change-cutoff fires). Constant images never bump (one produce, one version).
	std::uint64_t version = 0;

	friend bool operator==(const ImageRef&, const ImageRef&) noexcept = default;
};
} // namespace veng::gpu

#endif // VENG_IMAGEREF_HPP
