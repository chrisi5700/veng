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
// Deliberately NOT equality-comparable: `ValueData<T>` falls back to "always changed"
// for non-comparable T (Data.hpp), which is the correct, conservative behaviour for a
// GPU resource written by reference — the handle can repeat across frames (the swapchain
// recycles a small set of images) even though the contents were just overwritten, so we
// must never let a repeated handle be mistaken for "unchanged" and cached away.
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

	// Swapchain images only (0 / null for an offscreen target). `index` is the acquired
	// image index `present` needs; the two semaphores are this frame's present handoff,
	// carried as data so the PresentNode is fully driven by the graph (no per-frame
	// setter): the submit waits on `acquire_wait` and signals `present_signal`, which the
	// presentation engine then waits on.
	std::uint32_t index = 0;
	vk::Semaphore acquire_wait{};	// image_available
	vk::Semaphore present_signal{}; // render_finished

	// No operator== on purpose — see the file header.
};
} // namespace veng::gpu

#endif // VENG_IMAGEREF_HPP
