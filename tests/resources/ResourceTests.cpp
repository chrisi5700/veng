//
// L1 resource tests. Allocates real Buffer/Image objects via VMA on a
// headless device and checks creation, accessors, and move-only RAII (no double-free,
// verified under the llm-vcpkg ASan/UBSan gate).
//

#include <catch2/catch_test_macros.hpp>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/resources/Image.hpp>

#include "support/VkFault.hpp"

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("Resource Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}
} // namespace

TEST_CASE("Buffer allocates and reports its size", "[resources][buffer]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	auto buffer = veng::Buffer::create(ctx.allocator(), 256, vk::BufferUsageFlagBits::eStorageBuffer);
	REQUIRE(buffer.has_value());
	REQUIRE(buffer->buffer());
	REQUIRE(buffer->allocation() != nullptr);
	REQUIRE(buffer->size() == 256);
}

TEST_CASE("Buffer move transfers ownership without double free", "[resources][buffer][raii]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx	= make_context();
	auto buffer = veng::Buffer::create(ctx.allocator(), 64, vk::BufferUsageFlagBits::eUniformBuffer);
	REQUIRE(buffer.has_value());

	const vk::Buffer   raw	 = buffer->buffer();
	const veng::Buffer moved = std::move(*buffer);
	REQUIRE(moved.buffer() == raw);
	REQUIRE_FALSE(buffer->buffer()); // moved-from emptied -> freed once
}

TEST_CASE("Image allocates a 2D color target with a matching view", "[resources][image]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	auto image = veng::Image::create(ctx.allocator(), ctx.device(), vk::Extent2D{128, 128}, vk::Format::eR8G8B8A8Unorm,
									 vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc);
	REQUIRE(image.has_value());
	REQUIRE(image->image());
	REQUIRE(image->view());
	REQUIRE(image->format() == vk::Format::eR8G8B8A8Unorm);
	REQUIRE(image->extent().width == 128);
	REQUIRE(image->extent().height == 128);
}

TEST_CASE("Image move transfers ownership without double free", "[resources][image][raii]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx   = make_context();
	auto image = veng::Image::create(ctx.allocator(), ctx.device(), vk::Extent2D{64, 64}, vk::Format::eR8G8B8A8Unorm,
									 vk::ImageUsageFlagBits::eColorAttachment);
	REQUIRE(image.has_value());

	const vk::Image	  raw	= image->image();
	const veng::Image moved = std::move(*image);
	REQUIRE(moved.image() == raw);
	REQUIRE_FALSE(image->image()); // moved-from emptied -> freed once
}

// --- Allocation-failure branches (the std::expected error returns) ----------------------------
// Driven through a FailingAllocator: a real allocator with one VMA entry point forced to fail.

TEST_CASE("Buffer::create surfaces an allocation failure as an error", "[resources][buffer][error]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	auto							   ctx = make_context();
	const veng::test::FailingAllocator failing(ctx, veng::test::FailingAllocator::Fail::CreateBuffer);
	REQUIRE(failing.get()); // the allocator itself built fine; only buffer creation will fail

	const auto buffer = veng::Buffer::create(failing.get(), 256, vk::BufferUsageFlagBits::eStorageBuffer);
	REQUIRE_FALSE(buffer.has_value());
	REQUIRE(buffer.error() == vk::Result::eErrorOutOfDeviceMemory);
}

TEST_CASE("Image::create surfaces an allocation failure as an error", "[resources][image][error]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	auto							   ctx = make_context();
	const veng::test::FailingAllocator failing(ctx, veng::test::FailingAllocator::Fail::CreateImage);
	REQUIRE(failing.get());

	const auto image = veng::Image::create(failing.get(), ctx.device(), vk::Extent2D{64, 64},
										   vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eColorAttachment);
	REQUIRE_FALSE(image.has_value());
	REQUIRE(image.error() == vk::Result::eErrorOutOfDeviceMemory);
}

TEST_CASE("Image::create frees the image when view creation fails (no leak)", "[resources][image][error][raii]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	auto ctx = make_context();

	// The image allocates successfully (real allocator), but vkCreateImageView is forced to fail —
	// exercising the partial-construction cleanup that destroys the image before returning the error.
	// ASan/UBSan under the gate proves the image was freed, not leaked.
	const auto image = [&]
	{
		const veng::test::ScopedDispatchFault fault{
			VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateImageView,
			+[](VkDevice, const VkImageViewCreateInfo*, const VkAllocationCallbacks*, VkImageView*) -> VkResult
			{ return VK_ERROR_OUT_OF_DEVICE_MEMORY; }};
		return veng::Image::create(ctx.allocator(), ctx.device(), vk::Extent2D{64, 64}, vk::Format::eR8G8B8A8Unorm,
								   vk::ImageUsageFlagBits::eColorAttachment);
	}();
	REQUIRE_FALSE(image.has_value());
	REQUIRE(image.error() == vk::Result::eErrorOutOfDeviceMemory);
}

// --- Branches the happy-path tests skip -------------------------------------------------------

TEST_CASE("Image with a transfer-only usage gets no view", "[resources][image]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	// eTransferDst is not a view-capable usage, so Image::create skips view creation and view() is null.
	auto image = veng::Image::create(ctx.allocator(), ctx.device(), vk::Extent2D{32, 32}, vk::Format::eR8G8B8A8Unorm,
									 vk::ImageUsageFlagBits::eTransferDst);
	REQUIRE(image.has_value());
	REQUIRE(image->image());
	REQUIRE_FALSE(image->view()); // transfer-only target carries no view
}

TEST_CASE("Image clamps a zero mip-level request to one", "[resources][image]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	auto image = veng::Image::create(ctx.allocator(), ctx.device(), vk::Extent2D{16, 16}, vk::Format::eR8G8B8A8Unorm,
									 vk::ImageUsageFlagBits::eColorAttachment, vk::ImageAspectFlagBits::eColor, 0);
	REQUIRE(image.has_value());
	REQUIRE(image->mip_levels() == 1); // 0 was normalised to a single level
}

TEST_CASE("Buffer reports a mapped pointer only when created mapped", "[resources][buffer]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	SECTION("host-mapped staging buffer is persistently mapped")
	{
		auto mapped = veng::Buffer::create(
			ctx.allocator(), 128, vk::BufferUsageFlagBits::eTransferSrc, vma::MemoryUsage::eAutoPreferHost,
			vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessSequentialWrite);
		REQUIRE(mapped.has_value());
		REQUIRE(mapped->mapped() != nullptr);
	}
	SECTION("device-local buffer is not mapped")
	{
		auto device_local = veng::Buffer::create(ctx.allocator(), 128, vk::BufferUsageFlagBits::eStorageBuffer);
		REQUIRE(device_local.has_value());
		REQUIRE(device_local->mapped() == nullptr);
	}
}

// --- Move-assignment RAII (only the move *constructor* was covered before) ---------------------

TEST_CASE("Buffer move-assignment frees the overwritten target once", "[resources][buffer][raii]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	auto first	= veng::Buffer::create(ctx.allocator(), 64, vk::BufferUsageFlagBits::eUniformBuffer);
	auto second = veng::Buffer::create(ctx.allocator(), 128, vk::BufferUsageFlagBits::eStorageBuffer);
	REQUIRE(first.has_value());
	REQUIRE(second.has_value());

	const vk::Buffer second_raw = second->buffer();
	*first						= std::move(*second); // destroys first's original buffer, adopts second's
	REQUIRE(first->buffer() == second_raw);
	REQUIRE(first->size() == 128);
	REQUIRE_FALSE(second->buffer()); // moved-from emptied -> the adopted buffer is freed exactly once

	// Self-assignment is a no-op (guarded by `this != &other`) and must not free the live buffer.
	veng::Buffer* alias = &*first;
	*first				= std::move(*alias);
	REQUIRE(first->buffer() == second_raw);
}

TEST_CASE("Image move-assignment frees the overwritten target once", "[resources][image][raii]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	auto first	= veng::Image::create(ctx.allocator(), ctx.device(), vk::Extent2D{32, 32}, vk::Format::eR8G8B8A8Unorm,
									  vk::ImageUsageFlagBits::eColorAttachment);
	auto second = veng::Image::create(ctx.allocator(), ctx.device(), vk::Extent2D{48, 48}, vk::Format::eR8G8B8A8Unorm,
									  vk::ImageUsageFlagBits::eColorAttachment);
	REQUIRE(first.has_value());
	REQUIRE(second.has_value());

	const vk::Image second_raw = second->image();
	*first					   = std::move(*second); // destroys first's original image+view, adopts second's
	REQUIRE(first->image() == second_raw);
	REQUIRE(first->extent().width == 48);
	REQUIRE_FALSE(second->image()); // moved-from emptied -> freed exactly once
}
