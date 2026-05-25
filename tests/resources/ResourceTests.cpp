//
// L1 resource tests (design.md §L1). Allocates real Buffer/Image objects via VMA on a
// headless device and checks creation, accessors, and move-only RAII (no double-free,
// verified under the llm-vcpkg ASan/UBSan gate).
//

#include <catch2/catch_test_macros.hpp>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/resources/Image.hpp>

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
