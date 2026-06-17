#include <array>
#include <catch2/catch_test_macros.hpp>
#include <variant>
#include <veng/context/Context.hpp>
#include <veng/logging/Logger.hpp>

#include "support/VkFault.hpp"

TEST_CASE("VulkanContext creation", "[vulkan]")
{
	veng::Logger::instance().set_level(spdlog::level::trace);
	SECTION("constructs without throwing")
	{
		REQUIRE(veng::Context::create("Test App"));
	}
}

TEST_CASE("VulkanContext provides valid handles", "[vulkan]")
{
	auto ctx_res = veng::Context::create("Test App");
	REQUIRE(ctx_res);
	auto ctx = std::move(*ctx_res);

	SECTION("instance is valid")
	{
		REQUIRE(ctx.instance());
	}

	SECTION("physical device is valid")
	{
		REQUIRE(ctx.physical_device());
	}

	SECTION("device is valid")
	{
		REQUIRE(ctx.device());
	}

	SECTION("graphics queue is valid")
	{
		REQUIRE(ctx.graphics_queue());
	}

	SECTION("compute queue is valid")
	{
		REQUIRE(ctx.compute_queue());
	}
}

TEST_CASE("VulkanContext queue indices are reasonable", "[vulkan]")
{
	auto ctx_res = veng::Context::create("Test App");
	REQUIRE(ctx_res);
	auto ctx = std::move(*ctx_res);

	auto indices		= ctx.queue_indices();
	auto queue_families = ctx.physical_device().getQueueFamilyProperties();

	SECTION("graphics index is within bounds")
	{
		REQUIRE(indices.graphics < queue_families.size());
	}

	SECTION("compute index is within bounds")
	{
		REQUIRE(indices.compute < queue_families.size());
	}

	SECTION("graphics queue supports graphics")
	{
		auto flags = queue_families[indices.graphics].queueFlags;
		REQUIRE((flags & vk::QueueFlagBits::eGraphics));
	}

	SECTION("compute queue supports compute")
	{
		auto flags = queue_families[indices.compute].queueFlags;
		REQUIRE((flags & vk::QueueFlagBits::eCompute));
	}
}

TEST_CASE("VulkanContext physical device properties", "[vulkan]")
{
	auto ctx_res = veng::Context::create("Test App");
	REQUIRE(ctx_res);
	auto ctx = std::move(*ctx_res);

	auto props = ctx.physical_device().getProperties();

	SECTION("device has a name")
	{
		REQUIRE(std::string_view(props.deviceName.data()).size() > 0);
		INFO("Device: " << props.deviceName.data());
	}

	SECTION("device type is a usable renderer")
	{
		// Accept software/CPU devices (e.g. lavapipe in CI) alongside real GPUs — the engine
		// renders correctly on all of them; only an unknown/eOther type would be suspect.
		bool usable = props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu ||
					  props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu ||
					  props.deviceType == vk::PhysicalDeviceType::eVirtualGpu ||
					  props.deviceType == vk::PhysicalDeviceType::eCpu;
		REQUIRE(usable);
	}

	SECTION("API version is at least 1.3")
	{
		uint32_t major = VK_VERSION_MAJOR(props.apiVersion);
		uint32_t minor = VK_VERSION_MINOR(props.apiVersion);

		bool sufficient = (major > 1) || (major == 1 && minor >= 3);
		REQUIRE(sufficient);
		INFO("API Version: " << major << "." << minor);
	}
}

// --- immediate_submit -------------------------------------------------------------------------
// The engine's synchronous "do this on the GPU now" primitive. Its happy path and every early-out
// error branch (pool / allocate / begin / end / fence creation, and a failed submit) were untested.

TEST_CASE("immediate_submit records, submits, and waits", "[vulkan][immediate_submit]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx_res = veng::Context::create("Immediate Submit Test");
	REQUIRE(ctx_res);
	const auto ctx = std::move(*ctx_res);

	bool recorded = false;
	// A no-op recording is enough: the value under test is the pool/allocate/begin/end/submit/wait
	// sequence around the callback, not the recorded commands.
	const vk::Result result = ctx.immediate_submit(
		[&](vk::CommandBuffer cmd)
		{
			REQUIRE(cmd);
			recorded = true;
		});
	REQUIRE(result == vk::Result::eSuccess);
	REQUIRE(recorded);
}

TEST_CASE("immediate_submit surfaces each setup failure", "[vulkan][immediate_submit][error]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	auto ctx_res = veng::Context::create("Immediate Submit Test");
	REQUIRE(ctx_res);
	const auto ctx = std::move(*ctx_res);

	const auto submit_noop = [&] { return ctx.immediate_submit([](vk::CommandBuffer) {}); };

	SECTION("command-pool creation fails")
	{
		const veng::test::ScopedDispatchFault fault{
			VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateCommandPool,
			+[](VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*, VkCommandPool*) -> VkResult
			{ return VK_ERROR_OUT_OF_DEVICE_MEMORY; }};
		REQUIRE(submit_noop() == vk::Result::eErrorOutOfDeviceMemory);
	}
	SECTION("command-buffer allocation fails")
	{
		const veng::test::ScopedDispatchFault fault{
			VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateCommandBuffers,
			+[](VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer*) -> VkResult
			{ return VK_ERROR_OUT_OF_DEVICE_MEMORY; }};
		REQUIRE(submit_noop() == vk::Result::eErrorOutOfDeviceMemory);
	}
	SECTION("a failed submit skips the wait and returns the error")
	{
		const veng::test::ScopedDispatchFault fault{VULKAN_HPP_DEFAULT_DISPATCHER.vkQueueSubmit,
													+[](VkQueue, uint32_t, const VkSubmitInfo*, VkFence) -> VkResult
													{ return VK_ERROR_DEVICE_LOST; }};
		REQUIRE(submit_noop() == vk::Result::eErrorDeviceLost);
	}
}

// --- Context::create failure modes ------------------------------------------------------------
// A user on a misconfigured driver / unsupported device must get a typed error back, not a silent
// crash. These drive the two input-reachable failure paths and assert the right variant alternative.
// (The internal steps run through vk-bootstrap, which loads its own entry points, and create()
// re-inits the vulkan-hpp dispatcher — so the dispatcher fault seam can't reach inside it; these
// failures are induced through the public inputs instead.)

TEST_CASE("Context::create reports a missing instance extension, not a crash", "[vulkan][error]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	static constexpr std::array<const char*, 1> bogus{"VK_EXT_this_extension_does_not_exist"};
	const auto									ctx = veng::Context::create("Bad Extension", bogus);
	REQUIRE_FALSE(ctx.has_value());
	REQUIRE(std::holds_alternative<InstanceCreationError>(ctx.error()));
}

TEST_CASE("Context::create reports a null surface from the factory, not a crash", "[vulkan][error]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	static constexpr std::array<const char*, 1> surface_ext{VK_KHR_SURFACE_EXTENSION_NAME};
	// A surface factory that fails (returns a null handle) must surface a typed SurfaceCreationError.
	const auto ctx = veng::Context::create("Null Surface", surface_ext, [](VkInstance) { return VkSurfaceKHR{}; });
	REQUIRE_FALSE(ctx.has_value());
	REQUIRE(std::holds_alternative<SurfaceCreationError>(ctx.error()));
}

// --- Context::adopt -----------------------------------------------------------------------------
// adopt() wraps a host-owned instance/device (the Qt / embedder path). It must (1) produce a working
// context on the *borrowed* device and (2) free only what it created — never the adopted handles. We
// stand in for the host with a veng-created Context, adopt its handles, exercise the adopted context,
// then destroy it and prove the original is still usable. A teardown of the shared instance/device
// would be a double-free / use-after-free the gate's ASan would flag.
TEST_CASE("Context::adopt borrows a host device without owning it", "[vulkan][adopt]")
{
	veng::Logger::instance().set_level(spdlog::level::err);

	auto host_res = veng::Context::create("Adopt Host");
	REQUIRE(host_res.has_value());
	veng::Context& host = host_res.value();

	{
		auto adopted_res = veng::Context::adopt(host.instance(), host.physical_device(), host.device(),
												host.graphics_queue(), host.queue_indices().graphics);
		REQUIRE(adopted_res.has_value());
		veng::Context& adopted = adopted_res.value();

		// Shares the host device, but owns its own allocator + rhi registry: allocate a texture
		// through it to prove both are live on the borrowed device.
		auto tex = adopted.rhi().create_texture({.extent = {16, 16},
												 .format = veng::rhi::Format::RGBA8_UNORM,
												 .usage	 = veng::rhi::TextureUsageFlags::COLOR_ATTACHMENT});
		REQUIRE(tex.has_value());
		adopted.rhi().destroy_texture(tex.value());

		// And a trivial submit on the borrowed graphics queue.
		REQUIRE(adopted.immediate_submit([](vk::CommandBuffer) {}) == vk::Result::eSuccess);
	} // adopted destructs here: frees only its allocator/rhi, leaving the host instance/device intact.

	// The host device survived the adopted context's teardown — still usable.
	REQUIRE(host.immediate_submit([](vk::CommandBuffer) {}) == vk::Result::eSuccess);
}
