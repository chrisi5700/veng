#include <catch2/catch_test_macros.hpp>
#include <veng/context/Context.hpp>
#include <veng/logging/Logger.hpp>



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
    
    auto indices = ctx.queue_indices();
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
    
    SECTION("device is discrete or integrated GPU")
    {
        bool is_gpu = props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu ||
                      props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
        REQUIRE(is_gpu);
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

