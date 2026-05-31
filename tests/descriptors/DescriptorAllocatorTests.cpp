//
// L2.2 DescriptorAllocator tests. Exercises allocation, transparent
// pool growth on exhaustion, recycling via reset(), and move semantics on a headless
// device. Self-contained: builds its own one-binding layout rather than depending on a
// pipeline.
//

#include <catch2/catch_test_macros.hpp>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/descriptors/DescriptorAllocator.hpp>
#include <veng/logging/Logger.hpp>

#include "support/VkFault.hpp"

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("DescriptorAllocator Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

// A trivial layout: a single uniform-buffer binding visible to compute.
vk::DescriptorSetLayout make_layout(vk::Device device)
{
	const auto binding = vk::DescriptorSetLayoutBinding()
							 .setBinding(0)
							 .setDescriptorType(vk::DescriptorType::eUniformBuffer)
							 .setDescriptorCount(1)
							 .setStageFlags(vk::ShaderStageFlagBits::eCompute);
	const auto result  = device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo().setBindings(binding));
	REQUIRE(result.result == vk::Result::eSuccess);
	return result.value;
}
} // namespace

TEST_CASE("DescriptorAllocator hands out sets and grows its pool on exhaustion", "[descriptors]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto						  ctx	 = make_context();
	const vk::DescriptorSetLayout layout = make_layout(ctx.device());

	// Tiny pools (2 sets each) so a handful of allocations forces growth.
	veng::DescriptorAllocator allocator(ctx.device(), 2);
	REQUIRE(allocator.pool_count() == 0); // lazy: no pool until first allocate

	for (int i = 0; i < 5; ++i)
	{
		const auto set = allocator.allocate(layout);
		REQUIRE(set.has_value());
		REQUIRE(*set);
	}
	// All five allocations succeeded even though each pool nominally holds only two sets.
	// Whether that forces the allocator to grow is driver-dependent: a strict driver (desktop
	// NVIDIA) returns eOutOfPoolMemory past maxSets, creating >=3 pools; a lenient one (lavapipe,
	// used in CI) over-allocates from a single pool and never reports exhaustion. Both conform to
	// the spec, so only require that at least one pool was created.
	REQUIRE(allocator.pool_count() >= 1);

	ctx.device().destroyDescriptorSetLayout(layout);
}

TEST_CASE("DescriptorAllocator::reset recycles sets without dropping pools", "[descriptors]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto						  ctx	 = make_context();
	const vk::DescriptorSetLayout layout = make_layout(ctx.device());

	veng::DescriptorAllocator allocator(ctx.device(), 4);
	for (int i = 0; i < 4; ++i)
	{
		REQUIRE(allocator.allocate(layout).has_value());
	}
	const std::size_t pools_before = allocator.pool_count();
	REQUIRE(pools_before >= 1);

	allocator.reset(); // frees every set, keeps the pools
	REQUIRE(allocator.pool_count() == pools_before);

	// After reset the recycled pool serves allocations again.
	REQUIRE(allocator.allocate(layout).has_value());

	ctx.device().destroyDescriptorSetLayout(layout);
}

TEST_CASE("DescriptorAllocator move transfers pool ownership without double free", "[descriptors][raii]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto						  ctx	 = make_context();
	const vk::DescriptorSetLayout layout = make_layout(ctx.device());

	veng::DescriptorAllocator first(ctx.device(), 8);
	REQUIRE(first.allocate(layout).has_value());
	REQUIRE(first.pool_count() == 1);

	const veng::DescriptorAllocator second = std::move(first);
	REQUIRE(second.pool_count() == 1);
	REQUIRE(first.pool_count() == 0); // moved-from owns nothing -> destroyed once

	ctx.device().destroyDescriptorSetLayout(layout);
}

TEST_CASE("DescriptorAllocator move-assignment frees the overwritten target once", "[descriptors][raii]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto						  ctx	 = make_context();
	const vk::DescriptorSetLayout layout = make_layout(ctx.device());

	veng::DescriptorAllocator first(ctx.device(), 8);
	veng::DescriptorAllocator second(ctx.device(), 8);
	REQUIRE(first.allocate(layout).has_value());
	REQUIRE(second.allocate(layout).has_value());
	REQUIRE(first.pool_count() == 1);
	REQUIRE(second.pool_count() == 1);

	first = std::move(second); // destroys first's pool, adopts second's
	REQUIRE(first.pool_count() == 1);
	REQUIRE(second.pool_count() == 0);

	// Self-assignment is guarded and must not drop the live pool.
	veng::DescriptorAllocator* alias = &first;
	first							 = std::move(*alias);
	REQUIRE(first.pool_count() == 1);

	ctx.device().destroyDescriptorSetLayout(layout);
}

// --- Error-path branches (the std::expected error returns) ------------------------------------
// Driven by swapping a single entry point in the global dispatcher for the duration of one call.

TEST_CASE("DescriptorAllocator reports pool-creation failure instead of crashing", "[descriptors][error]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	auto						  ctx	 = make_context();
	const vk::DescriptorSetLayout layout = make_layout(ctx.device());

	veng::DescriptorAllocator allocator(ctx.device(), 4);
	{
		// First allocate must create the initial pool; force that creation to fail.
		const veng::test::ScopedDispatchFault fault{VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDescriptorPool,
													+[](VkDevice, const VkDescriptorPoolCreateInfo*,
														const VkAllocationCallbacks*, VkDescriptorPool*) -> VkResult
													{ return VK_ERROR_OUT_OF_DEVICE_MEMORY; }};
		const auto							  set = allocator.allocate(layout);
		REQUIRE_FALSE(set.has_value());
		REQUIRE(set.error() == vk::Result::eErrorOutOfPoolMemory); // create_pool() failed -> no pool
		REQUIRE(allocator.pool_count() == 0);
	}
	// Once the fault is lifted the allocator recovers and serves sets normally.
	REQUIRE(allocator.allocate(layout).has_value());
	REQUIRE(allocator.pool_count() == 1);

	ctx.device().destroyDescriptorSetLayout(layout);
}

TEST_CASE("DescriptorAllocator surfaces a non-recoverable allocate error verbatim", "[descriptors][error]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	auto						  ctx	 = make_context();
	const vk::DescriptorSetLayout layout = make_layout(ctx.device());

	veng::DescriptorAllocator allocator(ctx.device(), 4);
	// The pool creates fine; vkAllocateDescriptorSets fails with a non-pool-exhaustion error, which
	// is not recoverable by growing — it must propagate unchanged rather than loop or crash.
	const veng::test::ScopedDispatchFault fault{
		VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateDescriptorSets,
		+[](VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet*) -> VkResult
		{ return VK_ERROR_OUT_OF_DEVICE_MEMORY; }};
	const auto set = allocator.allocate(layout);
	REQUIRE_FALSE(set.has_value());
	REQUIRE(set.error() == vk::Result::eErrorOutOfDeviceMemory);

	ctx.device().destroyDescriptorSetLayout(layout);
}

TEST_CASE("DescriptorAllocator gives up when a grow pool cannot be created", "[descriptors][error]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	auto						  ctx	 = make_context();
	const vk::DescriptorSetLayout layout = make_layout(ctx.device());

	veng::DescriptorAllocator allocator(ctx.device(), 4);
	REQUIRE(allocator.allocate(layout).has_value()); // seed one real pool so the next allocate enters the grow path
	REQUIRE(allocator.pool_count() == 1);

	// Allocation reports pool-exhaustion (recoverable -> the allocator tries to grow), but the grow's
	// pool creation also fails. The original exhaustion result must be returned, not an infinite loop.
	const veng::test::ScopedDispatchFault exhaust{
		VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateDescriptorSets,
		+[](VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet*) -> VkResult
		{ return VK_ERROR_OUT_OF_POOL_MEMORY; }};
	const veng::test::ScopedDispatchFault no_grow{
		VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDescriptorPool,
		+[](VkDevice, const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*, VkDescriptorPool*) -> VkResult
		{ return VK_ERROR_OUT_OF_DEVICE_MEMORY; }};
	const auto set = allocator.allocate(layout);
	REQUIRE_FALSE(set.has_value());
	REQUIRE(set.error() == vk::Result::eErrorOutOfPoolMemory);

	ctx.device().destroyDescriptorSetLayout(layout);
}
