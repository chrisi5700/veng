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
	// 5 single-set allocations across pools of 2 => at least 3 pools were created.
	REQUIRE(allocator.pool_count() >= 3);

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
