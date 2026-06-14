//
// L1.5 test: the shared RenderTargetSet + clamp_sample_count. No rendering — it checks the
// device-clamp helper and the invariants the set guarantees regardless of sample count: a
// configured set acquires a single-sample color image that downstream consumers read (the resolve
// target under MSAA), with depth presence following the configured depth format. The full MSAA
// resolve + beginRendering path is exercised end-to-end, under the validation layer, by the
// "PbrPass renders with MSAA" case in PbrPassTests.
//

#include <catch2/catch_test_macros.hpp>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/resources/RenderTargetSet.hpp>
#include <veng/resources/ResourcePool.hpp>

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("RenderTargetSet Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

constexpr veng::rhi::Format COLOR = veng::rhi::Format::RGBA8_UNORM;
constexpr veng::rhi::Format DEPTH = veng::rhi::Format::D32_SFLOAT;
constexpr vk::Extent2D		EXTENT{8, 8};

// Whether `bits` is exactly one set bit (a valid single sample-count flag).
bool is_single_bit(vk::SampleCountFlags bits)
{
	const auto v = static_cast<unsigned>(static_cast<vk::SampleCountFlags::MaskType>(bits));
	return v != 0 && (v & (v - 1)) == 0;
}
} // namespace

TEST_CASE("clamp_sample_count never exceeds the request or device support", "[resources][msaa]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	const vk::PhysicalDeviceLimits limits = ctx.physical_device().getProperties().limits;
	const vk::SampleCountFlags supported  = limits.framebufferColorSampleCounts & limits.framebufferDepthSampleCounts;

	// X1 is always valid and asking for it must stay X1 (never upgraded).
	REQUIRE(veng::clamp_sample_count(ctx, veng::rhi::SampleCount::X1) == veng::rhi::SampleCount::X1);

	// Asking for the maximum yields a single, device-supported bit that does not exceed the request.
	for (const veng::rhi::SampleCount req : {veng::rhi::SampleCount::X2, veng::rhi::SampleCount::X4,
											 veng::rhi::SampleCount::X8, veng::rhi::SampleCount::X64})
	{
		const veng::rhi::SampleCount got = veng::clamp_sample_count(ctx, req);
		REQUIRE(is_single_bit(veng::rhi::to_vk(got)));
		REQUIRE(static_cast<unsigned>(got) <= static_cast<unsigned>(req));			   // never upgrades past the request
		REQUIRE((got == veng::rhi::SampleCount::X1 || (supported & veng::rhi::to_vk(got)))); // a count the device offers
	}
}

TEST_CASE("RenderTargetSet acquires a single-sample color target with depth", "[resources][msaa]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			   ctx = make_context();
	veng::ResourcePool pool(ctx.device(), ctx.rhi(), ctx.allocator(), 1);

	veng::RenderTargetSet targets;
	targets.configure(COLOR, DEPTH, veng::rhi::SampleCount::X1);
	REQUIRE(targets.has_depth());
	REQUIRE_FALSE(targets.multisampled());

	pool.begin_frame(0);
	REQUIRE(targets.acquire(pool, veng::rhi::to_rhi(EXTENT)).has_value());
	REQUIRE(targets.color() != nullptr);
	// The published/consumed image is always single-sample, MSAA or not.
	REQUIRE(targets.color()->sample_count() == vk::SampleCountFlagBits::e1);
	REQUIRE(targets.color()->extent() == EXTENT);
}

TEST_CASE("RenderTargetSet without depth reports none", "[resources][msaa]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			   ctx = make_context();
	veng::ResourcePool pool(ctx.device(), ctx.rhi(), ctx.allocator(), 1);

	veng::RenderTargetSet targets;
	targets.configure(COLOR, veng::rhi::Format::UNDEFINED, veng::rhi::SampleCount::X1);
	REQUIRE_FALSE(targets.has_depth());

	pool.begin_frame(0);
	REQUIRE(targets.acquire(pool, veng::rhi::to_rhi(EXTENT)).has_value());
	REQUIRE(targets.color() != nullptr);
}

TEST_CASE("RenderTargetSet keeps a single-sample resolve target under MSAA", "[resources][msaa]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	const veng::rhi::SampleCount samples = veng::clamp_sample_count(ctx, veng::rhi::SampleCount::X4);
	if (samples == veng::rhi::SampleCount::X1)
	{
		SKIP("device does not support multisampled framebuffers");
	}

	veng::ResourcePool	  pool(ctx.device(), ctx.rhi(), ctx.allocator(), 1);
	veng::RenderTargetSet targets;
	targets.configure(COLOR, DEPTH, samples);
	REQUIRE(targets.multisampled());
	REQUIRE(targets.sample_count() == veng::rhi::to_vk(samples));

	pool.begin_frame(0);
	REQUIRE(targets.acquire(pool, veng::rhi::to_rhi(EXTENT)).has_value());
	// color() is the resolve target a consumer samples — single-sample even though the pass renders
	// into a `samples`-count attachment internally.
	REQUIRE(targets.color() != nullptr);
	REQUIRE(targets.color()->sample_count() == vk::SampleCountFlagBits::e1);
}
