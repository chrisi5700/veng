//
// L1.5 test: the versioned, N-buffered ResourcePool's reuse + retention logic (the core of
// N-buffering correctness). No rendering — it allocates tiny images/buffers and checks, purely
// from copy counts and physical-copy identity, that: (1) single-buffered reuses one copy every
// frame; (2) a producer writing every frame at N-in-flight settles at N copies; (3) a copy a
// consumer keeps reading (because its producer is cached) is NOT recycled out from under it when
// the producer resumes — both reads and writes stamp a copy, so it is retained for the in-flight
// window. (3) is the case that makes a cached-producer / running-consumer frame boundary safe.
//

#include <catch2/catch_test_macros.hpp>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/resources/ResourcePool.hpp>

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("ResourcePool Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

constexpr vk::Format		  FORMAT = vk::Format::eR8G8B8A8Unorm;
constexpr vk::ImageUsageFlags USAGE	 = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc;
constexpr vk::Extent2D		  EXTENT{4, 4};
} // namespace

TEST_CASE("single-buffered ResourcePool reuses one copy every frame", "[resources][pool]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto				ctx = make_context();
	veng::ResourcePool	pool(ctx.device(), ctx.rhi(), ctx.allocator(), 1);
	const veng::ImageId id = pool.declare_image(FORMAT, USAGE);

	pool.begin_frame(0);
	auto a0 = pool.acquire_image(id, EXTENT);
	REQUIRE(a0.has_value());

	pool.begin_frame(1);
	auto a1 = pool.acquire_image(id, EXTENT);
	REQUIRE(a1.has_value());

	// Frame 0 retired before frame 1 (single-buffered), so its copy is reused — one copy total.
	REQUIRE(pool.image_copy_count(id) == 1);
	REQUIRE(a0.value() == a1.value());
}

TEST_CASE("a producer writing every frame settles at frames_in_flight copies", "[resources][pool]")
{
	auto				ctx = make_context();
	veng::ResourcePool	pool(ctx.device(), ctx.rhi(), ctx.allocator(), 2);
	const veng::ImageId id = pool.declare_image(FORMAT, USAGE);

	for (std::uint64_t frame = 0; frame < 6; ++frame)
	{
		pool.begin_frame(frame);
		REQUIRE(pool.acquire_image(id, EXTENT).has_value());
	}
	// At 2 in flight, a copy written at frame F is reusable at F+2, so the producer cycles
	// between exactly two physical copies.
	REQUIRE(pool.image_copy_count(id) == 2);
}

TEST_CASE("a copy a consumer keeps reading is not recycled when the producer resumes", "[resources][pool]")
{
	auto				ctx = make_context();
	veng::ResourcePool	pool(ctx.device(), ctx.rhi(), ctx.allocator(), 2);
	const veng::ImageId id = pool.declare_image(FORMAT, USAGE);

	// Frame 0: the producer writes copy A; the consumer reads it.
	pool.begin_frame(0);
	auto produced = pool.acquire_image(id, EXTENT);
	REQUIRE(produced.has_value());
	veng::Image* const a = produced.value();
	REQUIRE(pool.read_image(id) == a);

	// Frames 1..2: the producer is CACHED (no acquire); the consumer keeps sampling A. Each read
	// re-stamps A so it stays alive across these in-flight frames.
	pool.begin_frame(1);
	REQUIRE(pool.read_image(id) == a);
	pool.begin_frame(2);
	REQUIRE(pool.read_image(id) == a);

	// Frame 3: the producer resumes. A was read as recently as frame 2 (still inside the 2-frame
	// in-flight window), so it must NOT be recycled — the producer gets a fresh copy B != A.
	pool.begin_frame(3);
	auto resumed = pool.acquire_image(id, EXTENT);
	REQUIRE(resumed.has_value());
	REQUIRE(resumed.value() != a);
	REQUIRE(pool.image_copy_count(id) == 2);
}

TEST_CASE("a resize reallocates a logical image's copies", "[resources][pool]")
{
	auto				ctx = make_context();
	veng::ResourcePool	pool(ctx.device(), ctx.rhi(), ctx.allocator(), 2);
	const veng::ImageId id = pool.declare_image(FORMAT, USAGE);

	pool.begin_frame(0);
	REQUIRE(pool.acquire_image(id, EXTENT).has_value());
	pool.begin_frame(1);
	REQUIRE(pool.acquire_image(id, EXTENT).has_value()); // a second copy (frame 0 still in flight)
	REQUIRE(pool.image_copy_count(id) == 2);

	// A new extent drops the old copies and starts fresh.
	pool.begin_frame(2);
	REQUIRE(pool.acquire_image(id, vk::Extent2D{8, 8}).has_value());
	REQUIRE(pool.image_copy_count(id) == 1);
}

TEST_CASE("constant_image allocates one immutable copy that is never recycled", "[resources][pool][constant]")
{
	auto			   ctx = make_context();
	veng::ResourcePool pool(ctx.device(), ctx.rhi(), ctx.allocator(), 2);

	auto ref_result = pool.constant_image(ctx, vk::Extent2D{1, 1}, FORMAT, {0.0F, 0.0F, 0.0F, 1.0F});
	REQUIRE(ref_result.has_value());
	const veng::gpu::ImageRef ref = ref_result.value();
	REQUIRE(ctx.rhi().image(ref.texture));
	REQUIRE(ctx.rhi().view(ref.texture));
	REQUIRE(ref.pool_id != veng::gpu::ImageRef::INVALID_POOL_ID);
	REQUIRE(pool.image_copy_count(ref.pool_id) == 1);

	// Many frames pass and consumers touch the constant; it must not be recycled or duplicated.
	for (std::uint64_t i = 0; i < 10; ++i)
	{
		pool.begin_frame(i);
		pool.touch(ref.pool_id);
	}
	REQUIRE(pool.image_copy_count(ref.pool_id) == 1);
}

TEST_CASE("ResourcePool buffers reuse a copy once retired", "[resources][pool]")
{
	auto				 ctx = make_context();
	veng::ResourcePool	 pool(ctx.device(), ctx.rhi(), ctx.allocator(), 1);
	const veng::BufferId id = pool.declare_buffer(veng::rhi::BufferUsageFlags::UNIFORM);

	pool.begin_frame(0);
	auto b0 = pool.acquire_buffer(id, 64);
	REQUIRE(b0.has_value());
	REQUIRE(b0.value()->mapped() != nullptr);

	pool.begin_frame(1);
	auto b1 = pool.acquire_buffer(id, 64);
	REQUIRE(b1.has_value());
	REQUIRE(pool.buffer_copy_count(id) == 1); // single-buffered: reused
	REQUIRE(b0.value() == b1.value());
}
