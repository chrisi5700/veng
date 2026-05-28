//
// L2.4 manager tests (design.md §L2.4). CommandManager (per queue/slot/thread command pools)
// on a headless device: command recording that executes on-GPU, per-frame reset, and
// per-thread pool isolation under concurrency.
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <future>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/managers/CommandManager.hpp>
#include <veng/resources/Buffer.hpp>

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("Manager Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

// Submit a one-shot command buffer on the graphics queue and wait for it.
void submit_and_wait(const veng::Context& ctx, vk::CommandBuffer cmd)
{
	REQUIRE(cmd.end() == vk::Result::eSuccess);
	const auto fence = ctx.device().createFence({});
	REQUIRE(fence.result == vk::Result::eSuccess);
	REQUIRE(ctx.graphics_queue().submit(vk::SubmitInfo().setCommandBuffers(cmd), fence.value) == vk::Result::eSuccess);
	REQUIRE(ctx.device().waitForFences(fence.value, vk::True, UINT64_MAX) == vk::Result::eSuccess);
	ctx.device().destroyFence(fence.value);
}
} // namespace

TEST_CASE("CommandManager records a command buffer that executes on the GPU", "[managers][command]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();

	constexpr vk::DeviceSize SIZE  = 64;
	constexpr std::uint32_t	 VALUE = 0x1234ABCDU;
	auto					 target =
		veng::Buffer::create(ctx.allocator(), SIZE, vk::BufferUsageFlagBits::eTransferDst, vma::MemoryUsage::eAuto,
							 vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessRandom);
	REQUIRE(target.has_value());

	veng::CommandManager commands(ctx);
	auto				 cmd = commands.begin(veng::QueueKind::Graphics, 0);
	REQUIRE(cmd.has_value());

	cmd->fillBuffer(target->buffer(), 0, SIZE, VALUE);
	cmd->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {},
						 vk::MemoryBarrier()
							 .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
							 .setDstAccessMask(vk::AccessFlagBits::eHostRead),
						 {}, {});
	submit_and_wait(ctx, *cmd);

	std::uint32_t read = 0;
	std::memcpy(&read, target->mapped(), sizeof(read));
	REQUIRE(read == VALUE);

	// reset_frame recycles the slot's pools; a subsequent begin still works.
	commands.reset_frame(0);
	REQUIRE(commands.begin(veng::QueueKind::Graphics, 0).has_value());
}

TEST_CASE("CommandManager hands different threads distinct command buffers", "[managers][command][concurrent]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto				 ctx = make_context();
	veng::CommandManager commands(ctx);

	// Two threads recording the same (queue, slot) must get buffers from distinct
	// per-thread pools — never the same handle (which would race the pool).
	auto	   a		= std::async(std::launch::async, [&] { return commands.begin(veng::QueueKind::Graphics, 0); });
	auto	   b		= std::async(std::launch::async, [&] { return commands.begin(veng::QueueKind::Graphics, 0); });
	const auto buffer_a = a.get();
	const auto buffer_b = b.get();

	REQUIRE(buffer_a.has_value());
	REQUIRE(buffer_b.has_value());
	REQUIRE(*buffer_a != *buffer_b);
}
