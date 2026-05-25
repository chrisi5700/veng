//
// GPU bridge tests (design.md §L4). Proves the L3 -> Vulkan seam end to end: a GpuNode
// dispatched by Graph::execute with an injected GpuExecContext receives the command
// buffer, records a real command, and that command executes on the GPU with an
// observable result (a host-visible buffer fill read back through the mapped pointer).
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <expected>
#include <memory>
#include <span>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/Buffer.hpp>

using veng::gpu::GpuExecContext;
using veng::gpu::GpuNode;
using namespace veng::graph;

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("GPU Bridge Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

/// A GpuNode that fills a buffer with a constant via the command buffer it is handed.
/// It also records what the GpuExecContext exposed, so the test can assert the seam
/// delivered the right command buffer / frame slot.
class FillBufferNode final : public GpuNode
{
	 public:
	FillBufferNode(vk::Buffer target, vk::DeviceSize size, std::uint32_t value, DataHandle output)
		: m_target(target)
		, m_size(size)
		, m_value(value)
		, m_output(output)
	{
	}

	[[nodiscard]] std::span<const DataHandle> inputs() const override { return {}; }
	[[nodiscard]] std::span<const DataHandle> outputs() const override { return {&m_output, 1}; }

	vk::CommandBuffer seen_command_buffer{};
	std::size_t		  seen_frame_slot = ~std::size_t{0};

	 protected:
	[[nodiscard]] std::expected<bool, ExecError> record(GpuExecContext& ctx) override
	{
		seen_command_buffer = ctx.command_buffer();
		seen_frame_slot		= ctx.frame_slot();
		ctx.command_buffer().fillBuffer(m_target, 0, m_size, m_value);
		return true;
	}

	 private:
	vk::Buffer	   m_target;
	vk::DeviceSize m_size;
	std::uint32_t  m_value;
	DataHandle	   m_output;
};
} // namespace

TEST_CASE("a GpuNode records through the injected GpuExecContext and the command runs", "[gpu][bridge]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			 ctx	= make_context();
	const vk::Device device = ctx.device();

	constexpr vk::DeviceSize SIZE  = 256;
	constexpr std::uint32_t	 VALUE = 0xABCD1234U;

	// Host-visible, persistently-mapped target so we can read the GPU's write back.
	auto target =
		veng::Buffer::create(ctx.allocator(), SIZE, vk::BufferUsageFlagBits::eTransferDst, vma::MemoryUsage::eAuto,
							 vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessRandom);
	REQUIRE(target.has_value());
	REQUIRE(target->mapped() != nullptr);

	// Graph: one GpuNode producing a dummy output we demand as the sink.
	Graph			 graph;
	const DataHandle out		 = graph.add(std::make_unique<ValueData<int>>(0));
	auto			 node		 = std::make_unique<FillBufferNode>(target->buffer(), SIZE, VALUE, out);
	auto*			 node_ptr	 = node.get();
	const NodeHandle node_handle = graph.add(std::move(node));
	graph.set_producer(out, node_handle);

	// Record into a one-shot command buffer.
	const auto pool =
		device.createCommandPool(vk::CommandPoolCreateInfo().setQueueFamilyIndex(ctx.queue_indices().graphics));
	REQUIRE(pool.result == vk::Result::eSuccess);
	const auto command_buffers = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo()
																   .setCommandPool(pool.value)
																   .setLevel(vk::CommandBufferLevel::ePrimary)
																   .setCommandBufferCount(1));
	REQUIRE(command_buffers.result == vk::Result::eSuccess);
	const vk::CommandBuffer cmd = command_buffers.value.front();

	REQUIRE(cmd.begin(vk::CommandBufferBeginInfo().setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)) ==
			vk::Result::eSuccess);

	// Drive the graph with a GPU context: resolve the plan, then execute with the
	// injected GpuExecContext so the GpuNode records into `cmd`.
	GpuExecContext	 gpu_ctx(graph, ctx, cmd, /*frame_slot=*/0);
	InlineScheduler	 scheduler;
	const std::array sinks{out};
	const auto		 plan = graph.resolve(sinks);
	REQUIRE(plan.has_value());
	REQUIRE(plan->size() == 1);
	graph.execute(*plan, scheduler, gpu_ctx);

	// The seam delivered the right command buffer + frame slot.
	REQUIRE(node_ptr->seen_command_buffer == cmd);
	REQUIRE(node_ptr->seen_frame_slot == 0);

	// Make the transfer write visible to the host, then submit and wait.
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {},
						vk::MemoryBarrier()
							.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
							.setDstAccessMask(vk::AccessFlagBits::eHostRead),
						{}, {});
	REQUIRE(cmd.end() == vk::Result::eSuccess);

	const auto fence = device.createFence({});
	REQUIRE(fence.result == vk::Result::eSuccess);
	REQUIRE(ctx.graphics_queue().submit(vk::SubmitInfo().setCommandBuffers(cmd), fence.value) == vk::Result::eSuccess);
	REQUIRE(device.waitForFences(fence.value, vk::True, UINT64_MAX) == vk::Result::eSuccess);

	// The GPU actually filled the buffer the node targeted.
	std::uint32_t read = 0;
	std::memcpy(&read, target->mapped(), sizeof(read));
	REQUIRE(read == VALUE);

	device.destroyFence(fence.value);
	device.destroyCommandPool(pool.value);
}

TEST_CASE("a GpuNode reached via the CPU execute path fails safe", "[gpu][bridge][error]")
{
	// M9: GpuExecContext and the core CPU context are siblings, so dispatching a GpuNode
	// with the default (CPU) context must not be UB. It is turned into a FAILED node.
	veng::Logger::instance().set_level(spdlog::level::warn);
	Graph			 graph;
	const DataHandle out  = graph.add(std::make_unique<ValueData<int>>(0));
	auto			 node = std::make_unique<FillBufferNode>(vk::Buffer{}, 0, 0U, out); // record() never runs
	const NodeHandle nh	  = graph.add(std::move(node));
	graph.set_producer(out, nh);

	InlineScheduler scheduler;
	const auto		plan = graph.frame(out, scheduler); // default CPU context — no GpuExecContext
	REQUIRE(plan.has_value());
	REQUIRE(graph.get_node(nh)->state() == ExecutionState::FAILED);
}
