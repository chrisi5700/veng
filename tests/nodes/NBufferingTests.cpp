//
// N-buffering integration proof ([[pass-draw-redesign]]). The ResourcePool's
// reuse/retention logic is unit-tested in ResourcePoolTests; this proves the *wired* path: with
// frames_in_flight = 2 and a producer that re-renders every frame, two frames submitted while both
// are in flight must render into DISTINCT physical target copies — otherwise frame N+1 would stomp
// pixels frame N's GPU work is still using. A solid GraphicsNode renders a different color each
// frame; we capture the target copy it wrote (GraphicsNode::scene()) per frame and require the two
// to differ, and a readback of each proves each copy holds its own frame's color.
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/resources/Image.hpp>
#include <veng/resources/ResourcePool.hpp>

using namespace veng::graph;

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("NBuffering Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

constexpr vk::Format	COLOR			 = vk::Format::eR8G8B8A8Unorm;
constexpr std::uint32_t SIDE			 = 64;
constexpr std::size_t	FRAMES_IN_FLIGHT = 2;
} // namespace

TEST_CASE("frames in flight render into distinct N-buffered target copies", "[nodes][nbuffering][slice]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			 ctx	= make_context();
	const vk::Device device = ctx.device();

	// A fullscreen solid pass whose color is a push constant: changing the color each frame forces
	// the node to re-render (so the producer is never cached), exercising the per-frame copies.
	Graph			 graph;
	auto			 screen = graph.add_source<vk::Extent2D>(vk::Extent2D{SIDE, SIDE});
	auto			 color	= graph.add_source<glm::vec4>(glm::vec4{0.0F, 0.0F, 0.0F, 1.0F});
	const DataHandle token	= graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));

	auto node = std::make_unique<veng::nodes::GraphicsNode>("demo/fullscreen.vert", "tests/slice/solid.frag", COLOR,
															vk::Format::eUndefined, 3, screen, token);
	node->push_constant<glm::vec4>(color, vk::ShaderStageFlagBits::eFragment); // default final layout: TRANSFER_SRC
	auto*			 node_ptr	 = node.get();
	const NodeHandle node_handle = graph.add(std::move(node));
	graph.set_producer(token, node_handle);

	veng::ResourcePool pool(ctx.device(), ctx.allocator(), FRAMES_IN_FLIGHT);
	InlineScheduler	   scheduler;

	const auto cmd_pool = device.createCommandPool(vk::CommandPoolCreateInfo().setQueueFamilyIndex(
		ctx.queue_indices().graphics)); // one pool, a buffer per in-flight frame
	REQUIRE(cmd_pool.result == vk::Result::eSuccess);
	const auto cmds = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo()
														.setCommandPool(cmd_pool.value)
														.setLevel(vk::CommandBufferLevel::ePrimary)
														.setCommandBufferCount(FRAMES_IN_FLIGHT));
	REQUIRE(cmds.result == vk::Result::eSuccess);

	const std::array<glm::vec4, FRAMES_IN_FLIGHT>			  colors{glm::vec4{1.0F, 0.0F, 0.0F, 1.0F},	 // red
																	 glm::vec4{0.0F, 0.0F, 1.0F, 1.0F}}; // blue
	std::array<veng::Buffer*, FRAMES_IN_FLIGHT>				  staging_ptrs{};
	std::array<std::optional<veng::Buffer>, FRAMES_IN_FLIGHT> staging;
	std::array<vk::Fence, FRAMES_IN_FLIGHT>					  fences{};
	std::array<const veng::Image*, FRAMES_IN_FLIGHT>		  target_copies{};

	for (std::size_t i = 0; i < FRAMES_IN_FLIGHT; ++i)
	{
		staging[i]		= std::move(veng::Buffer::create(ctx.allocator(), static_cast<vk::DeviceSize>(SIDE) * SIDE * 4,
														 vk::BufferUsageFlagBits::eTransferDst, vma::MemoryUsage::eAuto,
														 vma::AllocationCreateFlagBits::eMapped |
															 vma::AllocationCreateFlagBits::eHostAccessRandom)
										.value());
		staging_ptrs[i] = &staging[i].value();
		const auto fence = device.createFence({});
		REQUIRE(fence.result == vk::Result::eSuccess);
		fences[i] = fence.value;

		// A fresh color this frame -> the node re-renders into a pool copy.
		graph.set(color, colors[i]);
		pool.begin_frame(i);
		const vk::CommandBuffer cmd = cmds.value[i];
		REQUIRE(cmd.begin(vk::CommandBufferBeginInfo().setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)) ==
				vk::Result::eSuccess);

		veng::gpu::GpuExecContext gpu_ctx(graph, ctx, pool, cmd, i % FRAMES_IN_FLIGHT);
		auto					  plan = graph.resolve(std::array{token});
		REQUIRE(plan.has_value());
		REQUIRE(graph.execute(*plan, scheduler, gpu_ctx));
		REQUIRE(node_ptr->scene() != nullptr);
		target_copies[i] = node_ptr->scene(); // the physical copy written this frame

		const auto region =
			vk::BufferImageCopy()
				.setImageSubresource(
					vk::ImageSubresourceLayers().setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1))
				.setImageExtent(vk::Extent3D{SIDE, SIDE, 1});
		const auto* readback_ref = dynamic_cast<ValueData<veng::gpu::ImageRef>*>(graph.get_data(token));
		pool.transition_image(readback_ref->value().pool_id, cmd, vk::ImageLayout::eTransferSrcOptimal,
							  vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
		cmd.copyImageToBuffer(target_copies[i]->image(), vk::ImageLayout::eTransferSrcOptimal,
							  staging_ptrs[i]->buffer(), region);
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {},
							vk::MemoryBarrier()
								.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
								.setDstAccessMask(vk::AccessFlagBits::eHostRead),
							{}, {});
		REQUIRE(cmd.end() == vk::Result::eSuccess);
		// Submit WITHOUT waiting: both frames are in flight at once.
		REQUIRE(ctx.graphics_queue().submit(vk::SubmitInfo().setCommandBuffers(cmd), fences[i]) ==
				vk::Result::eSuccess);
	}

	REQUIRE(device.waitForFences(fences, vk::True, UINT64_MAX) == vk::Result::eSuccess);

	// The proof: at 2 in flight the second frame could not reuse the first frame's copy (it had
	// not retired), so the pool handed out a DISTINCT physical copy.
	REQUIRE(target_copies[0] != target_copies[1]);

	// And each copy holds its own frame's color (no cross-frame stomp).
	const auto center = [](const veng::Buffer& buf, std::size_t channel)
	{
		const auto* pixels = static_cast<const std::uint8_t*>(buf.mapped());
		return pixels[(static_cast<std::size_t>(SIDE / 2) * SIDE + (SIDE / 2)) * 4 + channel];
	};
	REQUIRE(center(*staging_ptrs[0], 0) == 255); // frame 0: red
	REQUIRE(center(*staging_ptrs[0], 2) == 0);
	REQUIRE(center(*staging_ptrs[1], 0) == 0); // frame 1: blue
	REQUIRE(center(*staging_ptrs[1], 2) == 255);

	for (const vk::Fence fence : fences)
	{
		device.destroyFence(fence);
	}
	device.destroyCommandPool(cmd_pool.value);
}

TEST_CASE("ResourcePool defers buffer-resize frees past the in-flight window", "[nodes][nbuffering][resize]")
{
	// A buffer resize (a cull's light-index list growing as the camera moves) must NOT free the old
	// copies while an in-flight frame's descriptor set may still reference them — unlike an image
	// resize, it isn't gated by a device-idle. The old copies are set aside and freed only once their
	// last_use has aged past the in-flight window. Pure bookkeeping — no GPU work needed.
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			   ctx = make_context();
	veng::ResourcePool pool(ctx.device(), ctx.allocator(), 2); // two frames in flight
	const veng::BufferId id = pool.declare_buffer(vk::BufferUsageFlagBits::eStorageBuffer);

	// Frame 0: acquire size A and mark it read by an in-flight consumer this frame.
	pool.begin_frame(0);
	REQUIRE(pool.acquire_buffer(id, 64).has_value());
	pool.touch_buffer(id);
	REQUIRE(pool.retiring_buffer_count() == 0);

	// Frame 1: resize. Frame 0 has not retired (2 in flight), so the size-A copy is set aside, not freed.
	pool.begin_frame(1);
	REQUIRE(pool.acquire_buffer(id, 128).has_value());
	REQUIRE(pool.retiring_buffer_count() == 1);

	// Frame 2: frame 0 has now retired. The purge runs in acquire_buffer (during node record, AFTER
	// the slot's in-flight fence is waited), NOT in begin_frame (which runs before that wait) — so it
	// takes a record-time acquire to free the copy. Purging in begin_frame would destroy a buffer a
	// frame still executing on the GPU references (VUID-vkDestroyBuffer-00922).
	pool.begin_frame(2);
	REQUIRE(pool.retiring_buffer_count() == 1); // begin_frame alone must NOT free it (pre-fence-wait)
	REQUIRE(pool.acquire_buffer(id, 128).has_value());
	REQUIRE(pool.retiring_buffer_count() == 0); // the record-time acquire purges the now-retired copy
}
