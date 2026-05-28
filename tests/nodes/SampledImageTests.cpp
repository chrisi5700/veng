//
// L4 test (design.md §L4): one pass sampling another's output — the basis of multi-pass
// effects. Pass 1 (fullscreen) fills an image with a known color and leaves it in
// SHADER_READ_ONLY; pass 2 (fullscreen) samples it via add_sampled_image (a node-owned
// sampler + an image descriptor matched by reflected name) and writes it out. A readback
// proves the sampled color flowed across the pass boundary. Then the demand cascade: a held
// graph caches both; rebinding pass 2's sampled source to a *different* image pulls that
// image's producer into the plan and re-runs the sampler — and the readback shows the new
// color, proving set_sampled_image rebinds the descriptor, not just the demand.
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
#include <veng/resources/ResourcePool.hpp>

using namespace veng::graph;

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("SampledImage Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

bool plan_contains(const FramePlan& plan, NodeHandle node)
{
	for (const NodeHandle handle : plan.nodes())
	{
		if (handle == node)
		{
			return true;
		}
	}
	return false;
}

constexpr vk::Format	COLOR = vk::Format::eR8G8B8A8Unorm;
constexpr std::uint32_t SIDE  = 64;
} // namespace

TEST_CASE("a GraphicsNode samples another pass's output by reflected name", "[nodes][sampled][slice]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto			 ctx	= make_context();
	const vk::Device device = ctx.device();

	// Graph: two solid-color fullscreen passes (green, blue) -> images left SHADER_READ_ONLY;
	// a sampler pass samples one of them (start: green) and writes it out for readback.
	Graph			 graph;
	auto			 screen		 = graph.add_source<vk::Extent2D>(vk::Extent2D{SIDE, SIDE});
	auto			 green		 = graph.add_source<glm::vec4>(glm::vec4{0.0F, 1.0F, 0.0F, 1.0F});
	auto			 blue		 = graph.add_source<glm::vec4>(glm::vec4{0.0F, 0.0F, 1.0F, 1.0F});
	const DataHandle green_image = graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));
	const DataHandle blue_image	 = graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));
	const DataHandle out_image	 = graph.add(std::make_unique<ValueData<veng::gpu::ImageRef>>(veng::gpu::ImageRef{}));

	const auto make_solid = [&](TypedHandle<glm::vec4> color, DataHandle out)
	{
		auto node = std::make_unique<veng::nodes::GraphicsNode>("demo/fullscreen.vert", "tests/slice/solid.frag", COLOR,
																vk::Format::eUndefined, 3, screen, out);
		node->push_constant<glm::vec4>(color, vk::ShaderStageFlagBits::eFragment);
		return node;
	};
	const NodeHandle green_node = graph.add(make_solid(green, green_image));
	graph.set_producer(green_image, green_node);
	const NodeHandle blue_node = graph.add(make_solid(blue, blue_image));
	graph.set_producer(blue_image, blue_node);

	auto sampler = std::make_unique<veng::nodes::GraphicsNode>("demo/fullscreen.vert", "tests/slice/sample.frag", COLOR,
															   vk::Format::eUndefined, 3, screen, out_image);
	sampler->add_sampled_image(green_image, "source"); // start sampling the green image
	auto*			 sampler_ptr  = sampler.get();
	const NodeHandle sampler_node = graph.add(std::move(sampler));
	graph.set_producer(out_image, sampler_node);

	auto staging =
		veng::Buffer::create(ctx.allocator(), static_cast<vk::DeviceSize>(SIDE) * SIDE * 4,
							 vk::BufferUsageFlagBits::eTransferDst, vma::MemoryUsage::eAuto,
							 vma::AllocationCreateFlagBits::eMapped | vma::AllocationCreateFlagBits::eHostAccessRandom);
	REQUIRE(staging.has_value());

	const auto pool = device.createCommandPool(vk::CommandPoolCreateInfo()
												   .setQueueFamilyIndex(ctx.queue_indices().graphics)
												   .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer));
	REQUIRE(pool.result == vk::Result::eSuccess);
	const auto fence = device.createFence({});
	REQUIRE(fence.result == vk::Result::eSuccess);
	InlineScheduler scheduler;

	// One pool for the whole test (resources persist across both renders); the frame counter
	// advances the pool's retirement window so the single-buffered copies recycle between renders.
	veng::ResourcePool res_pool(ctx.device(), ctx.allocator(), 1);
	std::uint64_t	   frame_counter = 0;

	// Resolve + execute the demanded plan into a one-shot buffer, copy out_image to the staging
	// buffer, submit and wait; returns the plan (for assertions). The center pixel is then read
	// from `staging`.
	const auto render = [&]() -> FramePlan
	{
		const auto cmds = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo()
															.setCommandPool(pool.value)
															.setLevel(vk::CommandBufferLevel::ePrimary)
															.setCommandBufferCount(1));
		REQUIRE(cmds.result == vk::Result::eSuccess);
		const vk::CommandBuffer cmd = cmds.value.front();
		REQUIRE(cmd.begin(vk::CommandBufferBeginInfo().setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)) ==
				vk::Result::eSuccess);

		res_pool.begin_frame(frame_counter++);
		veng::gpu::GpuExecContext gpu_ctx(graph, ctx, res_pool, cmd, 0);
		auto					  plan = graph.resolve(std::array{out_image});
		REQUIRE(plan.has_value());
		REQUIRE(graph.execute(*plan, scheduler, gpu_ctx));
		REQUIRE(sampler_ptr->scene() != nullptr);

		const auto region =
			vk::BufferImageCopy()
				.setImageSubresource(
					vk::ImageSubresourceLayers().setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1))
				.setImageExtent(vk::Extent3D{SIDE, SIDE, 1});
		const auto* readback_ref = dynamic_cast<ValueData<veng::gpu::ImageRef>*>(graph.get_data(out_image));
		res_pool.transition_image(readback_ref->value().pool_id, cmd, vk::ImageLayout::eTransferSrcOptimal,
								  vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
		cmd.copyImageToBuffer(sampler_ptr->scene()->image(), vk::ImageLayout::eTransferSrcOptimal, staging->buffer(),
							  region);
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {},
							vk::MemoryBarrier()
								.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
								.setDstAccessMask(vk::AccessFlagBits::eHostRead),
							{}, {});
		REQUIRE(cmd.end() == vk::Result::eSuccess);
		REQUIRE(ctx.graphics_queue().submit(vk::SubmitInfo().setCommandBuffers(cmd), fence.value) ==
				vk::Result::eSuccess);
		REQUIRE(device.waitForFences(fence.value, vk::True, UINT64_MAX) == vk::Result::eSuccess);
		REQUIRE(device.resetFences(fence.value) == vk::Result::eSuccess);
		return std::move(plan.value());
	};

	const auto* pixels = static_cast<const std::uint8_t*>(staging->mapped());
	const auto	center = [&] { return (static_cast<std::size_t>(SIDE / 2) * SIDE + (SIDE / 2)) * 4; };

	// Cold: the green source pass + the sampler run; the blue pass is NOT demanded (nobody
	// samples it). The sampled green reached the output.
	const FramePlan cold = render();
	REQUIRE(plan_contains(cold, green_node));
	REQUIRE(plan_contains(cold, sampler_node));
	REQUIRE_FALSE(plan_contains(cold, blue_node));
	REQUIRE(pixels[center() + 0] == 0);
	REQUIRE(pixels[center() + 1] == 255); // green sampled through
	REQUIRE(pixels[center() + 2] == 0);

	// Held: nothing changed, the whole graph caches.
	const auto held = graph.resolve(std::array{out_image});
	REQUIRE(held.has_value());
	REQUIRE(held->empty());

	// Rebind the sampler's source to the blue image: that pulls the blue pass into demand
	// (cascade) and re-runs the sampler — and the readback shows blue, proving the descriptor
	// was actually rebound.
	sampler_ptr->set_sampled_image("source", blue_image);
	const FramePlan rebound = render();
	REQUIRE(plan_contains(rebound, blue_node));		   // blue pass now demanded
	REQUIRE(plan_contains(rebound, sampler_node));	   // sampler re-ran (its input changed)
	REQUIRE_FALSE(plan_contains(rebound, green_node)); // green no longer demanded
	REQUIRE(pixels[center() + 0] == 0);
	REQUIRE(pixels[center() + 1] == 0);
	REQUIRE(pixels[center() + 2] == 255); // now blue

	device.destroyFence(fence.value);
	device.destroyCommandPool(pool.value);
}
