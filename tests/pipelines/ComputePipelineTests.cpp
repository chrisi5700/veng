//
// L2 ComputePipeline tests. Builds a real compute pipeline on a
// headless device from a reflected Slang shader, and checks the builder's error and
// RAII (move-only) behaviour.
//

#include <catch2/catch_test_macros.hpp>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/pipelines/ComputePipeline.hpp>
#include <veng/shader/Shader.hpp>

#include "support/VkFault.hpp"

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("ComputePipeline Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

veng::Shader load(const veng::Context& context, std::string_view name)
{
	auto result = veng::Shader::create_shader(context.device(), name);
	REQUIRE(result.has_value());
	return std::move(result.value());
}
} // namespace

TEST_CASE("ComputePipelineBuilder builds a pipeline from a reflected compute shader", "[pipeline][compute]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto	   ctx	   = make_context();
	const auto compute = load(ctx, "tests/loading/compute/simple_comp");
	REQUIRE(compute.get_details().stage() == vk::ShaderStageFlagBits::eCompute);

	const auto pipeline = veng::ComputePipelineBuilder(compute).build(ctx);
	REQUIRE(pipeline.has_value());
	REQUIRE(pipeline->pipeline());
	REQUIRE(pipeline->layout());
	REQUIRE(pipeline->descriptor_set_layout()); // one storage-buffer binding from reflection
}

TEST_CASE("ComputePipelineBuilder rejects a non-compute shader", "[pipeline][compute][error]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto	   ctx	= make_context();
	const auto vert = load(ctx, "tests/loading/vertex/simple_vert");

	const auto pipeline = veng::ComputePipelineBuilder(vert).build(ctx);
	REQUIRE_FALSE(pipeline.has_value());
	REQUIRE(pipeline.error() == veng::PipelineError::WRONG_STAGE);
}

TEST_CASE("ComputePipeline move transfers ownership without double free", "[pipeline][raii]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto	   ctx	   = make_context();
	const auto compute = load(ctx, "tests/loading/compute/simple_comp");

	auto built = veng::ComputePipelineBuilder(compute).build(ctx);
	REQUIRE(built.has_value());

	const veng::ComputePipeline moved = std::move(*built);
	REQUIRE(moved.pipeline());
	REQUIRE_FALSE(built->pipeline()); // moved-from is emptied, so destruction frees once
}

// --- Vulkan-object creation-failure branches --------------------------------------------------
// Each forces one creation step to fail and checks the mapped PipelineError. The earlier-created
// objects are real, so their cleanup runs for real — ASan/UBSan under the gate proves no leak.

TEST_CASE("ComputePipelineBuilder maps a descriptor-set-layout failure", "[pipeline][compute][error]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	auto	   ctx	   = make_context();
	const auto compute = load(ctx, "tests/loading/compute/simple_comp");

	const veng::test::ScopedDispatchFault fault{VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDescriptorSetLayout,
												+[](VkDevice, const VkDescriptorSetLayoutCreateInfo*,
													const VkAllocationCallbacks*, VkDescriptorSetLayout*) -> VkResult
												{ return VK_ERROR_OUT_OF_DEVICE_MEMORY; }};
	const auto							  pipeline = veng::ComputePipelineBuilder(compute).build(ctx);
	REQUIRE_FALSE(pipeline.has_value());
	REQUIRE(pipeline.error() == veng::PipelineError::DESCRIPTOR_SET_LAYOUT_CREATION_FAILED);
}

TEST_CASE("ComputePipelineBuilder maps a pipeline-layout failure and cleans up", "[pipeline][compute][error][raii]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	auto	   ctx	   = make_context();
	const auto compute = load(ctx, "tests/loading/compute/simple_comp");

	const veng::test::ScopedDispatchFault fault{
		VULKAN_HPP_DEFAULT_DISPATCHER.vkCreatePipelineLayout,
		+[](VkDevice, const VkPipelineLayoutCreateInfo*, const VkAllocationCallbacks*, VkPipelineLayout*) -> VkResult
		{ return VK_ERROR_OUT_OF_DEVICE_MEMORY; }};
	const auto pipeline = veng::ComputePipelineBuilder(compute).build(ctx);
	REQUIRE_FALSE(pipeline.has_value());
	REQUIRE(pipeline.error() == veng::PipelineError::PIPELINE_LAYOUT_CREATION_FAILED);
}

TEST_CASE("ComputePipelineBuilder maps a pipeline-creation failure and cleans up", "[pipeline][compute][error][raii]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	auto	   ctx	   = make_context();
	const auto compute = load(ctx, "tests/loading/compute/simple_comp");

	const veng::test::ScopedDispatchFault fault{VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateComputePipelines,
												+[](VkDevice, VkPipelineCache, uint32_t,
													const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*,
													VkPipeline*) -> VkResult { return VK_ERROR_OUT_OF_DEVICE_MEMORY; }};
	const auto							  pipeline = veng::ComputePipelineBuilder(compute).build(ctx);
	REQUIRE_FALSE(pipeline.has_value());
	REQUIRE(pipeline.error() == veng::PipelineError::PIPELINE_CREATION_FAILED);
}
