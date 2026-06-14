//
// L2.1 GraphicsPipeline tests. Builds a real graphics pipeline via
// Vulkan 1.3 dynamic rendering from a reflected vertex+fragment pair on a headless
// device, and checks the builder's stage/compat/format guards and RAII.
//

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <utility>
#include <veng/context/Context.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/pipelines/GraphicsPipeline.hpp>
#include <veng/shader/Shader.hpp>

#include "support/VkFault.hpp"

namespace
{
veng::Context make_context()
{
	auto result = veng::Context::create("GraphicsPipeline Test");
	REQUIRE(result.has_value());
	return std::move(result.value());
}

veng::Shader load(const veng::Context& context, std::string_view name)
{
	auto result = veng::Shader::create_shader(context.device(), name);
	REQUIRE(result.has_value());
	return std::move(result.value());
}

constexpr veng::rhi::Format COLOR = veng::rhi::Format::RGBA8_UNORM;
} // namespace

TEST_CASE("GraphicsPipelineBuilder builds a dynamic-rendering pipeline from vert+frag", "[pipeline][graphics]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto	   ctx	= make_context();
	const auto vert = load(ctx, "tests/matching/vert_to_frag/simple.vert");
	const auto frag = load(ctx, "tests/matching/vert_to_frag/simple.frag");

	const std::array formats{COLOR};
	const auto		 pipeline = veng::GraphicsPipelineBuilder(vert, frag).color_formats(formats).build(ctx);

	REQUIRE(pipeline.has_value());
	REQUIRE(pipeline->pipeline());
	REQUIRE(pipeline->layout());
}

TEST_CASE("GraphicsPipelineBuilder requires at least one color format", "[pipeline][graphics][error]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto	   ctx	= make_context();
	const auto vert = load(ctx, "tests/matching/vert_to_frag/simple.vert");
	const auto frag = load(ctx, "tests/matching/vert_to_frag/simple.frag");

	const auto pipeline = veng::GraphicsPipelineBuilder(vert, frag).build(ctx); // no color_formats
	REQUIRE_FALSE(pipeline.has_value());
	REQUIRE(pipeline.error() == veng::PipelineError::MISSING_COLOR_FORMATS);
}

TEST_CASE("GraphicsPipelineBuilder rejects swapped stages", "[pipeline][graphics][error]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto	   ctx	= make_context();
	const auto vert = load(ctx, "tests/matching/vert_to_frag/simple.vert");
	const auto frag = load(ctx, "tests/matching/vert_to_frag/simple.frag");

	const std::array formats{COLOR};
	// frag in the vertex slot, vert in the fragment slot.
	const auto pipeline = veng::GraphicsPipelineBuilder(frag, vert).color_formats(formats).build(ctx);
	REQUIRE_FALSE(pipeline.has_value());
	REQUIRE(pipeline.error() == veng::PipelineError::WRONG_STAGE);
}

TEST_CASE("GraphicsPipelineBuilder rejects incompatible stage interfaces", "[pipeline][graphics][error]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto ctx = make_context();
	// passthrough.vert outputs do NOT directly match passthrough.frag inputs (a
	// geometry stage belongs between them) — the builder must reject the pair.
	const auto vert = load(ctx, "tests/matching/vert_to_geom_to_frag/passthrough.vert");
	const auto frag = load(ctx, "tests/matching/vert_to_geom_to_frag/passthrough.frag");

	const std::array formats{COLOR};
	const auto		 pipeline = veng::GraphicsPipelineBuilder(vert, frag).color_formats(formats).build(ctx);
	REQUIRE_FALSE(pipeline.has_value());
	REQUIRE(pipeline.error() == veng::PipelineError::STAGE_INCOMPATIBLE);
}

TEST_CASE("GraphicsPipeline move transfers ownership without double free", "[pipeline][graphics][raii]")
{
	veng::Logger::instance().set_level(spdlog::level::warn);
	auto	   ctx	= make_context();
	const auto vert = load(ctx, "tests/matching/vert_to_frag/simple.vert");
	const auto frag = load(ctx, "tests/matching/vert_to_frag/simple.frag");

	const std::array formats{COLOR};
	auto			 built = veng::GraphicsPipelineBuilder(vert, frag).color_formats(formats).build(ctx);
	REQUIRE(built.has_value());

	const veng::GraphicsPipeline moved = std::move(*built);
	REQUIRE(moved.pipeline());
	REQUIRE_FALSE(built->pipeline()); // moved-from emptied -> destroyed once
}

// --- Vulkan-object creation-failure branches --------------------------------------------------
// Mirror the compute builder: each forces one creation step to fail and checks the mapped error.
// Earlier objects are real, so their cleanup runs for real — ASan/UBSan under the gate proves no leak.

namespace
{
veng::GraphicsPipelineBuilder vert_frag_builder(const veng::Shader& vert, const veng::Shader& frag)
{
	static constexpr std::array formats{COLOR};
	return veng::GraphicsPipelineBuilder(vert, frag).color_formats(formats);
}
} // namespace

TEST_CASE("GraphicsPipelineBuilder maps a descriptor-set-layout failure", "[pipeline][graphics][error]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	auto	   ctx	= make_context();
	const auto vert = load(ctx, "tests/matching/vert_to_frag/simple.vert");
	const auto frag = load(ctx, "tests/matching/vert_to_frag/simple.frag");

	const veng::test::ScopedDispatchFault fault{VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDescriptorSetLayout,
												+[](VkDevice, const VkDescriptorSetLayoutCreateInfo*,
													const VkAllocationCallbacks*, VkDescriptorSetLayout*) -> VkResult
												{ return VK_ERROR_OUT_OF_DEVICE_MEMORY; }};
	const auto							  pipeline = vert_frag_builder(vert, frag).build(ctx);
	REQUIRE_FALSE(pipeline.has_value());
	REQUIRE(pipeline.error() == veng::PipelineError::DESCRIPTOR_SET_LAYOUT_CREATION_FAILED);
}

TEST_CASE("GraphicsPipelineBuilder maps a pipeline-layout failure and cleans up", "[pipeline][graphics][error][raii]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	auto	   ctx	= make_context();
	const auto vert = load(ctx, "tests/matching/vert_to_frag/simple.vert");
	const auto frag = load(ctx, "tests/matching/vert_to_frag/simple.frag");

	const veng::test::ScopedDispatchFault fault{
		VULKAN_HPP_DEFAULT_DISPATCHER.vkCreatePipelineLayout,
		+[](VkDevice, const VkPipelineLayoutCreateInfo*, const VkAllocationCallbacks*, VkPipelineLayout*) -> VkResult
		{ return VK_ERROR_OUT_OF_DEVICE_MEMORY; }};
	const auto pipeline = vert_frag_builder(vert, frag).build(ctx);
	REQUIRE_FALSE(pipeline.has_value());
	REQUIRE(pipeline.error() == veng::PipelineError::PIPELINE_LAYOUT_CREATION_FAILED);
}

TEST_CASE("GraphicsPipelineBuilder maps a pipeline-creation failure and cleans up", "[pipeline][graphics][error][raii]")
{
	veng::Logger::instance().set_level(spdlog::level::err);
	auto	   ctx	= make_context();
	const auto vert = load(ctx, "tests/matching/vert_to_frag/simple.vert");
	const auto frag = load(ctx, "tests/matching/vert_to_frag/simple.frag");

	const veng::test::ScopedDispatchFault fault{VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateGraphicsPipelines,
												+[](VkDevice, VkPipelineCache, uint32_t,
													const VkGraphicsPipelineCreateInfo*, const VkAllocationCallbacks*,
													VkPipeline*) -> VkResult { return VK_ERROR_OUT_OF_DEVICE_MEMORY; }};
	const auto							  pipeline = vert_frag_builder(vert, frag).build(ctx);
	REQUIRE_FALSE(pipeline.has_value());
	REQUIRE(pipeline.error() == veng::PipelineError::PIPELINE_CREATION_FAILED);
}
