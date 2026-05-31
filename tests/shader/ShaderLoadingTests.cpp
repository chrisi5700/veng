#include <catch2/catch_test_macros.hpp>
#include <map>
#include <veng/context/Context.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/shader/Shader.hpp>

#include "support/VkFault.hpp"

TEST_CASE("Simple vertex shader loads correctly", "[shader][loading][vertex]")
{
	veng::Logger::instance().set_level(spdlog::level::trace);
	auto ctx_res = veng::Context::create("Shader Test");
	REQUIRE(ctx_res);
	auto ctx = std::move(*ctx_res);

	auto shader_result = veng::Shader::create_shader(ctx.device(), "tests/loading/vertex/simple_vert");

	SECTION("shader loads without error")
	{
		REQUIRE(shader_result.has_value());
	}

	REQUIRE(shader_result.has_value());
	auto& shader = shader_result.value();

	SECTION("shader module is valid")
	{
		REQUIRE(shader.get_shader_module());
	}

	SECTION("shader has correct stage")
	{
		REQUIRE(shader.get_details().stage() == vk::ShaderStageFlagBits::eVertex);
	}

	SECTION("shader has one descriptor (transforms UBO)")
	{
		const auto& descriptors = shader.get_descriptor_infos();
		REQUIRE(descriptors.size() == 1);

		const auto& desc = descriptors[0];
		REQUIRE(desc.name == "transforms");
		REQUIRE(desc.binding == 0);
		REQUIRE(desc.set == 0);
		REQUIRE(desc.descriptor_count == 1);
		REQUIRE(desc.type == vk::DescriptorType::eUniformBuffer);
		REQUIRE(desc.stage == vk::ShaderStageFlagBits::eVertex);
		REQUIRE(desc.size == 192); // 3x float4x4 = 3 * 64 = 192 bytes
	}

	SECTION("shader has no push constants")
	{
		REQUIRE(!shader.get_push_constant_info().has_value());
	}

	SECTION("vertex details are correct")
	{
		const auto* vertex_details = std::get_if<veng::VertexDetails>(&shader.get_details());
		REQUIRE(vertex_details != nullptr);

		REQUIRE(vertex_details->inputs.size() == 2);   // position, normal
		REQUIRE(vertex_details->outputs.size() == 1);  // normal (position is SV_Position)
		REQUIRE(vertex_details->bindings.size() == 1); // Single binding

		// Check binding
		REQUIRE(vertex_details->bindings[0].binding == 0);
		REQUIRE(vertex_details->bindings[0].stride == 24); // float3 (12) + float3 (12) = 24

		// Check inputs
		REQUIRE(vertex_details->inputs[0].name == "position");
		REQUIRE(vertex_details->inputs[0].location == 0);
		REQUIRE(vertex_details->inputs[0].binding == 0);
		REQUIRE(vertex_details->inputs[0].offset == 0);
		REQUIRE(vertex_details->inputs[0].format == vk::Format::eR32G32B32Sfloat);

		REQUIRE(vertex_details->inputs[1].name == "normal");
		REQUIRE(vertex_details->inputs[1].location == 1);
		REQUIRE(vertex_details->inputs[1].binding == 0);
		REQUIRE(vertex_details->inputs[1].offset == 12); // After position (float3 = 12 bytes)
		REQUIRE(vertex_details->inputs[1].format == vk::Format::eR32G32B32Sfloat);

		// Check outputs
		REQUIRE(vertex_details->outputs[0].name == "normal");
		REQUIRE(vertex_details->outputs[0].location == 0);
		REQUIRE(vertex_details->outputs[0].format == vk::Format::eR32G32B32Sfloat);
	}
}

TEST_CASE("Vertex shader with descriptor array", "[shader][loading][vertex]")
{
	veng::Logger::instance().set_level(spdlog::level::trace);
	auto ctx_res = veng::Context::create("Shader Test");
	REQUIRE(ctx_res);
	auto ctx = std::move(*ctx_res);

	auto shader_result = veng::Shader::create_shader(ctx.device(), "tests/loading/vertex/descriptor_array.vert");

	REQUIRE(shader_result.has_value());
	auto& shader = shader_result.value();

	SECTION("shader has descriptor array of 4 MVPs")
	{
		const auto& descriptors = shader.get_descriptor_infos();
		REQUIRE(descriptors.size() == 1);

		const auto& desc = descriptors[0];
		REQUIRE(desc.name == "mvps");
		REQUIRE(desc.binding == 0);
		REQUIRE(desc.set == 0);
		REQUIRE(desc.descriptor_count == 4); // Array of 4
		REQUIRE(desc.type == vk::DescriptorType::eUniformBuffer);
		REQUIRE(desc.stage == vk::ShaderStageFlagBits::eVertex);
		REQUIRE(desc.size == 192); // Each MVP is 3x float4x4 = 192 bytes
	}
}

TEST_CASE("Simple fragment shader loads correctly", "[shader][loading][fragment]")
{
	veng::Logger::instance().set_level(spdlog::level::trace);
	auto ctx_res = veng::Context::create("Shader Test");
	REQUIRE(ctx_res);
	auto ctx = std::move(*ctx_res);

	auto shader_result = veng::Shader::create_shader(ctx.device(), "tests/loading/fragment/simple_frag");

	REQUIRE(shader_result.has_value());
	auto& shader = shader_result.value();

	SECTION("shader has correct stage")
	{
		REQUIRE(shader.get_details().stage() == vk::ShaderStageFlagBits::eFragment);
	}

	SECTION("shader has no descriptors")
	{
		const auto& descriptors = shader.get_descriptor_infos();
		REQUIRE(descriptors.empty());
	}

	SECTION("fragment details are correct")
	{
		const auto* frag_details = std::get_if<veng::FragmentDetails>(&shader.get_details());
		REQUIRE(frag_details != nullptr);

		REQUIRE(frag_details->inputs.size() == 1);	// normal
		REQUIRE(frag_details->outputs.size() == 0); // Dont count SV_Target

		// Check input
		REQUIRE(frag_details->inputs[0].name == "normal");
		REQUIRE(frag_details->inputs[0].location == 0);
		REQUIRE(frag_details->inputs[0].format == vk::Format::eR32G32B32Sfloat);
	}
}

TEST_CASE("Push constant vertex shader loads correctly", "[shader][loading][vertex]")
{
	veng::Logger::instance().set_level(spdlog::level::trace);
	auto ctx_res = veng::Context::create("Shader Test");
	REQUIRE(ctx_res);
	auto ctx = std::move(*ctx_res);

	auto shader_result = veng::Shader::create_shader(ctx.device(), "tests/loading/vertex/push_constant_vert");

	REQUIRE(shader_result.has_value());
	auto& shader = shader_result.value();

	SECTION("shader has no descriptors")
	{
		const auto& descriptors = shader.get_descriptor_infos();
		REQUIRE(descriptors.empty());
	}

	SECTION("shader has push constants")
	{
		const auto& push_const = shader.get_push_constant_info();
		REQUIRE(push_const.has_value());

		REQUIRE(push_const->name == "pushData");
		REQUIRE(push_const->size == 80); // float4x4 (64) + float (4) = 68 bytes -> Aligned to 80
		REQUIRE(push_const->offset == 0);
		REQUIRE(push_const->stage & vk::ShaderStageFlagBits::eVertex);
	}

	SECTION("vertex details are correct")
	{
		const auto* vertex_details = std::get_if<veng::VertexDetails>(&shader.get_details());
		REQUIRE(vertex_details != nullptr);

		REQUIRE(vertex_details->inputs.size() == 1);   // position only
		REQUIRE(vertex_details->outputs.size() == 0);  // no outputs (only SV_Position)
		REQUIRE(vertex_details->bindings.size() == 1); // Single binding

		// Check binding
		REQUIRE(vertex_details->bindings[0].binding == 0);
		REQUIRE(vertex_details->bindings[0].stride == 12); // float3 = 12 bytes

		REQUIRE(vertex_details->inputs[0].name == "position");
		REQUIRE(vertex_details->inputs[0].location == 0);
		REQUIRE(vertex_details->inputs[0].binding == 0);
		REQUIRE(vertex_details->inputs[0].offset == 0);
		REQUIRE(vertex_details->inputs[0].format == vk::Format::eR32G32B32Sfloat);
	}
}

TEST_CASE("Textured fragment shader loads correctly", "[shader][loading][fragment]")
{
	veng::Logger::instance().set_level(spdlog::level::trace);
	auto ctx_res = veng::Context::create("Shader Test");
	REQUIRE(ctx_res);
	auto ctx = std::move(*ctx_res);

	auto shader_result = veng::Shader::create_shader(ctx.device(), "tests/loading/fragment/textured_frag");

	REQUIRE(shader_result.has_value());
	auto& shader = shader_result.value();

	SECTION("shader has two descriptors (texture and sampler)")
	{
		const auto& descriptors = shader.get_descriptor_infos();
		REQUIRE(descriptors.size() == 2);

		// Find texture and sampler
		const veng::DescriptorInfo* texture = nullptr;
		const veng::DescriptorInfo* sampler = nullptr;

		for (const auto& desc : descriptors)
		{
			if (desc.name == "albedoTexture")
			{
				texture = &desc;
			}
			else if (desc.name == "textureSampler")
			{
				sampler = &desc;
			}
		}

		REQUIRE(texture != nullptr);
		REQUIRE(sampler != nullptr);

		// Texture
		REQUIRE(texture->binding == 0);
		REQUIRE(texture->set == 0);
		REQUIRE(texture->descriptor_count == 1);
		REQUIRE(texture->type == vk::DescriptorType::eSampledImage);
		REQUIRE(texture->stage == vk::ShaderStageFlagBits::eFragment);
		REQUIRE(texture->size == 0); // Textures don't have a size in the traditional sense

		// Sampler
		REQUIRE(sampler->binding == 1);
		REQUIRE(sampler->set == 0);
		REQUIRE(sampler->descriptor_count == 1);
		REQUIRE(sampler->type == vk::DescriptorType::eSampler);
		REQUIRE(sampler->stage == vk::ShaderStageFlagBits::eFragment);
		REQUIRE(sampler->size == 0);
	}

	SECTION("fragment details are correct")
	{
		const auto* frag_details = std::get_if<veng::FragmentDetails>(&shader.get_details());
		REQUIRE(frag_details != nullptr);

		REQUIRE(frag_details->inputs.size() == 1);	// texCoord
		REQUIRE(frag_details->outputs.size() == 0); // Dont count SV_Target

		REQUIRE(frag_details->inputs[0].name == "texCoord");
		REQUIRE(frag_details->inputs[0].location == 0);
		REQUIRE(frag_details->inputs[0].format == vk::Format::eR32G32Sfloat);
	}
}

TEST_CASE("Multi-descriptor fragment shader loads correctly", "[shader][loading][fragment]")
{
	veng::Logger::instance().set_level(spdlog::level::trace);
	auto ctx_res = veng::Context::create("Shader Test");
	REQUIRE(ctx_res);
	auto ctx = std::move(*ctx_res);

	auto shader_result = veng::Shader::create_shader(ctx.device(), "tests/loading/fragment/multi_descriptor_frag");

	REQUIRE(shader_result.has_value());
	auto& shader = shader_result.value();

	SECTION("shader has four descriptors")
	{
		const auto& descriptors = shader.get_descriptor_infos();
		REQUIRE(descriptors.size() == 4);

		// Map descriptors by name
		std::map<std::string, const veng::DescriptorInfo*> desc_map;
		for (const auto& desc : descriptors)
		{
			desc_map[desc.name] = &desc;
		}

		REQUIRE(desc_map.contains("material"));
		REQUIRE(desc_map.contains("albedoMap"));
		REQUIRE(desc_map.contains("normalMap"));
		REQUIRE(desc_map.contains("textureSampler"));

		// Material UBO
		const auto* material = desc_map["material"];
		REQUIRE(material->binding == 0);
		REQUIRE(material->set == 0);
		REQUIRE(material->descriptor_count == 1);
		REQUIRE(material->type == vk::DescriptorType::eUniformBuffer);
		REQUIRE(material->size == 32); // float4 (16) + float (4) + float (4) = 24 bytes -> Aligned to 32

		// Albedo texture
		const auto* albedo = desc_map["albedoMap"];
		REQUIRE(albedo->binding == 1);
		REQUIRE(albedo->set == 0);
		REQUIRE(albedo->descriptor_count == 1);
		REQUIRE(albedo->type == vk::DescriptorType::eSampledImage);

		// Normal texture
		const auto* normal = desc_map["normalMap"];
		REQUIRE(normal->binding == 2);
		REQUIRE(normal->set == 0);
		REQUIRE(normal->descriptor_count == 1);
		REQUIRE(normal->type == vk::DescriptorType::eSampledImage);

		// Sampler
		const auto* sampler = desc_map["textureSampler"];
		REQUIRE(sampler->binding == 3);
		REQUIRE(sampler->set == 0);
		REQUIRE(sampler->descriptor_count == 1);
		REQUIRE(sampler->type == vk::DescriptorType::eSampler);

		// All fragment stage
		for (const auto& desc : descriptors)
		{
			REQUIRE(desc.stage == vk::ShaderStageFlagBits::eFragment);
		}
	}
}

TEST_CASE("Simple geometry shader loads correctly", "[shader][loading][geometry]")
{
	veng::Logger::instance().set_level(spdlog::level::trace);
	auto ctx_res = veng::Context::create("Shader Test");
	REQUIRE(ctx_res);
	auto ctx = std::move(*ctx_res);

	auto shader_result = veng::Shader::create_shader(ctx.device(), "tests/loading/geometry/simple.geom");

	REQUIRE(shader_result.has_value());
	auto& shader = shader_result.value();

	SECTION("shader has correct stage")
	{
		REQUIRE(shader.get_details().stage() == vk::ShaderStageFlagBits::eGeometry);
	}

	SECTION("geometry details are correct")
	{
		const auto* geom_details = std::get_if<veng::GeometryDetails>(&shader.get_details());
		REQUIRE(geom_details != nullptr);

		// Inputs (from vertex shader)
		REQUIRE(geom_details->inputs.size() == 2); // position, color
		REQUIRE(geom_details->inputs[0].name == "position");
		REQUIRE(geom_details->inputs[0].location == 0);
		REQUIRE(geom_details->inputs[0].format == vk::Format::eR32G32B32Sfloat);

		REQUIRE(geom_details->inputs[1].name == "color");
		REQUIRE(geom_details->inputs[1].location == 1);
		REQUIRE(geom_details->inputs[1].format == vk::Format::eR32G32B32A32Sfloat);

		// Outputs (to fragment shader)
		REQUIRE(geom_details->outputs.size() ==
				1); // color // Note this is supposed to fail because my patch has not been merged yet
		REQUIRE(geom_details->outputs[0].name == "color");
		REQUIRE(geom_details->outputs[0].location == 0);
		REQUIRE(geom_details->outputs[0].format == vk::Format::eR32G32B32A32Sfloat);
	}
}

TEST_CASE("Simple tessellation control shader loads correctly", "[shader][loading][tessellation]")
{
	veng::Logger::instance().set_level(spdlog::level::trace);
	auto ctx_res = veng::Context::create("Shader Test");
	REQUIRE(ctx_res);
	auto ctx = std::move(*ctx_res);

	auto shader_result = veng::Shader::create_shader(ctx.device(), "tests/loading/tessellation/simple.hull");

	REQUIRE(shader_result.has_value());
	auto& shader = shader_result.value();

	SECTION("shader has correct stage")
	{
		REQUIRE(shader.get_details().stage() == vk::ShaderStageFlagBits::eTessellationControl);
	}

	SECTION("tessellation control details are correct")
	{
		const auto* tess_details = std::get_if<veng::TessellationControlDetails>(&shader.get_details());
		REQUIRE(tess_details != nullptr);

		// Inputs
		REQUIRE(tess_details->inputs.size() == 2); // position, normal
		REQUIRE(tess_details->inputs[0].name == "position");
		REQUIRE(tess_details->inputs[0].location == 0);
		REQUIRE(tess_details->inputs[0].format == vk::Format::eR32G32B32Sfloat);

		REQUIRE(tess_details->inputs[1].name == "normal");
		REQUIRE(tess_details->inputs[1].location == 1);
		REQUIRE(tess_details->inputs[1].format == vk::Format::eR32G32B32Sfloat);

		// Outputs
		REQUIRE(tess_details->outputs.size() == 2); // position, normal
		REQUIRE(tess_details->outputs[0].name == "position");
		REQUIRE(tess_details->outputs[0].location == 0);
		REQUIRE(tess_details->outputs[0].format == vk::Format::eR32G32B32Sfloat);

		REQUIRE(tess_details->outputs[1].name == "normal");
		REQUIRE(tess_details->outputs[1].location == 1);
		REQUIRE(tess_details->outputs[1].format == vk::Format::eR32G32B32Sfloat);
	}
}

TEST_CASE("Simple tessellation evaluation shader loads correctly", "[shader][loading][tessellation]")
{
	veng::Logger::instance().set_level(spdlog::level::trace);
	auto ctx_res = veng::Context::create("Shader Test");
	REQUIRE(ctx_res);
	auto ctx = std::move(*ctx_res);

	auto shader_result = veng::Shader::create_shader(ctx.device(), "tests/loading/tessellation/simple.domain");

	REQUIRE(shader_result.has_value());
	auto& shader = shader_result.value();

	SECTION("shader has correct stage")
	{
		REQUIRE(shader.get_details().stage() == vk::ShaderStageFlagBits::eTessellationEvaluation);
	}

	SECTION("tessellation evaluation details are correct")
	{
		const auto* tess_details = std::get_if<veng::TessellationEvaluationDetails>(&shader.get_details());
		REQUIRE(tess_details != nullptr);

		// Inputs
		REQUIRE(tess_details->inputs.size() == 2); // position, normal
		REQUIRE(tess_details->inputs[0].name == "position");
		REQUIRE(tess_details->inputs[0].location == 0);
		REQUIRE(tess_details->inputs[0].format == vk::Format::eR32G32B32Sfloat);

		REQUIRE(tess_details->inputs[1].name == "normal");
		REQUIRE(tess_details->inputs[1].location == 1);
		REQUIRE(tess_details->inputs[1].format == vk::Format::eR32G32B32Sfloat);

		// Outputs
		REQUIRE(tess_details->outputs.size() == 1); // normal
		REQUIRE(tess_details->outputs[0].name == "normal");
		REQUIRE(tess_details->outputs[0].location == 0);
		REQUIRE(tess_details->outputs[0].format == vk::Format::eR32G32B32Sfloat);
	}
}

TEST_CASE("Vertex shader with multiple bindings loads correctly", "[shader][loading][vertex]")
{
	veng::Logger::instance().set_level(spdlog::level::trace);
	auto ctx_res = veng::Context::create("Shader Test");
	REQUIRE(ctx_res);
	auto ctx = std::move(*ctx_res);

	auto shader_result = veng::Shader::create_shader(ctx.device(), "tests/loading/vertex/multiple_bindings.vert");

	REQUIRE(shader_result.has_value());
	auto& shader = shader_result.value();

	SECTION("shader has correct stage")
	{
		REQUIRE(shader.get_details().stage() == vk::ShaderStageFlagBits::eVertex);
	}

	SECTION("vertex details have correct number of bindings and inputs")
	{
		const auto* vertex_details = std::get_if<veng::VertexDetails>(&shader.get_details());
		REQUIRE(vertex_details != nullptr);

		REQUIRE(vertex_details->bindings.size() == 2); // PerVertex and PerInstance
		REQUIRE(vertex_details->inputs.size() == 6);   // 2 from PerVertex + 4 from PerInstance
	}

	SECTION("binding 0 (PerVertex) is correct")
	{
		const auto* vertex_details = std::get_if<veng::VertexDetails>(&shader.get_details());
		REQUIRE(vertex_details != nullptr);

		const auto& binding0 = vertex_details->bindings[0];
		REQUIRE(binding0.binding == 0);
		REQUIRE(binding0.stride == 24); // float3 (12) + float3 (12) = 24
		REQUIRE(binding0.name == "PerVertex");
	}

	SECTION("binding 1 (PerInstance) is correct")
	{
		const auto* vertex_details = std::get_if<veng::VertexDetails>(&shader.get_details());
		REQUIRE(vertex_details != nullptr);

		const auto& binding1 = vertex_details->bindings[1];
		REQUIRE(binding1.binding == 1);
		REQUIRE(binding1.stride == 64); // 4 * float4 (4 * 16) = 64
		REQUIRE(binding1.name == "PerInstance");
	}

	SECTION("PerVertex attributes are correct")
	{
		const auto* vertex_details = std::get_if<veng::VertexDetails>(&shader.get_details());
		REQUIRE(vertex_details != nullptr);

		// Position (location 0, binding 0)
		const auto& position = vertex_details->inputs[0];
		REQUIRE(position.name == "position");
		REQUIRE(position.location == 0);
		REQUIRE(position.binding == 0);
		REQUIRE(position.offset == 0);
		REQUIRE(position.format == vk::Format::eR32G32B32Sfloat);

		// Normal (location 1, binding 0)
		const auto& normal = vertex_details->inputs[1];
		REQUIRE(normal.name == "normal");
		REQUIRE(normal.location == 1);
		REQUIRE(normal.binding == 0);
		REQUIRE(normal.offset == 12); // After position (float3 = 12 bytes)
		REQUIRE(normal.format == vk::Format::eR32G32B32Sfloat);
	}

	SECTION("PerInstance attributes are correct")
	{
		const auto* vertex_details = std::get_if<veng::VertexDetails>(&shader.get_details());
		REQUIRE(vertex_details != nullptr);

		// Matrix column 0 (location 2, binding 1)
		const auto& col0 = vertex_details->inputs[2];
		REQUIRE(col0.name == "matrixCol0");
		REQUIRE(col0.location == 2);
		REQUIRE(col0.binding == 1);
		REQUIRE(col0.offset == 0);
		REQUIRE(col0.format == vk::Format::eR32G32B32A32Sfloat);

		// Matrix column 1 (location 3, binding 1)
		const auto& col1 = vertex_details->inputs[3];
		REQUIRE(col1.name == "matrixCol1");
		REQUIRE(col1.location == 3);
		REQUIRE(col1.binding == 1);
		REQUIRE(col1.offset == 16); // After col0 (float4 = 16 bytes)
		REQUIRE(col1.format == vk::Format::eR32G32B32A32Sfloat);

		// Matrix column 2 (location 4, binding 1)
		const auto& col2 = vertex_details->inputs[4];
		REQUIRE(col2.name == "matrixCol2");
		REQUIRE(col2.location == 4);
		REQUIRE(col2.binding == 1);
		REQUIRE(col2.offset == 32); // After col1 (2 * float4 = 32 bytes)
		REQUIRE(col2.format == vk::Format::eR32G32B32A32Sfloat);

		// Matrix column 3 (location 5, binding 1)
		const auto& col3 = vertex_details->inputs[5];
		REQUIRE(col3.name == "matrixCol3");
		REQUIRE(col3.location == 5);
		REQUIRE(col3.binding == 1);
		REQUIRE(col3.offset == 48); // After col2 (3 * float4 = 48 bytes)
		REQUIRE(col3.format == vk::Format::eR32G32B32A32Sfloat);
	}

	SECTION("attributes can be converted to Vulkan descriptions")
	{
		const auto* vertex_details = std::get_if<veng::VertexDetails>(&shader.get_details());
		REQUIRE(vertex_details != nullptr);

		// Test conversion for position attribute
		auto position_desc = vertex_details->inputs[0].to_attribute_description();
		REQUIRE(position_desc.location == 0);
		REQUIRE(position_desc.binding == 0);
		REQUIRE(position_desc.offset == 0);
		REQUIRE(position_desc.format == vk::Format::eR32G32B32Sfloat);

		// Test conversion for instance matrix column
		auto col0_desc = vertex_details->inputs[2].to_attribute_description();
		REQUIRE(col0_desc.location == 2);
		REQUIRE(col0_desc.binding == 1);
		REQUIRE(col0_desc.offset == 0);
		REQUIRE(col0_desc.format == vk::Format::eR32G32B32A32Sfloat);
	}

	SECTION("bindings can be converted to Vulkan descriptions")
	{
		const auto* vertex_details = std::get_if<veng::VertexDetails>(&shader.get_details());
		REQUIRE(vertex_details != nullptr);

		// Test PerVertex binding (per-vertex rate)
		auto vertex_binding_desc = vertex_details->bindings[0].to_binding_description(vk::VertexInputRate::eVertex);
		REQUIRE(vertex_binding_desc.binding == 0);
		REQUIRE(vertex_binding_desc.stride == 24);
		REQUIRE(vertex_binding_desc.inputRate == vk::VertexInputRate::eVertex);

		// Test PerInstance binding (per-instance rate)
		auto instance_binding_desc = vertex_details->bindings[1].to_binding_description(vk::VertexInputRate::eInstance);
		REQUIRE(instance_binding_desc.binding == 1);
		REQUIRE(instance_binding_desc.stride == 64);
		REQUIRE(instance_binding_desc.inputRate == vk::VertexInputRate::eInstance);
	}
}

TEST_CASE("Invalid shader fails gracefully", "[shader][loading][error]")
{
	veng::Logger::instance().set_level(spdlog::level::trace);
	auto ctx_res = veng::Context::create("Shader Test");
	REQUIRE(ctx_res);
	auto ctx = std::move(*ctx_res);

	SECTION("nonexistent shader returns error")
	{
		auto result = veng::Shader::create_shader(ctx.device(), "tests/loading/nonexistent_shader");
		REQUIRE(!result.has_value());
		REQUIRE(!result.error().empty());
	}

	SECTION("invalid entry point returns error")
	{
		auto result = veng::Shader::create_shader(ctx.device(), "tests/loading/vertex/simple_vert", "invalid_entry");
		REQUIRE(!result.has_value());
		REQUIRE(!result.error().empty());
	}

	SECTION("a failed vkCreateShaderModule is reported, not asserted")
	{
		// The shader compiles and reflects fine; only module creation fails. Before the engine emptied
		// VULKAN_HPP_ASSERT_ON_RESULT this branch was unreachable (the convenience wrapper aborted).
		const veng::test::ScopedDispatchFault fault{
			VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateShaderModule,
			+[](VkDevice, const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*, VkShaderModule*) -> VkResult
			{ return VK_ERROR_OUT_OF_DEVICE_MEMORY; }};
		auto result = veng::Shader::create_shader(ctx.device(), "tests/loading/vertex/simple_vert");
		REQUIRE(!result.has_value());
		REQUIRE(!result.error().empty());
	}
}
