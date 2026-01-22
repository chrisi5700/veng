#include <catch2/catch_test_macros.hpp>
#include <variant>
#include <veng/context/Context.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/shader/Shader.hpp>

// Helper function to load a shader and require it succeeds
veng::Shader load_shader(vk::Device device, std::string_view name)
{
    auto result = veng::Shader::create_shader(device, name);
    REQUIRE(result.has_value());
    return std::move(result.value());
}

TEST_CASE("Vertex to Fragment matching - simple pipeline", "[shader][validation][matching]")
{
    veng::Logger::instance().set_level(spdlog::level::warn);
    veng::VulkanContext ctx("veng::Shader Validation Test");

    auto vert = load_shader(ctx.device(), "tests/matching/vert_to_frag/simple.vert");
    auto frag = load_shader(ctx.device(), "tests/matching/vert_to_frag/simple.frag");

    SECTION("vertex outputs match fragment inputs")
    {
        REQUIRE(vert.get_details().matches(frag.get_details()));
    }

    SECTION("vertex details are correct")
    {
        const auto* vert_details = std::get_if<veng::VertexDetails>(&vert.get_details());
        REQUIRE(vert_details != nullptr);
        REQUIRE(vert_details->outputs.size() == 2); // normal, texCoord
    }

    SECTION("fragment details are correct")
    {
        const auto* frag_details = std::get_if<veng::FragmentDetails>(&frag.get_details());
        REQUIRE(frag_details != nullptr);
        REQUIRE(frag_details->inputs.size() == 2); // normal, texCoord
    }
}

TEST_CASE("Vertex to Geometry to Fragment matching", "[shader][validation][matching]")
{
    veng::Logger::instance().set_level(spdlog::level::warn);
    veng::VulkanContext ctx("veng::Shader Validation Test");

    auto vert = load_shader(ctx.device(), "tests/matching/vert_to_geom_to_frag/passthrough.vert");
    auto geom = load_shader(ctx.device(), "tests/matching/vert_to_geom_to_frag/passthrough.geom");
    auto frag = load_shader(ctx.device(), "tests/matching/vert_to_geom_to_frag/passthrough.frag");

    SECTION("vertex outputs match geometry inputs")
    {
        REQUIRE(vert.get_details().matches(geom.get_details()));
    }

    SECTION("geometry outputs match fragment inputs")
    {
        REQUIRE(geom.get_details().matches(frag.get_details()));
    }

    SECTION("vertex does NOT directly match fragment")
    {
        // Vertex outputs color, but fragment expects color AND barycentrics
        // This should fail because the interfaces don't match directly
        REQUIRE_FALSE(vert.get_details().matches(frag.get_details()));
    }

    SECTION("geometry details are correct")
    {
        const auto* geom_details = std::get_if<veng::GeometryDetails>(&geom.get_details());
        REQUIRE(geom_details != nullptr);
        REQUIRE(geom_details->inputs.size() == 1);  // color
        REQUIRE(geom_details->outputs.size() == 2); // color, barycentrics
    }
}

TEST_CASE("Vertex to Tessellation to Fragment matching", "[shader][validation][matching]")
{
    veng::Logger::instance().set_level(spdlog::level::warn);
    veng::VulkanContext ctx("veng::Shader Validation Test");

    auto vert = load_shader(ctx.device(), "tests/matching/vert_to_tess_to_frag/tess.vert");
    auto hull = load_shader(ctx.device(), "tests/matching/vert_to_tess_to_frag/tess.hull");
    auto domain = load_shader(ctx.device(), "tests/matching/vert_to_tess_to_frag/tess.domain");
    auto frag = load_shader(ctx.device(), "tests/matching/vert_to_tess_to_frag/tess.frag");

    SECTION("vertex outputs match tessellation control inputs")
    {
        REQUIRE(vert.get_details().matches(hull.get_details()));
    }

    SECTION("tessellation control outputs match tessellation evaluation inputs")
    {
        REQUIRE(hull.get_details().matches(domain.get_details()));
    }

    SECTION("tessellation evaluation outputs match fragment inputs")
    {
        REQUIRE(domain.get_details().matches(frag.get_details()));
    }

    SECTION("tessellation control details are correct")
    {
        const auto* hull_details = std::get_if<veng::TessellationControlDetails>(&hull.get_details());
        REQUIRE(hull_details != nullptr);
        REQUIRE(hull_details->inputs.size() == 1);  // normal
        REQUIRE(hull_details->outputs.size() == 2); // normal, tessLevel
    }

    SECTION("tessellation evaluation details are correct")
    {
        const auto* domain_details = std::get_if<veng::TessellationEvaluationDetails>(&domain.get_details());
        REQUIRE(domain_details != nullptr);
        REQUIRE(domain_details->inputs.size() == 2);  // normal, tessLevel
        REQUIRE(domain_details->outputs.size() == 1); // normal
    }
}

TEST_CASE("Format mismatch detection", "[shader][validation][mismatch]")
{
    veng::Logger::instance().set_level(spdlog::level::warn);
    veng::VulkanContext ctx("veng::Shader Validation Test");

    auto vert = load_shader(ctx.device(), "tests/mismatched/format_mismatch/wrong_format.vert");
    auto frag = load_shader(ctx.device(), "tests/mismatched/format_mismatch/wrong_format.frag");

    SECTION("vertex outputs float3 at location 0")
    {
        const auto* vert_details = std::get_if<veng::VertexDetails>(&vert.get_details());
        REQUIRE(vert_details != nullptr);
        REQUIRE(vert_details->outputs.size() == 1);
        REQUIRE(vert_details->outputs[0].location == 0);
        REQUIRE(vert_details->outputs[0].format == vk::Format::eR32G32B32Sfloat);
    }

    SECTION("fragment expects float4 at location 0")
    {
        const auto* frag_details = std::get_if<veng::FragmentDetails>(&frag.get_details());
        REQUIRE(frag_details != nullptr);
        REQUIRE(frag_details->inputs.size() == 1);
        REQUIRE(frag_details->inputs[0].location == 0);
        REQUIRE(frag_details->inputs[0].format == vk::Format::eR32G32B32A32Sfloat);
    }

    SECTION("shaders do NOT match due to format mismatch")
    {
        // float3 vs float4 at same location
        REQUIRE_FALSE(vert.get_details().matches(frag.get_details()));
    }
}

TEST_CASE("Missing output detection", "[shader][validation][mismatch]")
{
    veng::Logger::instance().set_level(spdlog::level::warn);
    veng::VulkanContext ctx("veng::Shader Validation Test");

    auto vert = load_shader(ctx.device(), "tests/mismatched/missing_output/incomplete.vert");
    auto frag = load_shader(ctx.device(), "tests/mismatched/missing_output/expects_more.frag");

    SECTION("vertex only outputs normal")
    {
        const auto* vert_details = std::get_if<veng::VertexDetails>(&vert.get_details());
        REQUIRE(vert_details != nullptr);
        REQUIRE(vert_details->outputs.size() == 1);
        REQUIRE(vert_details->outputs[0].location == 0);
    }

    SECTION("fragment expects normal AND texCoord")
    {
        const auto* frag_details = std::get_if<veng::FragmentDetails>(&frag.get_details());
        REQUIRE(frag_details != nullptr);
        REQUIRE(frag_details->inputs.size() == 2);
    }

    SECTION("shaders do NOT match due to missing output")
    {
        // Vertex doesn't provide texCoord at location 1
        REQUIRE_FALSE(vert.get_details().matches(frag.get_details()));
    }
}

TEST_CASE("Location mismatch detection", "[shader][validation][mismatch]")
{
    veng::Logger::instance().set_level(spdlog::level::warn);
    veng::VulkanContext ctx("veng::Shader Validation Test");

    auto vert = load_shader(ctx.device(), "tests/mismatched/location_mismatch/wrong_location.vert");
    auto frag = load_shader(ctx.device(), "tests/mismatched/location_mismatch/expects_location_0.frag");

    SECTION("vertex outputs at location 1")
    {
        const auto* vert_details = std::get_if<veng::VertexDetails>(&vert.get_details());
        REQUIRE(vert_details != nullptr);
        REQUIRE(vert_details->outputs.size() == 1);
        REQUIRE(vert_details->outputs[0].location == 1);
    }

    SECTION("fragment expects input at location 0")
    {
        const auto* frag_details = std::get_if<veng::FragmentDetails>(&frag.get_details());
        REQUIRE(frag_details != nullptr);
        REQUIRE(frag_details->inputs.size() == 1);
        REQUIRE(frag_details->inputs[0].location == 0);
    }

    SECTION("shaders do NOT match due to location mismatch")
    {
        // Vertex outputs at location 1, fragment expects location 0
        REQUIRE_FALSE(vert.get_details().matches(frag.get_details()));
    }
}

TEST_CASE("Fragment shader cannot match with another shader", "[shader][validation][constraints]")
{
    veng::Logger::instance().set_level(spdlog::level::warn);
    veng::VulkanContext ctx("veng::Shader Validation Test");

    auto frag1 = load_shader(ctx.device(), "tests/matching/vert_to_frag/simple.frag");
    auto frag2 = load_shader(ctx.device(), "tests/matching/vert_to_tess_to_frag/tess.frag");

    SECTION("fragment is always end of pipeline")
    {
        // Fragment shader cannot output to another shader
        REQUIRE_FALSE(frag1.get_details().matches(frag2.get_details()));
    }
}

TEST_CASE("Tessellation control must connect to tessellation evaluation", "[shader][validation][constraints]")
{
    veng::Logger::instance().set_level(spdlog::level::warn);
    veng::VulkanContext ctx("veng::Shader Validation Test");

    auto hull = load_shader(ctx.device(), "tests/matching/vert_to_tess_to_frag/tess.hull");
    auto frag = load_shader(ctx.device(), "tests/matching/vert_to_tess_to_frag/tess.frag");

    SECTION("tessellation control cannot connect directly to fragment")
    {
        // Hull shader must go through domain shader first
        REQUIRE_FALSE(hull.get_details().matches(frag.get_details()));
    }
}

TEST_CASE("Geometry shader must connect to fragment", "[shader][validation][constraints]")
{
    veng::Logger::instance().set_level(spdlog::level::warn);
    veng::VulkanContext ctx("veng::Shader Validation Test");

    auto geom = load_shader(ctx.device(), "tests/matching/vert_to_geom_to_frag/passthrough.geom");
    auto vert = load_shader(ctx.device(), "tests/matching/vert_to_geom_to_frag/passthrough.vert");

    SECTION("geometry cannot connect back to vertex")
    {
        REQUIRE_FALSE(geom.get_details().matches(vert.get_details()));
    }
}

TEST_CASE("Vertex shader can skip to fragment", "[shader][validation][skip-stages]")
{
    veng::Logger::instance().set_level(spdlog::level::warn);
    veng::VulkanContext ctx("veng::Shader Validation Test");

    auto vert = load_shader(ctx.device(), "tests/matching/vert_to_frag/simple.vert");
    auto frag = load_shader(ctx.device(), "tests/matching/vert_to_frag/simple.frag");

    SECTION("vertex can connect directly to fragment")
    {
        // Most common pipeline: just vertex and fragment
        REQUIRE(vert.get_details().matches(frag.get_details()));
    }
}

TEST_CASE("Vertex shader can skip to geometry", "[shader][validation][skip-stages]")
{
    veng::Logger::instance().set_level(spdlog::level::warn);
    veng::VulkanContext ctx("veng::Shader Validation Test");

    auto vert = load_shader(ctx.device(), "tests/matching/vert_to_geom_to_frag/passthrough.vert");
    auto geom = load_shader(ctx.device(), "tests/matching/vert_to_geom_to_frag/passthrough.geom");

    SECTION("vertex can connect to geometry shader")
    {
        REQUIRE(vert.get_details().matches(geom.get_details()));
    }
}

TEST_CASE("Tessellation evaluation can skip to fragment", "[shader][validation][skip-stages]")
{
    veng::Logger::instance().set_level(spdlog::level::warn);
    veng::VulkanContext ctx("veng::Shader Validation Test");

    auto domain = load_shader(ctx.device(), "tests/matching/vert_to_tess_to_frag/tess.domain");
    auto frag = load_shader(ctx.device(), "tests/matching/vert_to_tess_to_frag/tess.frag");

    SECTION("tessellation evaluation can connect directly to fragment")
    {
        // Can skip geometry shader
        REQUIRE(domain.get_details().matches(frag.get_details()));
    }
}
