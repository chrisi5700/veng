//
// Created by chris on 1/22/26.
//

#ifndef VENG_SHADER_HPP
#define VENG_SHADER_HPP
//
// Created by chris on 1/7/26.
//

#include <expected>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>
#include <slang.h>
#include <veng/context/Context.hpp>

namespace veng
{
struct DescriptorInfo
{
	std::string name;
	std::size_t size; // Size per descriptor
	std::size_t binding; // Binding ID
	std::size_t set; // Set ID
	std::size_t descriptor_count; // 1 if not array
	vk::DescriptorType type;
	vk::ShaderStageFlagBits stage;
};
struct PushConstantInfo {
	std::string name;
	std::size_t offset;
	std::size_t size;
	vk::ShaderStageFlags stage;
};


struct StageVariable {
    std::string name;
    uint32_t location;
    vk::Format format;
};

// Forward declarations
struct ShaderDetails;
struct TessellationControlDetails;
struct TessellationEvaluationDetails;
struct GeometryDetails;
struct FragmentDetails;
struct ComputeDetails;

struct VertexAttribute {
	std::string name;
	uint32_t location;
	uint32_t binding;
	uint32_t offset;      // Within binding's stride
	vk::Format format;

	[[nodiscard]] vk::VertexInputAttributeDescription to_attribute_description() const {
		return vk::VertexInputAttributeDescription()
			.setLocation(location)
			.setBinding(binding)
			.setFormat(format)
			.setOffset(offset);
	}
};

struct VertexBinding {
	uint32_t binding;
	uint32_t stride;
	std::string name;  // Struct name (e.g., "PerVertex", "PerInstance")

	[[nodiscard]] vk::VertexInputBindingDescription to_binding_description(
		vk::VertexInputRate input_rate) const
	{
		return vk::VertexInputBindingDescription()
			.setBinding(binding)
			.setStride(stride)
			.setInputRate(input_rate);
	}
};

struct VertexDetails {
	std::vector<VertexAttribute> inputs;
	std::vector<VertexBinding> bindings;
	std::vector<StageVariable> outputs;  // To next stage (unchanged)

	explicit VertexDetails(slang::IComponentType* linked);
	[[nodiscard]] bool matches(const ShaderDetails& next) const;
};

struct TessellationControlDetails {
    std::vector<StageVariable> inputs;
    std::vector<StageVariable> outputs;
    uint32_t output_vertices;  // Vertices per patch

    explicit TessellationControlDetails(slang::IComponentType* linked);
    [[nodiscard]] bool matches(const ShaderDetails& next) const;
};

struct TessellationEvaluationDetails {
    std::vector<StageVariable> inputs;
    std::vector<StageVariable> outputs;

    enum class Domain { Triangles, Quads, Isolines };
    enum class Spacing { Equal, FractionalEven, FractionalOdd };

    Domain domain;
    Spacing spacing;
    bool clockwise;

    explicit TessellationEvaluationDetails(slang::IComponentType* linked);
    [[nodiscard]] bool matches(const ShaderDetails& next) const;
};

struct GeometryDetails {
    std::vector<StageVariable> inputs;
    std::vector<StageVariable> outputs;

    vk::PrimitiveTopology input_primitive;
    vk::PrimitiveTopology output_primitive;
    uint32_t max_output_vertices;
    uint32_t invocations;

    explicit GeometryDetails(slang::IComponentType* linked);
    [[nodiscard]] bool matches(const ShaderDetails& next) const;
};

struct FragmentDetails {
    std::vector<StageVariable> inputs;
    std::vector<StageVariable> outputs;  // Color attachments
    bool writes_depth;

    explicit FragmentDetails(slang::IComponentType* linked);
    [[nodiscard]] bool matches(const ShaderDetails& next) const;  // Always false, end of chain
};

struct ComputeDetails {
    uint32_t local_size_x;
    uint32_t local_size_y;
    uint32_t local_size_z;

    explicit ComputeDetails(slang::IComponentType* linked);
    [[nodiscard]] bool matches(const ShaderDetails& next) const;  // Always false, standalone stage
};

using ShaderDetailsBase = std::variant<
	VertexDetails,
	TessellationControlDetails,
	TessellationEvaluationDetails,
	GeometryDetails,
	FragmentDetails,
	ComputeDetails>;

struct ShaderDetails : ShaderDetailsBase
{

    explicit ShaderDetails(slang::IComponentType* linked);
    [[nodiscard]] bool matches(const ShaderDetails& next) const;
    [[nodiscard]] vk::ShaderStageFlagBits stage() const;
};

class Shader
{
public:

	static std::expected<Shader, std::string> create_shader(vk::Device device, std::string_view name, std::string_view entry_point = "main");

	[[nodiscard]] const std::vector<DescriptorInfo>& get_descriptor_infos() const;
	[[nodiscard]] const std::optional<PushConstantInfo>& get_push_constant_info() const;
	[[nodiscard]] vk::ShaderModule get_shader_module() const;
	[[nodiscard]] vk::PipelineShaderStageCreateInfo create_pipeline_shader_stage_create_info() const;
	[[nodiscard]] const ShaderDetails& get_details() const;
	[[nodiscard]] const std::string& get_entry_point() const { return m_entry_point; }

	auto operator<=>(const Shader& other) const
	{
		return m_stage <=> other.m_stage;
	}

	// Delete copy operations
	Shader(const Shader&) = delete;
	Shader& operator=(const Shader&) = delete;

	// Move operations
	Shader(Shader&& other) noexcept;
	Shader& operator=(Shader&& other) noexcept;
	~Shader();

private:
	Shader(const vk::Device& m_device, const vk::ShaderModule& m_shader_module, vk::ShaderStageFlagBits m_stage, ShaderDetails details,
		   const std::vector<DescriptorInfo>& m_descriptor_infos, std::optional<PushConstantInfo> push_constant_info, std::string entry_point);
	vk::Device m_device;
	vk::ShaderModule m_shader_module;
	vk::ShaderStageFlagBits m_stage;
	ShaderDetails m_details;
	std::vector<DescriptorInfo> m_descriptor_infos;
	std::optional<PushConstantInfo> m_push_constant_info;
	std::string m_entry_point;

};
}


#endif // VENG_SHADER_HPP
