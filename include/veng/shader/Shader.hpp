/**
 * @file
 * @author chris
 * @brief Shader loading, SPIR-V compilation via Slang, and per-stage reflection data.
 *
 * A @ref veng::Shader wraps a `vk::ShaderModule` together with the SPIR-V reflection data
 * extracted by the Slang compiler: descriptor bindings, push-constant info, and a
 * stage-specific @ref veng::ShaderDetails variant that carries vertex-input layouts,
 * inter-stage interface variables, or compute work-group sizes. The pipeline builders
 * (@ref veng::ComputePipelineBuilder, @ref veng::GraphicsPipelineBuilder) consume this data to
 * derive Vulkan create-info structures automatically.
 *
 * @ingroup shaders
 */

#ifndef VENG_SHADER_HPP
#define VENG_SHADER_HPP

#include <expected>
#include <optional>
#include <slang.h>
#include <string_view>
#include <variant>
#include <vector>
#include <veng/context/Context.hpp>

namespace veng
{
/**
 * @brief Reflected descriptor-binding metadata extracted from a compiled shader.
 * @ingroup shaders
 */
struct DescriptorInfo
{
	std::string				name;			  ///< GLSL/Slang variable name.
	std::size_t				size;			  ///< Size per descriptor element (bytes).
	std::size_t				binding;		  ///< Binding index within its set.
	std::size_t				set;			  ///< Descriptor set index.
	std::size_t				descriptor_count; ///< Array element count; 1 if not an array.
	vk::DescriptorType		type;			  ///< Vulkan descriptor type.
	vk::ShaderStageFlagBits stage;			  ///< Shader stage that declares this binding.
};

/**
 * @brief Reflected push-constant block metadata extracted from a compiled shader.
 * @ingroup shaders
 */
struct PushConstantInfo
{
	std::string			 name;	 ///< Slang variable name of the push-constant block.
	std::size_t			 offset; ///< Byte offset within the push-constant range.
	std::size_t			 size;	 ///< Size of the push-constant block in bytes.
	vk::ShaderStageFlags stage;	 ///< Stage flags that access this push-constant block.
};

/**
 * @brief A single inter-stage interface variable (varying) with its location and format.
 * @ingroup shaders
 */
struct StageVariable
{
	std::string name;	  ///< Variable name as declared in the shader.
	uint32_t	location; ///< `layout(location = N)` index.
	vk::Format	format;	  ///< Vulkan format derived from the Slang scalar/vector type.
};

// Forward declarations needed for the matches() signatures below.
struct ShaderDetails;
struct TessellationControlDetails;
struct TessellationEvaluationDetails;
struct GeometryDetails;
struct FragmentDetails;
struct ComputeDetails;

/**
 * @brief A single vertex attribute as reflected from a vertex-stage input struct.
 * @ingroup shaders
 */
struct VertexAttribute
{
	std::string name;	  ///< Field name in the Slang input struct.
	uint32_t	location; ///< Attribute location index.
	uint32_t	binding;  ///< Vertex buffer binding index (one per input parameter struct).
	uint32_t	offset;	  ///< Byte offset within the binding's stride.
	vk::Format	format;	  ///< Vulkan format derived from the Slang scalar/vector type.

	/**
	 * @brief Convert to the Vulkan attribute description used in pipeline creation.
	 * @return A `vk::VertexInputAttributeDescription` filled from this attribute's fields.
	 */
	[[nodiscard]] vk::VertexInputAttributeDescription to_attribute_description() const
	{
		return vk::VertexInputAttributeDescription()
			.setLocation(location)
			.setBinding(binding)
			.setFormat(format)
			.setOffset(offset);
	}
};

/**
 * @brief A vertex buffer binding as reflected from a vertex-stage entry-point parameter.
 * @ingroup shaders
 */
struct VertexBinding
{
	uint32_t	binding; ///< Binding index.
	uint32_t	stride;	 ///< Byte stride of one vertex element.
	std::string name;	 ///< Struct name (e.g., `"PerVertex"`, `"PerInstance"`).

	/**
	 * @brief Convert to the Vulkan binding description used in pipeline creation.
	 * @param input_rate Whether to step per vertex or per instance.
	 * @return A `vk::VertexInputBindingDescription` filled from this binding's fields.
	 */
	[[nodiscard]] vk::VertexInputBindingDescription to_binding_description(vk::VertexInputRate input_rate) const
	{
		return vk::VertexInputBindingDescription().setBinding(binding).setStride(stride).setInputRate(input_rate);
	}
};

/**
 * @brief Reflected data for a vertex shader stage.
 * @ingroup shaders
 */
struct VertexDetails
{
	std::vector<VertexAttribute> inputs;   ///< Reflected vertex attributes (from input struct fields).
	std::vector<VertexBinding>	 bindings; ///< One binding per entry-point input parameter struct.
	std::vector<StageVariable>	 outputs;  ///< Outputs passed to the next pipeline stage.

	/** @brief Extract vertex details from a linked Slang program. */
	explicit VertexDetails(slang::IComponentType* linked);
	/**
	 * @brief Check whether this stage's outputs are compatible with `next` stage's inputs.
	 * @param next The following pipeline stage's @ref veng::ShaderDetails.
	 * @return `true` if every consumer input has a matching producer output at the same location and format.
	 */
	[[nodiscard]] bool matches(const ShaderDetails& next) const;
};

/**
 * @brief Reflected data for a tessellation-control (hull) shader stage.
 * @ingroup shaders
 */
struct TessellationControlDetails
{
	std::vector<StageVariable> inputs;
	std::vector<StageVariable> outputs;
	uint32_t				   output_vertices; ///< Vertices per output patch (from `[outputcontrolpoints(N)]`).

	/** @brief Extract tessellation-control details from a linked Slang program. */
	explicit TessellationControlDetails(slang::IComponentType* linked);
	/**
	 * @brief Check compatibility with the following tessellation-evaluation stage.
	 * @param next The following pipeline stage's @ref veng::ShaderDetails.
	 * @return `true` if interfaces match.
	 */
	[[nodiscard]] bool matches(const ShaderDetails& next) const;
};

/**
 * @brief Reflected data for a tessellation-evaluation (domain) shader stage.
 * @ingroup shaders
 */
struct TessellationEvaluationDetails
{
	std::vector<StageVariable> inputs;
	std::vector<StageVariable> outputs;

	/** @brief Tessellation domain (patch shape). */
	enum class Domain
	{
		Triangles,
		Quads,
		Isolines
	};
	/** @brief Tessellation spacing mode. */
	enum class Spacing
	{
		Equal,
		FractionalEven,
		FractionalOdd
	};

	Domain	domain;	   ///< Patch domain shape.
	Spacing spacing;   ///< Tessellation level spacing mode.
	bool	clockwise; ///< Whether output triangles are clockwise.

	/** @brief Extract tessellation-evaluation details from a linked Slang program. */
	explicit TessellationEvaluationDetails(slang::IComponentType* linked);
	/**
	 * @brief Check compatibility with the following geometry or fragment stage.
	 * @param next The following pipeline stage's @ref veng::ShaderDetails.
	 * @return `true` if interfaces match.
	 */
	[[nodiscard]] bool matches(const ShaderDetails& next) const;
};

/**
 * @brief Reflected data for a geometry shader stage.
 * @ingroup shaders
 */
struct GeometryDetails
{
	std::vector<StageVariable> inputs;
	std::vector<StageVariable> outputs;

	vk::PrimitiveTopology input_primitive;	   ///< Expected input primitive type.
	vk::PrimitiveTopology output_primitive;	   ///< Emitted primitive type.
	uint32_t			  max_output_vertices; ///< Maximum vertices the shader emits per invocation.
	uint32_t			  invocations;		   ///< Number of geometry shader invocations.

	/** @brief Extract geometry-stage details from a linked Slang program. */
	explicit GeometryDetails(slang::IComponentType* linked);
	/**
	 * @brief Check compatibility with the following fragment stage.
	 * @param next The following pipeline stage's @ref veng::ShaderDetails.
	 * @return `true` if interfaces match.
	 */
	[[nodiscard]] bool matches(const ShaderDetails& next) const;
};

/**
 * @brief Reflected data for a fragment shader stage.
 * @ingroup shaders
 */
struct FragmentDetails
{
	std::vector<StageVariable> inputs;
	std::vector<StageVariable> outputs;		 ///< Color attachment outputs.
	bool					   writes_depth; ///< Whether the shader writes `SV_Depth`.

	/** @brief Extract fragment-stage details from a linked Slang program. */
	explicit FragmentDetails(slang::IComponentType* linked);
	/**
	 * @brief Always returns `false`; the fragment stage is the end of the pipeline chain.
	 * @param next Unused.
	 * @return `false`.
	 */
	[[nodiscard]] bool matches(const ShaderDetails& next) const;
};

/**
 * @brief Reflected data for a compute shader stage.
 * @ingroup shaders
 */
struct ComputeDetails
{
	uint32_t local_size_x; ///< Work-group X dimension from `[numthreads(x, y, z)]`.
	uint32_t local_size_y; ///< Work-group Y dimension.
	uint32_t local_size_z; ///< Work-group Z dimension.

	/** @brief Extract compute-stage details from a linked Slang program. */
	explicit ComputeDetails(slang::IComponentType* linked);
	/**
	 * @brief Always returns `false`; compute is a standalone stage with no successor.
	 * @param next Unused.
	 * @return `false`.
	 */
	[[nodiscard]] bool matches(const ShaderDetails& next) const;
};

/// @brief Variant base holding one of the six per-stage reflected-data types.
/// @ingroup shaders
using ShaderDetailsBase = std::variant<VertexDetails, TessellationControlDetails, TessellationEvaluationDetails,
									   GeometryDetails, FragmentDetails, ComputeDetails>;

/**
 * @brief Stage-specific reflection metadata, stored as a typed variant over all pipeline stages.
 *
 * Constructed by @ref veng::Shader::create_shader from the Slang-linked program. The pipeline
 * builders read `stage()` to enforce that the correct shader is passed, and call
 * `matches()` to verify adjacent stage interface compatibility before emitting
 * `vkCreate*Pipeline`.
 *
 * @ingroup shaders
 * @see Shader
 */
struct ShaderDetails : ShaderDetailsBase
{
	/** @brief Dispatch to the stage-specific constructor for the active variant alternative. */
	explicit ShaderDetails(slang::IComponentType* linked);
	/**
	 * @brief Check whether this stage's outputs are compatible with `next` stage's inputs.
	 * @param next The following pipeline stage's @ref veng::ShaderDetails.
	 * @return `true` if the interfaces match; `false` (with logged errors) otherwise.
	 */
	[[nodiscard]] bool matches(const ShaderDetails& next) const;
	/**
	 * @brief Return the Vulkan shader-stage flag corresponding to the active variant.
	 * @return `vk::ShaderStageFlagBits` for this stage.
	 */
	[[nodiscard]] vk::ShaderStageFlagBits stage() const;
};

/**
 * @brief RAII owner of a compiled `vk::ShaderModule` and its complete SPIR-V reflection data.
 *
 * Constructed via the `create_shader` factory, which drives the Slang compilation pipeline:
 * load → link → emit SPIR-V → reflect → create `vk::ShaderModule`. The reflection data
 * (descriptor bindings, push-constant block, stage-specific interface variables) is
 * consumed by @ref veng::ComputePipelineBuilder and @ref veng::GraphicsPipelineBuilder to build
 * Vulkan pipelines without any hand-written create-info boilerplate. Move-only; copy is
 * deleted.
 *
 * @ingroup shaders
 * @see ComputePipelineBuilder
 * @see GraphicsPipelineBuilder
 * @see ShaderDetails
 */
class Shader
{
	 public:
	/**
	 * @brief Load, compile, and reflect a Slang shader, producing a ready-to-use @ref veng::Shader.
	 *
	 * The Slang global session is lazily created on first call and persists for the
	 * application lifetime. Compilation errors and Vulkan creation failures are returned
	 * as error strings rather than throwing.
	 *
	 * @param device      The Vulkan device on which to create the `vk::ShaderModule`.
	 * @param name        The Slang module name (maps to a `.slang` file in `SHADER_DIR`).
	 * @param entry_point The entry-point function name (default: `"main"`).
	 * @return A fully constructed @ref veng::Shader, or an error string describing the failure.
	 */
	static std::expected<Shader, std::string> create_shader(vk::Device device, std::string_view name,
															std::string_view entry_point = "main");

	/// @brief Convenience overload taking the engine @ref veng::Context (so a caller need not pull out a
	///        `vk::Device`); delegates to the `vk::Device` overload with `context.device()`.
	static std::expected<Shader, std::string> create_shader(const Context& context, std::string_view name,
															std::string_view entry_point = "main");

	/// @brief RHI-vocabulary overload for past-L3 callers: a node/pass builds a shader naming only the
	///        @ref veng::rhi::Device; delegates to the `vk::Device` overload with `rhi.device()`.
	static std::expected<Shader, std::string> create_shader(rhi::Device& rhi, std::string_view name,
															std::string_view entry_point = "main");

	/** @brief All descriptor bindings reflected from this shader's SPIR-V. */
	[[nodiscard]] const std::vector<DescriptorInfo>& get_descriptor_infos() const;
	/** @brief The push-constant block, if the shader declares one; `std::nullopt` otherwise. */
	[[nodiscard]] const std::optional<PushConstantInfo>& get_push_constant_info() const;
	/** @brief The raw `vk::ShaderModule` handle. */
	[[nodiscard]] vk::ShaderModule get_shader_module() const;
	/**
	 * @brief Build a `vk::PipelineShaderStageCreateInfo` ready for pipeline creation.
	 * @return A stage create-info referencing this shader's module, stage flag, and entry point.
	 */
	[[nodiscard]] vk::PipelineShaderStageCreateInfo create_pipeline_shader_stage_create_info() const;
	/** @brief Stage-specific reflection details (vertex layout, interface variables, etc.). */
	[[nodiscard]] const ShaderDetails& get_details() const;
	/** @brief The entry-point function name used during compilation. */
	[[nodiscard]] const std::string& get_entry_point() const { return m_entry_point; }

	Shader(const Shader&)			 = delete;
	Shader& operator=(const Shader&) = delete;

	Shader(Shader&& other) noexcept;
	Shader& operator=(Shader&& other) noexcept;
	~Shader();

	 private:
	Shader(const vk::Device& m_device, const vk::ShaderModule& m_shader_module, vk::ShaderStageFlagBits m_stage,
		   ShaderDetails details, const std::vector<DescriptorInfo>& m_descriptor_infos,
		   std::optional<PushConstantInfo> push_constant_info, std::string entry_point);
	vk::Device						m_device;
	vk::ShaderModule				m_shader_module;
	vk::ShaderStageFlagBits			m_stage;
	ShaderDetails					m_details;
	std::vector<DescriptorInfo>		m_descriptor_infos;
	std::optional<PushConstantInfo> m_push_constant_info;
	std::string						m_entry_point;
};
} // namespace veng

#endif // VENG_SHADER_HPP
