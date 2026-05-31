/**
 * @file
 * @author chris
 * @brief @ref veng::Shader implementation: Slang session management, compilation, SPIR-V
 *        generation, and per-stage reflection extraction.
 * @ingroup shaders
 */
#include <map>
#include <slang-com-ptr.h>
#include <veng/common.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/shader/Shader.hpp>

namespace veng
{

// ============================================================================
// Slang Session Management
// ============================================================================
// These functions manage the global Slang compiler session. The session is
// created lazily on first use and persists for the application lifetime.
// ==========================================================================

Slang::ComPtr<slang::IGlobalSession> create_global_session()
{
	Slang::ComPtr<slang::IGlobalSession> session;
	SlangGlobalSessionDesc				 desc = {};
	createGlobalSession(&desc, session.writeRef());
	Logger::instance().debug("Created Slang global session");
	return session;
}

Slang::ComPtr<slang::ISession> create_spirv_session(slang::IGlobalSession* global)
{
	slang::SessionDesc session_desc = {};

	slang::TargetDesc target_desc = {};
	target_desc.format			  = SLANG_SPIRV;
	target_desc.profile			  = global->findProfile("spirv_1_5");
	session_desc.targets		  = &target_desc;
	session_desc.targetCount	  = 1;

	const char* search_paths[]	 = {SHADER_DIR};
	session_desc.searchPaths	 = search_paths;
	session_desc.searchPathCount = 1;

	Slang::ComPtr<slang::ISession> session;
	global->createSession(session_desc, session.writeRef());
	Logger::instance().debug("Created SPIR-V session with search path: {}", SHADER_DIR);
	return session;
}

slang::ISession* get_session()
{
	static Slang::ComPtr<slang::IGlobalSession> global	= create_global_session();
	static Slang::ComPtr<slang::ISession>		session = create_spirv_session(global);
	return session.get();
}

// ============================================================================
// Shader Compilation
// ============================================================================
// Functions for loading, linking, and compiling shaders to SPIR-V.
// ==========================================================================

/// Check Slang diagnostics blob for errors. Returns error string if present.
std::optional<std::string> check_diagnostics(slang::IBlob* diagnostics)
{
	if (!diagnostics || diagnostics->getBufferSize() == 0)
	{
		return std::nullopt;
	}
	return std::string{static_cast<const char*>(diagnostics->getBufferPointer()), diagnostics->getBufferSize()};
}

/// Load a shader module and find the specified entry point.
/// Creates a composite component type combining module + entry point.
std::expected<Slang::ComPtr<slang::IComponentType>, std::string> load_shader_program(std::string_view name,
																					 std::string_view entry_point)
{
	Logger::instance().debug("Loading shader module '{}' with entry point '{}'", name, entry_point);

	auto*						session = get_session();
	Slang::ComPtr<slang::IBlob> diagnostics;

	Slang::ComPtr<slang::IModule> module(session->loadModule(name.data(), diagnostics.writeRef()));
	if (auto error = check_diagnostics(diagnostics.get()))
	{
		Logger::instance().error("Failed to load module '{}': {}", name, *error);
		return std::unexpected{*error};
	}
	if (!module)
	{
		Logger::instance().error("Module '{}' is null", name);
		return std::unexpected{std::format("Failed to load module '{}'", name)};
	}

	Slang::ComPtr<slang::IEntryPoint> entry;
	module->findEntryPointByName(entry_point.data(), entry.writeRef());
	if (!entry)
	{
		Logger::instance().error("Entry point '{}' not found in module '{}'", entry_point, name);
		return std::unexpected{std::format("Entry point '{}' not found", entry_point)};
	}

	slang::IComponentType*				 components[] = {module, entry};
	Slang::ComPtr<slang::IComponentType> program;
	session->createCompositeComponentType(components, 2, program.writeRef(), diagnostics.writeRef());
	if (auto error = check_diagnostics(diagnostics.get()))
	{
		Logger::instance().error("Failed to create composite type: {}", *error);
		return std::unexpected{*error};
	}

	Logger::instance().trace("Successfully loaded shader program '{}':'{}'", name, entry_point);
	return program;
}

/// Link a shader program, resolving all dependencies.
std::expected<Slang::ComPtr<slang::IComponentType>, std::string> link_program(
	Slang::ComPtr<slang::IComponentType> program)
{
	Slang::ComPtr<slang::IComponentType> linked;
	Slang::ComPtr<slang::IBlob>			 diagnostics;

	program->link(linked.writeRef(), diagnostics.writeRef());
	if (auto error = check_diagnostics(diagnostics.get()))
	{
		Logger::instance().error("Failed to link program: {}", *error);
		return std::unexpected{*error};
	}

	Logger::instance().trace("Successfully linked shader program");
	return linked;
}

/// Generate SPIR-V bytecode from a linked program.
std::expected<Slang::ComPtr<slang::IBlob>, std::string> get_spirv_code(slang::IComponentType* linked)
{
	Slang::ComPtr<slang::IBlob> code;
	Slang::ComPtr<slang::IBlob> diagnostics;

	linked->getEntryPointCode(0, 0, code.writeRef(), diagnostics.writeRef());
	if (auto error = check_diagnostics(diagnostics.get()))
	{
		Logger::instance().error("Failed to get SPIR-V code: {}", *error);
		return std::unexpected{*error};
	}

	Logger::instance().trace("Generated SPIR-V code: {} bytes", code->getBufferSize());
	return code;
}

/// Create a Vulkan shader module from SPIR-V bytecode.
std::expected<vk::ShaderModule, std::string> create_shader_module(vk::Device device, slang::IBlob* spirv)
{
	auto create_info = vk::ShaderModuleCreateInfo()
						   .setCodeSize(spirv->getBufferSize())
						   .setPCode(static_cast<const uint32_t*>(spirv->getBufferPointer()));

	auto module_res = device.createShaderModule(create_info);
	if (module_res.result != vk::Result::eSuccess)
	{
		Logger::instance().error("Failed to create Shader Module {}", to_string(module_res.result));
		return std::unexpected{std::format("vkCreateShaderModule failed: {}", to_string(module_res.result))};
	}
	Logger::instance().debug("Created shader module ({} bytes)", spirv->getBufferSize());
	return module_res.value;
}

// ============================================================================
// Type Conversion Utilities
// ============================================================================
// Mappings between Slang reflection types and Vulkan types.
// ==========================================================================

/// Convert Slang scalar/vector type to Vulkan format.
/// Supports float, int, uint in 1-4 component vectors.
vk::Format to_vk_format(slang::TypeReflection* type)
{
	auto scalar = type->getScalarType();
	auto count	= type->getElementCount();
	if (count == 0)
		count = 1;

	using ST = slang::TypeReflection::ScalarType;

	if (scalar == ST::Float32)
	{
		constexpr vk::Format formats[] = {vk::Format::eR32Sfloat, vk::Format::eR32G32Sfloat,
										  vk::Format::eR32G32B32Sfloat, vk::Format::eR32G32B32A32Sfloat};
		if (count <= 4)
			return formats[count - 1];
	}
	if (scalar == ST::Int32)
	{
		constexpr vk::Format formats[] = {vk::Format::eR32Sint, vk::Format::eR32G32Sint, vk::Format::eR32G32B32Sint,
										  vk::Format::eR32G32B32A32Sint};
		if (count <= 4)
			return formats[count - 1];
	}
	if (scalar == ST::UInt32)
	{
		constexpr vk::Format formats[] = {vk::Format::eR32Uint, vk::Format::eR32G32Uint, vk::Format::eR32G32B32Uint,
										  vk::Format::eR32G32B32A32Uint};
		if (count <= 4)
			return formats[count - 1];
	}

	Logger::instance().warn("Unknown vertex format, defaulting to R32G32B32A32Sfloat");
	return vk::Format::eR32G32B32A32Sfloat;
}

/// Convert Slang shader stage to Vulkan shader stage flag.
vk::ShaderStageFlagBits to_vk_shader_stage(SlangStage stage)
{
	switch (stage)
	{
		case SLANG_STAGE_VERTEX: return vk::ShaderStageFlagBits::eVertex;
		case SLANG_STAGE_FRAGMENT: return vk::ShaderStageFlagBits::eFragment;
		case SLANG_STAGE_GEOMETRY: return vk::ShaderStageFlagBits::eGeometry;
		case SLANG_STAGE_HULL: return vk::ShaderStageFlagBits::eTessellationControl;
		case SLANG_STAGE_DOMAIN: return vk::ShaderStageFlagBits::eTessellationEvaluation;
		case SLANG_STAGE_COMPUTE: return vk::ShaderStageFlagBits::eCompute;
		case SLANG_STAGE_RAY_GENERATION: return vk::ShaderStageFlagBits::eRaygenKHR;
		case SLANG_STAGE_INTERSECTION: return vk::ShaderStageFlagBits::eIntersectionKHR;
		case SLANG_STAGE_ANY_HIT: return vk::ShaderStageFlagBits::eAnyHitKHR;
		case SLANG_STAGE_CLOSEST_HIT: return vk::ShaderStageFlagBits::eClosestHitKHR;
		case SLANG_STAGE_MISS: return vk::ShaderStageFlagBits::eMissKHR;
		case SLANG_STAGE_CALLABLE: return vk::ShaderStageFlagBits::eCallableKHR;
		case SLANG_STAGE_MESH: return vk::ShaderStageFlagBits::eMeshEXT;
		case SLANG_STAGE_AMPLIFICATION: return vk::ShaderStageFlagBits::eTaskEXT;
		default:
			Logger::instance().warn("Unknown shader stage: {}", static_cast<int>(stage));
			return vk::ShaderStageFlagBits::eVertex;
	}
}

/// Convert Slang binding type to human-readable string (for logging).
std::string_view to_string(slang::BindingType binding_type)
{
	using enum slang::BindingType;
	auto base_type =
		static_cast<slang::BindingType>(static_cast<uint32_t>(binding_type) & static_cast<uint32_t>(BaseMask));

	switch (base_type)
	{
		case Sampler: return "Sampler";
		case Texture: return "Texture";
		case ConstantBuffer: return "ConstantBuffer";
		case TypedBuffer: return "TypedBuffer";
		case RawBuffer: return "RawBuffer";
		case CombinedTextureSampler: return "CombinedTextureSampler";
		case InputRenderTarget: return "InputRenderTarget";
		case InlineUniformData: return "InlineUniformData";
		case RayTracingAccelerationStructure: return "AccelerationStructure";
		case ParameterBlock: return "ParameterBlock";
		case VaryingInput: return "VaryingInput";
		case VaryingOutput: return "VaryingOutput";
		case PushConstant: return "PushConstant";
		default: return "Unknown";
	}
}

/// Convert Slang binding type to Vulkan descriptor type.
vk::DescriptorType to_vk_descriptor_type(slang::BindingType binding_type)
{
	using enum slang::BindingType;

	auto base_type =
		static_cast<slang::BindingType>(static_cast<uint32_t>(binding_type) & static_cast<uint32_t>(BaseMask));
	bool is_mutable = (static_cast<uint32_t>(binding_type) & static_cast<uint32_t>(MutableFlag)) != 0;

	switch (base_type)
	{
		case Sampler: return vk::DescriptorType::eSampler;
		case Texture: return is_mutable ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
		case ConstantBuffer: return vk::DescriptorType::eUniformBuffer;
		case TypedBuffer:
			return is_mutable ? vk::DescriptorType::eStorageTexelBuffer : vk::DescriptorType::eUniformTexelBuffer;
		case RawBuffer: return vk::DescriptorType::eStorageBuffer;
		case CombinedTextureSampler: return vk::DescriptorType::eCombinedImageSampler;
		case InputRenderTarget: return vk::DescriptorType::eInputAttachment;
		case InlineUniformData: return vk::DescriptorType::eInlineUniformBlock;
		case RayTracingAccelerationStructure: return vk::DescriptorType::eAccelerationStructureKHR;
		default:
			Logger::instance().warn("Unhandled binding type: {}", to_string(binding_type));
			return vk::DescriptorType::eUniformBuffer;
	}
}

// ============================================================================
// Descriptor & Push Constant Extraction
// ============================================================================
// Extract descriptor set layouts and push constant ranges from shader
// reflection. Uses TypeLayoutReflection for accurate binding information.
// ==========================================================================

/// Recursively extract size from a type layout, unwrapping arrays/structs.
std::size_t extract_size(slang::TypeLayoutReflection* type_layout)
{
	auto size = type_layout->getSize();
	if (size > 0)
	{
		return size;
	}

	auto* element_type = type_layout->getElementTypeLayout();
	if (element_type && element_type != type_layout)
	{
		return extract_size(element_type);
	}

	return 0;
}

/// Extract all descriptor bindings from a single shader parameter.
std::vector<DescriptorInfo> extract_bindings(slang::VariableLayoutReflection* param, vk::ShaderStageFlagBits stage)
{
	std::vector<DescriptorInfo> out;

	auto*		type_layout	 = param->getTypeLayout();
	const char* name		 = param->getName();
	uint32_t	base_binding = param->getBindingIndex();
	uint32_t	set			 = param->getBindingSpace();

	auto binding_range_count = type_layout->getBindingRangeCount();
	if (binding_range_count == 0)
	{
		Logger::instance().trace("  Parameter '{}': no binding ranges (push constant or varying)", name);
		return out;
	}

	for (unsigned r = 0; r < binding_range_count; r++)
	{
		auto binding_type = type_layout->getBindingRangeType(r);

		// Skip non-descriptor types
		if (binding_type == slang::BindingType::VaryingInput || binding_type == slang::BindingType::VaryingOutput ||
			binding_type == slang::BindingType::PushConstant)
		{
			continue;
		}

		uint32_t actual_binding = base_binding + r;
		auto	 count			= type_layout->getBindingRangeBindingCount(r);

		auto*		leaf_type = type_layout->getBindingRangeLeafTypeLayout(r);
		std::size_t size	  = leaf_type ? extract_size(leaf_type) : 0;

		auto vk_type = to_vk_descriptor_type(binding_type);

		Logger::instance().trace("  Binding: set={} binding={} name='{}' type={} count={} size={}", set, actual_binding,
								 name, to_string(binding_type), count, size);

		out.emplace_back(name, size, actual_binding, set, count, vk_type, stage);
	}

	return out;
}

/// Extract all descriptor bindings from a linked shader program.
std::vector<DescriptorInfo> extract_descriptors(slang::IComponentType* linked, vk::ShaderStageFlagBits stage)
{
	slang::ProgramLayout*		layout = linked->getLayout();
	std::vector<DescriptorInfo> descriptors;

	Logger::instance().debug("Extracting descriptors ({} parameters)", layout->getParameterCount());

	for (unsigned i = 0; i < layout->getParameterCount(); i++)
	{
		auto extracted = extract_bindings(layout->getParameterByIndex(i), stage);
		descriptors.append_range(std::move(extracted));
	}

	Logger::instance().debug("Extracted {} descriptor bindings", descriptors.size());
	return descriptors;
}

/// Extract push constant information from a linked shader program.
/// Returns nullopt if no push constants are declared.
std::optional<PushConstantInfo> extract_push_constants(slang::IComponentType* linked, vk::ShaderStageFlagBits stage)
{
	slang::ProgramLayout* layout = linked->getLayout();

	for (unsigned i = 0; i < layout->getParameterCount(); i++)
	{
		auto* param		  = layout->getParameterByIndex(i);
		auto* type_layout = param->getTypeLayout();

		for (unsigned r = 0; r < type_layout->getBindingRangeCount(); r++)
		{
			if (type_layout->getBindingRangeType(r) == slang::BindingType::PushConstant)
			{
				auto offset = param->getOffset();
				auto size	= extract_size(type_layout);

				Logger::instance().debug("Push constant '{}': offset={} size={}", param->getName(), offset, size);

				return PushConstantInfo{.name = param->getName(), .offset = offset, .size = size, .stage = stage};
			}
		}
	}

	Logger::instance().trace("No push constants found");
	return std::nullopt;
}

/// Extract shader stage from entry point reflection.
vk::ShaderStageFlagBits extract_shader_stage(slang::IComponentType* linked)
{
	slang::ProgramLayout* layout = linked->getLayout();

	if (layout->getEntryPointCount() == 0)
	{
		Logger::instance().warn("No entry points found, defaulting to vertex stage");
		return vk::ShaderStageFlagBits::eVertex;
	}

	auto* entry_point = layout->getEntryPointByIndex(0);
	auto  stage		  = entry_point->getStage();
	auto  vk_stage	  = to_vk_shader_stage(stage);

	Logger::instance().debug("Shader stage: {} -> {}", static_cast<int>(stage), vk::to_string(vk_stage));
	return vk_stage;
}

// ============================================================================
// Stage Variable Extraction (Inputs/Outputs)
// ============================================================================
// Extract shader interface variables (varyings) for pipeline stage matching.
//
// Uses TypeLayoutReflection uniformly for all types including streams,
// thanks to patched Slang that supports getElementTypeLayout() on streams.
//
// Location is determined from binding index. System values (SV_Position,
// SV_Target, etc.) are skipped since they map to Vulkan builtins.
// ==========================================================================

/// Get the entry point from a linked program, or nullptr if none.
slang::EntryPointReflection* get_entry_point(slang::IComponentType* linked)
{
	auto* layout = linked->getLayout();
	if (layout->getEntryPointCount() == 0)
	{
		return nullptr;
	}
	return layout->getEntryPointByIndex(0);
}

/// Recursively unwrap container types (arrays, streams, patches) to find
/// the underlying struct type layout. Returns nullptr if no struct found.
slang::TypeLayoutReflection* unwrap_to_struct(slang::TypeLayoutReflection* type)
{
	if (!type)
		return nullptr;
	if (type->getKind() == slang::TypeReflection::Kind::Struct)
		return type;
	if (auto* element = type->getElementTypeLayout(); element && element != type)
	{
		return unwrap_to_struct(element);
	}
	return nullptr;
}

/// Check if a type is a geometry shader output stream.
bool is_output_stream(slang::TypeLayoutReflection* type)
{
	if (type->getKind() != slang::TypeReflection::Kind::OutputStream)
	{
		return false;
	}
	return true;
}

/// Extract stage variables from a struct TypeLayoutReflection.
/// Uses semantic info to skip system values and getBindingIndex() for locations.
std::vector<StageVariable> extract_variables(slang::TypeLayoutReflection* struct_type)
{
	std::vector<StageVariable> vars;

	for (unsigned f = 0; f < struct_type->getFieldCount(); f++)
	{
		auto* field = struct_type->getFieldByIndex(f);

		// Skip system values (SV_Position, SV_Target, etc.)
		if (auto* semantic = field->getSemanticName())
		{
			if (std::string_view(semantic).starts_with("SV_"))
			{
				continue;
			}
		}

		vars.push_back({.name	  = field->getName(),
						.location = field->getBindingIndex(),
						.format	  = to_vk_format(field->getTypeLayout()->getType())});
	}

	return vars;
}

/// Extract all input variables from shader entry point parameters.
std::vector<StageVariable> extract_inputs(slang::EntryPointReflection* entry)
{
	std::vector<StageVariable> inputs;

	for (unsigned p = 0; p < entry->getParameterCount(); p++)
	{
		auto* param		  = entry->getParameterByIndex(p);
		auto* type_layout = param->getTypeLayout();

		// Skip output streams
		if (is_output_stream(type_layout))
			continue;

		if (auto* struct_type = unwrap_to_struct(type_layout))
		{
			auto vars = extract_variables(struct_type);
			inputs.insert(inputs.end(), vars.begin(), vars.end());
		}
	}

	return inputs;
}

/// Extract all output variables from shader entry point.
/// Handles both return values (vertex/fragment) and stream params (geometry).
std::vector<StageVariable> extract_outputs(slang::EntryPointReflection* entry)
{
	std::vector<StageVariable> outputs;

	// Return value path (vertex, fragment, tessellation evaluation)
	if (auto* result = entry->getResultVarLayout())
	{
		if (auto* struct_type = unwrap_to_struct(result->getTypeLayout()))
		{
			outputs = extract_variables(struct_type);
		}
	}

	// Stream parameter path (geometry shaders)
	for (unsigned p = 0; p < entry->getParameterCount(); p++)
	{
		auto* param		  = entry->getParameterByIndex(p);
		auto* type_layout = param->getTypeLayout();

		if (is_output_stream(type_layout))
		{
			// getElementTypeLayout() now works for streams (patched Slang)
			if (auto* struct_type = unwrap_to_struct(type_layout))
			{
				auto vars = extract_variables(struct_type);
				outputs.insert(outputs.end(), vars.begin(), vars.end());

				Logger::instance().trace("Extracted {} output variables from stream parameter '{}'", vars.size(),
										 param->getName());
			}
		}
	}

	return outputs;
}

/// Check if producer outputs match consumer inputs by location and format.
/// Logs detailed errors for any mismatches.
bool interfaces_match(const std::vector<StageVariable>& producer, const std::vector<StageVariable>& consumer,
					  std::string_view producer_name, std::string_view consumer_name)
{
	bool valid = true;

	std::map<uint32_t, const StageVariable*> producer_map;
	for (const auto& var : producer)
	{
		producer_map[var.location] = &var;
	}

	for (const auto& input : consumer)
	{
		auto it = producer_map.find(input.location);

		if (it == producer_map.end())
		{
			Logger::instance().error("{} input '{}' at location {} has no matching {} output", consumer_name,
									 input.name, input.location, producer_name);
			valid = false;
			continue;
		}

		if (it->second->format != input.format)
		{
			Logger::instance().error("Location {}: {} outputs {} but {} expects {}", input.location, producer_name,
									 vk::to_string(it->second->format), consumer_name, vk::to_string(input.format));
			valid = false;
		}
	}

	return valid;
}

// ============================================================================
// Shader Stage Details
// ============================================================================
// Per-stage metadata structures. Each stage type extracts its inputs/outputs
// and validates connections to the next pipeline stage.
// ============================================================================

// --- VertexDetails ---
uint32_t format_size(vk::Format format)
{
	switch (format)
	{
		case vk::Format::eR32Sfloat:
		case vk::Format::eR32Sint:
		case vk::Format::eR32Uint: return 4;
		case vk::Format::eR32G32Sfloat:
		case vk::Format::eR32G32Sint:
		case vk::Format::eR32G32Uint: return 8;
		case vk::Format::eR32G32B32Sfloat:
		case vk::Format::eR32G32B32Sint:
		case vk::Format::eR32G32B32Uint: return 12;
		case vk::Format::eR32G32B32A32Sfloat:
		case vk::Format::eR32G32B32A32Sint:
		case vk::Format::eR32G32B32A32Uint: return 16;
		default: return 16;
	}
}
/// Extract vertex inputs with binding information.
/// Each entry point parameter becomes a separate binding.
std::pair<std::vector<VertexAttribute>, std::vector<VertexBinding>> extract_vertex_inputs(
	slang::EntryPointReflection* entry)
{
	std::vector<VertexAttribute> attributes;
	std::vector<VertexBinding>	 bindings;

	for (uint32_t param_idx = 0; param_idx < entry->getParameterCount(); ++param_idx)
	{
		auto* param		  = entry->getParameterByIndex(param_idx);
		auto* type_layout = param->getTypeLayout();

		// Skip non-struct parameters (SV_VertexID, SV_InstanceID are scalars)
		if (type_layout->getKind() != slang::TypeReflection::Kind::Struct)
		{
			continue;
		}

		const char* struct_name = type_layout->getType()->getName();
		if (!struct_name || struct_name[0] == '\0')
		{
			Logger::instance().error("Vertex input struct at parameter {} has no name. "
									 "This is a reflection bug.",
									 param_idx);
			throw std::runtime_error("Vertex input struct has no name");
		}

		uint32_t offset = 0;

		for (uint32_t f = 0; f < type_layout->getFieldCount(); ++f)
		{
			auto* field	 = type_layout->getFieldByIndex(f);
			auto  format = to_vk_format(field->getTypeLayout()->getType());
			auto  size	 = format_size(format);

			attributes.push_back({.name		= field->getName(),
								  .location = field->getBindingIndex(),
								  .binding	= param_idx,
								  .offset	= offset,
								  .format	= format});

			offset += size;
		}

		bindings.push_back({.binding = param_idx, .stride = offset, .name = struct_name});
	}

	return {attributes, bindings};
}

VertexDetails::VertexDetails(slang::IComponentType* linked)
{
	auto* entry = get_entry_point(linked);
	if (!entry)
		return;

	std::tie(inputs, bindings) = extract_vertex_inputs(entry);
	outputs					   = extract_outputs(entry);

	Logger::instance().debug("VertexDetails: {} inputs across {} bindings, {} outputs", inputs.size(), bindings.size(),
							 outputs.size());

	for (const auto& b : bindings)
	{
		Logger::instance().trace("  Binding {}: stride={}", b.binding, b.stride);
	}
}

bool VertexDetails::matches(const ShaderDetails& next) const
{
	return std::visit(overloaded{[this](const TessellationControlDetails& tc)
								 { return interfaces_match(outputs, tc.inputs, "vertex", "tess_control"); },
								 [this](const GeometryDetails& g)
								 { return interfaces_match(outputs, g.inputs, "vertex", "geometry"); },
								 [this](const FragmentDetails& f)
								 { return interfaces_match(outputs, f.inputs, "vertex", "fragment"); },
								 [](const auto&)
								 {
									 Logger::instance().error("Invalid pipeline: vertex cannot connect to this stage");
									 return false;
								 }},
					  next);
}

// --- TessellationControlDetails ---

TessellationControlDetails::TessellationControlDetails(slang::IComponentType* linked)
	: output_vertices(0)
{
	auto* entry = get_entry_point(linked);
	if (!entry)
		return;

	inputs	= extract_inputs(entry);
	outputs = extract_outputs(entry);

	// Note: output_vertices cannot be extracted via Slang reflection (Issue #613)
	// Must be specified via [outputcontrolpoints(N)] attribute in shader

	Logger::instance().debug("TessControlDetails: {} inputs, {} outputs, output_vertices={}", inputs.size(),
							 outputs.size(), output_vertices);
}

bool TessellationControlDetails::matches(const ShaderDetails& next) const
{
	return std::visit(overloaded{[this](const TessellationEvaluationDetails& te)
								 { return interfaces_match(outputs, te.inputs, "tess_control", "tess_eval"); },
								 [](const auto&)
								 {
									 Logger::instance().error(
										 "Invalid pipeline: tess_control must connect to tess_eval");
									 return false;
								 }},
					  next);
}

// --- TessellationEvaluationDetails ---

TessellationEvaluationDetails::TessellationEvaluationDetails(slang::IComponentType* linked)
	: domain(Domain::Triangles)
	, spacing(Spacing::Equal)
	, clockwise(false)
{
	auto* entry = get_entry_point(linked);
	if (!entry)
		return;

	inputs	= extract_inputs(entry);
	outputs = extract_outputs(entry);

	// Note: domain/spacing/winding cannot be extracted via Slang reflection
	// Must be specified via [domain], [partitioning], [outputtopology] attributes

	Logger::instance().debug("TessEvalDetails: {} inputs, {} outputs", inputs.size(), outputs.size());
}

bool TessellationEvaluationDetails::matches(const ShaderDetails& next) const
{
	return std::visit(
		overloaded{
			[this](const GeometryDetails& g) { return interfaces_match(outputs, g.inputs, "tess_eval", "geometry"); },
			[this](const FragmentDetails& f) { return interfaces_match(outputs, f.inputs, "tess_eval", "fragment"); },
			[](const auto&)
			{
				Logger::instance().error("Invalid pipeline: tess_eval must connect to geometry or fragment");
				return false;
			}},
		next);
}

// --- GeometryDetails ---

GeometryDetails::GeometryDetails(slang::IComponentType* linked)
	: input_primitive(vk::PrimitiveTopology::eTriangleList)
	, output_primitive(vk::PrimitiveTopology::eTriangleStrip)
	, max_output_vertices(0)
	, invocations(1)
{
	auto* entry = get_entry_point(linked);
	if (!entry)
		return;

	inputs	= extract_inputs(entry);
	outputs = extract_outputs(entry);

	// Note: primitive topology and max vertices cannot be extracted via Slang reflection
	// Must be specified via [maxvertexcount(N)] attribute, input from parameter type

	Logger::instance().debug("GeometryDetails: {} inputs, {} outputs, max_vertices={}", inputs.size(), outputs.size(),
							 max_output_vertices);
}

bool GeometryDetails::matches(const ShaderDetails& next) const
{
	return std::visit(overloaded{[this](const FragmentDetails& f)
								 { return interfaces_match(outputs, f.inputs, "geometry", "fragment"); },
								 [](const auto&)
								 {
									 Logger::instance().error("Invalid pipeline: geometry must connect to fragment");
									 return false;
								 }},
					  next);
}

// --- FragmentDetails ---

FragmentDetails::FragmentDetails(slang::IComponentType* linked)
	: writes_depth(false)
{
	auto* entry = get_entry_point(linked);
	if (!entry)
		return;

	inputs	= extract_inputs(entry);
	outputs = extract_outputs(entry);

	// TODO: Check if shader writes SV_Depth

	Logger::instance().debug("FragmentDetails: {} inputs, {} outputs", inputs.size(), outputs.size());
}

bool FragmentDetails::matches(const ShaderDetails&) const
{
	Logger::instance().error("Invalid pipeline: fragment is the final stage");
	return false;
}

// --- ComputeDetails ---

ComputeDetails::ComputeDetails(slang::IComponentType* linked)
	: local_size_x(1)
	, local_size_y(1)
	, local_size_z(1)
{
	auto* entry = get_entry_point(linked);
	if (!entry)
		return;

	// Try to extract local size from attributes
	// Slang uses [numthreads(x, y, z)] attribute
	// The values will be in the compiled SPIR-V, but for now use defaults
	// TODO: Extract from entry point attributes if Slang API provides access

	Logger::instance().debug("ComputeDetails: local_size({}, {}, {})", local_size_x, local_size_y, local_size_z);
}

bool ComputeDetails::matches(const ShaderDetails&) const
{
	Logger::instance().error("Invalid pipeline: compute is a standalone stage");
	return false;
}

// --- ShaderDetails (variant wrapper) -

ShaderDetailsBase make_shader_details(slang::IComponentType* linked)
{
	auto* entry = get_entry_point(linked);
	if (!entry)
	{
		Logger::instance().error("No entry point found");
		throw std::invalid_argument("No entry point found");
	}

	switch (entry->getStage())
	{
		case SLANG_STAGE_VERTEX: return VertexDetails(linked);
		case SLANG_STAGE_HULL: return TessellationControlDetails(linked);
		case SLANG_STAGE_DOMAIN: return TessellationEvaluationDetails(linked);
		case SLANG_STAGE_GEOMETRY: return GeometryDetails(linked);
		case SLANG_STAGE_FRAGMENT: return FragmentDetails(linked);
		case SLANG_STAGE_COMPUTE: return ComputeDetails(linked);
		default:
			Logger::instance().error("Unsupported shader stage: {}", static_cast<int>(entry->getStage()));
			throw std::invalid_argument("Unsupported shader stage");
	}
}

ShaderDetails::ShaderDetails(slang::IComponentType* linked)
	: ShaderDetailsBase(make_shader_details(linked))
{
}

bool ShaderDetails::matches(const ShaderDetails& next) const
{
	return std::visit([&next](const auto& self) { return self.matches(next); }, *this);
}

vk::ShaderStageFlagBits ShaderDetails::stage() const
{
	return std::visit(overloaded{[](const VertexDetails&) { return vk::ShaderStageFlagBits::eVertex; },
								 [](const TessellationControlDetails&)
								 { return vk::ShaderStageFlagBits::eTessellationControl; },
								 [](const TessellationEvaluationDetails&)
								 { return vk::ShaderStageFlagBits::eTessellationEvaluation; },
								 [](const GeometryDetails&) { return vk::ShaderStageFlagBits::eGeometry; },
								 [](const FragmentDetails&) { return vk::ShaderStageFlagBits::eFragment; },
								 [](const ComputeDetails&) { return vk::ShaderStageFlagBits::eCompute; }},
					  *this);
}

// ============================================================================
// Shader Class
// ============================================================================
// Main shader abstraction. Loads, compiles, and extracts reflection data.
// ============================================================================

std::expected<Shader, std::string> Shader::create_shader(vk::Device device, std::string_view name,
														 std::string_view entry_point)
{
	Logger::instance().info("Creating shader '{}':'{}'", name, entry_point);

	auto linked =
		load_shader_program(name, entry_point).and_then([](auto prog) { return link_program(std::move(prog)); });

	if (!linked)
	{
		return std::unexpected{linked.error()};
	}

	auto spirv = get_spirv_code(linked->get());
	if (!spirv)
	{
		return std::unexpected{spirv.error()};
	}

	auto stage			= extract_shader_stage(linked->get());
	auto details		= ShaderDetails{linked->get()};
	auto push_constants = extract_push_constants(linked->get(), stage);
	auto descriptors	= extract_descriptors(linked->get(), stage);

	// Propagate module-creation failure instead of wrapping a null handle in a
	// "successful" Shader that would later crash pipeline creation (M4).
	auto module = create_shader_module(device, spirv->get());
	if (!module)
	{
		return std::unexpected{module.error()};
	}

	Logger::instance().info("Shader '{}' created successfully ({} descriptors)", name, descriptors.size());

	return Shader{device, *module, stage, details, descriptors, push_constants, std::string{entry_point}};
}

const std::vector<DescriptorInfo>& Shader::get_descriptor_infos() const
{
	return m_descriptor_infos;
}

const std::optional<PushConstantInfo>& Shader::get_push_constant_info() const
{
	return m_push_constant_info;
}

vk::ShaderModule Shader::get_shader_module() const
{
	return m_shader_module;
}

vk::PipelineShaderStageCreateInfo Shader::create_pipeline_shader_stage_create_info() const
{
	return vk::PipelineShaderStageCreateInfo{}
		.setStage(m_stage)
		.setModule(m_shader_module)
		.setPName(m_entry_point.c_str());
}

const ShaderDetails& Shader::get_details() const
{
	return m_details;
}

Shader::~Shader()
{
	if (m_shader_module)
	{
		m_device.destroyShaderModule(m_shader_module);
		Logger::instance().trace("Destroyed shader module");
	}
}

Shader::Shader(Shader&& other) noexcept
	: m_device(other.m_device)
	, m_shader_module(std::exchange(other.m_shader_module, nullptr))
	, m_stage(other.m_stage)
	, m_details(std::move(other.m_details))
	, m_descriptor_infos(std::move(other.m_descriptor_infos))
	, m_push_constant_info(std::move(other.m_push_constant_info))
	, m_entry_point(std::move(other.m_entry_point))
{
}

Shader& Shader::operator=(Shader&& other) noexcept
{
	if (this != &other)
	{
		if (m_shader_module)
		{
			m_device.destroyShaderModule(m_shader_module);
		}

		m_device			 = other.m_device;
		m_shader_module		 = std::exchange(other.m_shader_module, nullptr);
		m_stage				 = other.m_stage;
		m_details			 = std::move(other.m_details);
		m_descriptor_infos	 = std::move(other.m_descriptor_infos);
		m_push_constant_info = std::move(other.m_push_constant_info);
		m_entry_point		 = std::move(other.m_entry_point);
	}
	return *this;
}

Shader::Shader(const vk::Device& device, const vk::ShaderModule& shader_module, vk::ShaderStageFlagBits stage,
			   ShaderDetails details, const std::vector<DescriptorInfo>& descriptor_infos,
			   std::optional<PushConstantInfo> push_constant_info, std::string entry_point)
	: m_device(device)
	, m_shader_module(shader_module)
	, m_stage(stage)
	, m_details(std::move(details))
	, m_descriptor_infos(descriptor_infos)
	, m_push_constant_info(std::move(push_constant_info))
	, m_entry_point(std::move(entry_point))
{
}
} // namespace veng
