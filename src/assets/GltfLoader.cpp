/**
 * @file
 * @author chris
 * @brief @ref veng::assets::load_gltf implementation: fastgltf parsing, mesh/texture upload, and
 *        render-graph wiring for glTF 2.0 scenes.
 * @ingroup assets
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <variant>
#include <vector>
#include <veng/assets/GltfLoader.hpp>
#include <veng/assets/MeshData.hpp>
#include <veng/context/Context.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/gpu/Vertex.hpp>
#include <veng/nodes/MeshNode.hpp>
#include <veng/rendergraph/data/Data.hpp>

namespace veng::assets
{
namespace
{
// fastgltf math matrices are column-major (like glm), so copy element-wise: m[col][row].
glm::mat4 to_glm(const fastgltf::math::fmat4x4& m)
{
	glm::mat4 r(1.0F);
	for (int c = 0; c < 4; ++c)
	{
		for (int row = 0; row < 4; ++row)
		{
			r[c][row] = m[static_cast<std::size_t>(c)][static_cast<std::size_t>(row)];
		}
	}
	return r;
}

// Encoded image bytes for a DataSource that holds them inline (Array/Vector/ByteView). Buffer-view
// sources are resolved by the caller. Returns empty for an unsupported/URI source.
std::span<const std::byte> inline_bytes(const fastgltf::DataSource& source)
{
	if (const auto* array = std::get_if<fastgltf::sources::Array>(&source))
	{
		return {array->bytes.data(), array->bytes.size()};
	}
	if (const auto* vec = std::get_if<fastgltf::sources::Vector>(&source))
	{
		return {vec->bytes.data(), vec->bytes.size()};
	}
	if (const auto* view = std::get_if<fastgltf::sources::ByteView>(&source))
	{
		return view->bytes;
	}
	return {};
}

// The encoded bytes of an image, resolving a buffer-view source through its buffer (the GLB case).
std::span<const std::byte> image_bytes(const fastgltf::Asset& asset, const fastgltf::Image& image)
{
	if (const auto* bv = std::get_if<fastgltf::sources::BufferView>(&image.data))
	{
		const fastgltf::BufferView&		 view	= asset.bufferViews[bv->bufferViewIndex];
		const std::span<const std::byte> buffer = inline_bytes(asset.buffers[view.bufferIndex].data);
		if (view.byteOffset + view.byteLength > buffer.size())
		{
			return {};
		}
		return buffer.subspan(view.byteOffset, view.byteLength);
	}
	return inline_bytes(image.data);
}

// State for one load: caches uploaded textures by (image, colour space) and owns the in-progress model.
class Loader
{
	 public:
	Loader(const Context& ctx, graph::Graph& graph, fastgltf::Asset& asset)
		: m_ctx(&ctx)
		, m_graph(&graph)
		, m_asset(&asset)
	{
	}

	std::expected<GltfModel, GltfError> run()
	{
		if (auto defaults = make_defaults(); !defaults.has_value())
		{
			return std::unexpected(defaults.error());
		}
		if (auto materials = build_materials(); !materials.has_value())
		{
			return std::unexpected(materials.error());
		}

		const std::size_t scene = m_asset->defaultScene.value_or(0);
		if (scene < m_asset->scenes.size())
		{
			for (const std::size_t root : m_asset->scenes[scene].nodeIndices)
			{
				if (auto visited = visit_node(root, glm::mat4(1.0F)); !visited.has_value())
				{
					return std::unexpected(visited.error());
				}
			}
		}
		if (m_model.primitives.empty())
		{
			return std::unexpected(GltfError::NoGeometry);
		}
		if (m_has_bounds)
		{
			m_model.bounds_min = m_bounds_min;
			m_model.bounds_max = m_bounds_max;
		}
		return std::move(m_model);
	}

	 private:
	std::expected<void, GltfError> make_defaults()
	{
		const std::array<std::byte, 4> white{std::byte{255}, std::byte{255}, std::byte{255}, std::byte{255}};
		const std::array<std::byte, 4> flat{std::byte{128}, std::byte{128}, std::byte{255}, std::byte{255}};
		auto						   white_srgb	= Texture::from_pixels(*m_ctx, white, 1, 1, ColorSpace::Srgb);
		auto						   white_linear = Texture::from_pixels(*m_ctx, white, 1, 1, ColorSpace::Linear);
		auto						   normal		= Texture::from_pixels(*m_ctx, flat, 1, 1, ColorSpace::Linear);
		if (!white_srgb.has_value() || !white_linear.has_value() || !normal.has_value())
		{
			return std::unexpected(GltfError::GpuUpload);
		}
		m_default_color	 = add_texture(std::move(white_srgb.value()));
		m_default_linear = add_texture(std::move(white_linear.value()));
		m_default_normal = add_texture(std::move(normal.value()));
		return {};
	}

	graph::DataHandle add_texture(Texture texture)
	{
		const graph::DataHandle handle = m_graph->add_source<veng::gpu::ImageRef>(texture.ref()).handle;
		m_model.textures.push_back(std::move(texture));
		return handle;
	}

	// A graph source for the texture at glTF texture-index `texture_index`, decoded in `space`,
	// cached by (image, space). Falls back to a default for an out-of-range / image-less texture.
	std::expected<graph::DataHandle, GltfError> texture_source(std::size_t texture_index, ColorSpace space,
															   graph::DataHandle fallback)
	{
		if (texture_index >= m_asset->textures.size() || !m_asset->textures[texture_index].imageIndex.has_value())
		{
			return fallback;
		}
		const std::size_t image_index = *m_asset->textures[texture_index].imageIndex;
		const auto		  key		  = std::pair{image_index, space};
		if (const auto cached = m_texture_cache.find(key); cached != m_texture_cache.end())
		{
			return cached->second;
		}
		const std::span<const std::byte> bytes = image_bytes(*m_asset, m_asset->images[image_index]);
		if (bytes.empty())
		{
			return std::unexpected(GltfError::TextureLoad);
		}
		auto texture = Texture::from_encoded(*m_ctx, bytes, space);
		if (!texture.has_value())
		{
			return std::unexpected(GltfError::TextureLoad);
		}
		const graph::DataHandle handle = add_texture(std::move(texture.value()));
		m_texture_cache.emplace(key, handle);
		return handle;
	}

	std::expected<void, GltfError> build_materials()
	{
		const auto resolve = [&](const auto& opt_info, ColorSpace space,
								 graph::DataHandle fallback) -> std::expected<graph::DataHandle, GltfError>
		{
			if (!opt_info.has_value())
			{
				return fallback;
			}
			return texture_source(opt_info->textureIndex, space, fallback);
		};

		for (const fastgltf::Material& material : m_asset->materials)
		{
			GltfMaterialDesc desc;
			auto			 base = resolve(material.pbrData.baseColorTexture, ColorSpace::Srgb, m_default_color);
			auto mr		   = resolve(material.pbrData.metallicRoughnessTexture, ColorSpace::Linear, m_default_linear);
			auto normal	   = resolve(material.normalTexture, ColorSpace::Linear, m_default_normal);
			auto emissive  = resolve(material.emissiveTexture, ColorSpace::Srgb, m_default_color);
			auto occlusion = resolve(material.occlusionTexture, ColorSpace::Linear, m_default_linear);
			if (!base.has_value() || !mr.has_value() || !normal.has_value() || !emissive.has_value() ||
				!occlusion.has_value())
			{
				return std::unexpected(GltfError::TextureLoad);
			}
			desc.base_color	 = base.value();
			desc.metal_rough = mr.value();
			desc.normal		 = normal.value();
			desc.emissive	 = emissive.value();
			desc.occlusion	 = occlusion.value();

			const auto& pbr		   = material.pbrData;
			desc.base_color_factor = glm::vec4(pbr.baseColorFactor[0], pbr.baseColorFactor[1], pbr.baseColorFactor[2],
											   pbr.baseColorFactor[3]);
			desc.metallic_factor   = static_cast<float>(pbr.metallicFactor);
			desc.roughness_factor  = static_cast<float>(pbr.roughnessFactor);
			desc.normal_scale	   = material.normalTexture.has_value() ? material.normalTexture->scale : 1.0F;
			desc.occlusion_strength =
				material.occlusionTexture.has_value() ? material.occlusionTexture->strength : 1.0F;
			const float strength = static_cast<float>(material.emissiveStrength);
			desc.emissive_factor =
				glm::vec3(material.emissiveFactor[0], material.emissiveFactor[1], material.emissiveFactor[2]) *
				strength;
			switch (material.alphaMode)
			{
				case fastgltf::AlphaMode::Mask: desc.alpha_mode = AlphaMode::Mask; break;
				case fastgltf::AlphaMode::Blend: desc.alpha_mode = AlphaMode::Blend; break;
				default: desc.alpha_mode = AlphaMode::Opaque; break;
			}
			desc.alpha_cutoff = static_cast<float>(material.alphaCutoff);
			m_model.materials.push_back(desc);
		}

		// A trailing default material (white dielectric) for primitives that reference none.
		m_default_material = static_cast<std::uint32_t>(m_model.materials.size());
		m_model.materials.push_back(GltfMaterialDesc{.base_color	  = m_default_color,
													 .normal		  = m_default_normal,
													 .metal_rough	  = m_default_linear,
													 .emissive		  = m_default_color,
													 .occlusion		  = m_default_linear,
													 .metallic_factor = 0.0F});
		return {};
	}

	std::expected<void, GltfError> visit_node(std::size_t node_index, const glm::mat4& parent)
	{
		const fastgltf::Node& node	= m_asset->nodes[node_index];
		const glm::mat4		  world = parent * to_glm(fastgltf::getTransformMatrix(node));
		if (node.meshIndex.has_value())
		{
			if (auto added = add_mesh(*node.meshIndex, world); !added.has_value())
			{
				return std::unexpected(added.error());
			}
		}
		for (const std::size_t child : node.children)
		{
			if (auto visited = visit_node(child, world); !visited.has_value())
			{
				return std::unexpected(visited.error());
			}
		}
		return {};
	}

	std::expected<void, GltfError> add_mesh(std::size_t mesh_index, const glm::mat4& world)
	{
		for (const fastgltf::Primitive& prim : m_asset->meshes[mesh_index].primitives)
		{
			if (prim.type != fastgltf::PrimitiveType::Triangles)
			{
				continue; // only triangle geometry is rendered
			}
			auto geometry = build_geometry(prim);
			if (!geometry.has_value())
			{
				return std::unexpected(geometry.error());
			}
			if (geometry->indices.empty())
			{
				continue;
			}

			const graph::DataHandle model_handle = m_graph->add_source<glm::mat4>(world).handle;
			const std::uint32_t		material =
				prim.materialIndex.has_value() ? static_cast<std::uint32_t>(*prim.materialIndex) : m_default_material;

			// Normal mode emits one mesh per primitive; LOD mode (m_lods set) emits a decimated chain.
			// Both run the shared pipeline; glTF's authored normals/UVs are carried through and preserved.
			if (m_lods.empty())
			{
				const ProcessedMesh mesh = process(*geometry, m_process_opts);
				m_model.primitives.push_back(GltfPrimitive{
					.mesh = make_mesh_node(mesh.vertices, mesh.indices), .model = model_handle, .material = material});
			}
			else
			{
				m_lod_model.primitives.push_back(
					GltfLodPrimitive{.lods = build_lods(*geometry), .model = model_handle, .material = material});
			}

			for (const glm::vec3& position : geometry->positions)
			{
				const glm::vec3 world_pos = glm::vec3(world * glm::vec4(position, 1.0F));
				m_bounds_min			  = m_has_bounds ? glm::min(m_bounds_min, world_pos) : world_pos;
				m_bounds_max			  = m_has_bounds ? glm::max(m_bounds_max, world_pos) : world_pos;
				m_has_bounds			  = true;
			}
		}
		return {};
	}

	// Create a MeshNode for one mesh and return its published MeshRef edge.
	graph::DataHandle make_mesh_node(std::span<const veng::gpu::PbrVertex> vertices,
									 std::span<const std::uint32_t>		   indices)
	{
		const graph::DataHandle mesh_handle =
			m_graph->add(std::make_unique<graph::ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));
		const graph::NodeHandle node_handle =
			m_graph->add(std::make_unique<veng::nodes::MeshNode>(vertices, indices, mesh_handle));
		m_graph->set_producer(mesh_handle, node_handle);
		return mesh_handle;
	}

	// Decimate one primitive into the configured LOD chain via the shared pipeline: prepare once, then
	// finalise per level. Decimation is attribute-aware (authored normals + UVs feed the metric) and
	// surviving vertices keep their exact attributes, so texturing is identical across levels.
	std::vector<graph::DataHandle> build_lods(const MeshData& data)
	{
		const PreparedMesh			   prep = prepare(data, m_process_opts);
		std::vector<graph::DataHandle> result;
		result.reserve(m_lods.size());
		for (const LodLevel& level : m_lods)
		{
			const ProcessedMesh mesh = finalize(prep, level, m_process_opts);
			result.push_back(make_mesh_node(mesh.vertices, mesh.indices));
		}
		return result;
	}

	 public:
	// Like run(), but every primitive becomes a decimated LOD chain (see @ref load_gltf_lods).
	std::expected<GltfLodModel, GltfError> run_lods(std::span<const LodLevel> levels)
	{
		m_lod_storage.assign(levels.begin(), levels.end());
		if (m_lod_storage.empty())
		{
			m_lod_storage.push_back(LodLevel{}); // no levels ⇒ a single full-detail mesh
		}
		m_lods = m_lod_storage;

		if (auto defaults = make_defaults(); !defaults.has_value())
		{
			return std::unexpected(defaults.error());
		}
		if (auto materials = build_materials(); !materials.has_value())
		{
			return std::unexpected(materials.error());
		}
		const std::size_t scene = m_asset->defaultScene.value_or(0);
		if (scene < m_asset->scenes.size())
		{
			for (const std::size_t root : m_asset->scenes[scene].nodeIndices)
			{
				if (auto visited = visit_node(root, glm::mat4(1.0F)); !visited.has_value())
				{
					return std::unexpected(visited.error());
				}
			}
		}
		if (m_lod_model.primitives.empty())
		{
			return std::unexpected(GltfError::NoGeometry);
		}
		m_lod_model.textures  = std::move(m_model.textures);
		m_lod_model.materials = std::move(m_model.materials);
		if (m_has_bounds)
		{
			m_lod_model.bounds_min = m_bounds_min;
			m_lod_model.bounds_max = m_bounds_max;
		}
		return std::move(m_lod_model);
	}

	 private:
	// Parse one primitive into a MeshData. UVs and tangents are left empty when the asset omits them,
	// so the shared pipeline synthesises them (box UVs / computed tangents) rather than this loader.
	std::expected<MeshData, GltfError> build_geometry(const fastgltf::Primitive& prim)
	{
		const auto* position = prim.findAttribute("POSITION");
		if (position == prim.attributes.cend())
		{
			return std::unexpected(GltfError::MissingPosition);
		}
		const auto* normal_attr = prim.findAttribute("NORMAL");
		if (normal_attr == prim.attributes.cend())
		{
			return std::unexpected(GltfError::MissingNormal);
		}

		MeshData data;
		fastgltf::iterateAccessor<glm::vec3>(*m_asset, m_asset->accessors[position->accessorIndex],
											 [&](glm::vec3 v) { data.positions.push_back(v); });
		fastgltf::iterateAccessor<glm::vec3>(*m_asset, m_asset->accessors[normal_attr->accessorIndex],
											 [&](glm::vec3 v) { data.normals.push_back(v); });
		if (data.positions.empty() || data.normals.size() != data.positions.size())
		{
			return std::unexpected(GltfError::MissingNormal);
		}

		if (const auto* uv = prim.findAttribute("TEXCOORD_0"); uv != prim.attributes.cend())
		{
			data.uvs.resize(data.positions.size());
			std::size_t at = 0;
			fastgltf::iterateAccessor<glm::vec2>(*m_asset, m_asset->accessors[uv->accessorIndex],
												 [&](glm::vec2 v)
												 {
													 if (at < data.uvs.size())
													 {
														 data.uvs[at++] = v;
													 }
												 });
		}

		if (prim.indicesAccessor.has_value())
		{
			fastgltf::iterateAccessor<std::uint32_t>(*m_asset, m_asset->accessors[*prim.indicesAccessor],
													 [&](std::uint32_t i) { data.indices.push_back(i); });
		}
		else
		{
			data.indices.resize(data.positions.size());
			for (std::uint32_t i = 0; i < data.positions.size(); ++i)
			{
				data.indices[i] = i;
			}
		}

		if (const auto* tan = prim.findAttribute("TANGENT"); tan != prim.attributes.cend())
		{
			data.tangents.resize(data.positions.size(), glm::vec4(1.0F, 0.0F, 0.0F, 1.0F));
			std::size_t at = 0;
			fastgltf::iterateAccessor<glm::vec4>(*m_asset, m_asset->accessors[tan->accessorIndex],
												 [&](glm::vec4 v)
												 {
													 if (at < data.tangents.size())
													 {
														 data.tangents[at++] = v;
													 }
												 });
		}
		return data;
	}

	const Context*													m_ctx;
	graph::Graph*													m_graph;
	fastgltf::Asset*												m_asset;
	GltfModel														m_model;
	std::map<std::pair<std::size_t, ColorSpace>, graph::DataHandle> m_texture_cache;
	graph::DataHandle												m_default_color;
	graph::DataHandle												m_default_linear;
	graph::DataHandle												m_default_normal;
	std::uint32_t													m_default_material = 0;
	glm::vec3														m_bounds_min{0.0F};
	glm::vec3														m_bounds_max{0.0F};
	bool															m_has_bounds = false;

	ProcessOptions			  m_process_opts{}; ///< Geometry pipeline options (authored attrs are preserved).
	std::vector<LodLevel>	  m_lod_storage;	///< Owns the levels for LOD mode (run_lods).
	std::span<const LodLevel> m_lods{};			///< Non-empty ⇒ LOD mode (add_mesh emits chains).
	GltfLodModel			  m_lod_model;		///< Assembled by run_lods.
};
} // namespace

std::expected<GltfModel, GltfError> load_gltf(const Context& ctx, graph::Graph& graph, const std::string& path)
{
	auto data = fastgltf::GltfDataBuffer::FromPath(path);
	if (data.error() != fastgltf::Error::None)
	{
		return std::unexpected(GltfError::FileUnreadable);
	}

	fastgltf::Parser parser;
	const auto		 options = fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages |
							   fastgltf::Options::GenerateMeshIndices | fastgltf::Options::DecomposeNodeMatrices;
	auto			 loaded	 = parser.loadGltf(data.get(), std::filesystem::path(path).parent_path(), options);
	if (loaded.error() != fastgltf::Error::None)
	{
		return std::unexpected(GltfError::ParseFailed);
	}

	Loader loader(ctx, graph, loaded.get());
	return loader.run();
}

std::expected<GltfLodModel, GltfError> load_gltf_lods(const Context& ctx, graph::Graph& graph, const std::string& path,
													  std::span<const LodLevel> levels)
{
	auto data = fastgltf::GltfDataBuffer::FromPath(path);
	if (data.error() != fastgltf::Error::None)
	{
		return std::unexpected(GltfError::FileUnreadable);
	}

	fastgltf::Parser parser;
	const auto		 options = fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages |
							   fastgltf::Options::GenerateMeshIndices | fastgltf::Options::DecomposeNodeMatrices;
	auto			 loaded	 = parser.loadGltf(data.get(), std::filesystem::path(path).parent_path(), options);
	if (loaded.error() != fastgltf::Error::None)
	{
		return std::unexpected(GltfError::ParseFailed);
	}

	Loader loader(ctx, graph, loaded.get());
	return loader.run_lods(levels);
}
} // namespace veng::assets
