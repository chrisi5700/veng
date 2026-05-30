//
// See veng/assets/GltfLoader.hpp and review.md items 6 + 7.
//

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

// Per-vertex tangents (xyz + handedness) from positions, normals and UVs — used when a primitive
// has a normal map but no TANGENT attribute. A simple accumulate-then-orthonormalize (not
// mikktspace; see findings.md). Returns default +X tangents when there are no usable UVs.
std::vector<glm::vec4> compute_tangents(const std::vector<glm::vec3>& positions, const std::vector<glm::vec3>& normals,
										const std::vector<glm::vec2>& uvs, const std::vector<std::uint32_t>& indices)
{
	std::vector<glm::vec3> tan(positions.size(), glm::vec3(0.0F));
	std::vector<glm::vec3> bitan(positions.size(), glm::vec3(0.0F));
	for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
	{
		const std::uint32_t i0	= indices[i];
		const std::uint32_t i1	= indices[i + 1];
		const std::uint32_t i2	= indices[i + 2];
		const glm::vec3		e1	= positions[i1] - positions[i0];
		const glm::vec3		e2	= positions[i2] - positions[i0];
		const glm::vec2		d1	= uvs[i1] - uvs[i0];
		const glm::vec2		d2	= uvs[i2] - uvs[i0];
		const float			det = (d1.x * d2.y) - (d2.x * d1.y);
		const float			f	= std::abs(det) < 1e-8F ? 0.0F : 1.0F / det;
		const glm::vec3		t	= f * ((d2.y * e1) - (d1.y * e2));
		const glm::vec3		b	= f * ((d1.x * e2) - (d2.x * e1));
		for (const std::uint32_t idx : {i0, i1, i2})
		{
			tan[idx] += t;
			bitan[idx] += b;
		}
	}

	std::vector<glm::vec4> result(positions.size());
	for (std::size_t v = 0; v < positions.size(); ++v)
	{
		const glm::vec3 n = normals[v];
		glm::vec3		t = tan[v] - (n * glm::dot(n, tan[v])); // Gram-Schmidt
		if (glm::dot(t, t) < 1e-12F)
		{
			// Degenerate (no UV gradient): pick any axis perpendicular to the normal.
			t = std::abs(n.x) < 0.9F ? glm::cross(n, glm::vec3(1, 0, 0)) : glm::cross(n, glm::vec3(0, 1, 0));
		}
		t					   = glm::normalize(t);
		const float handedness = (glm::dot(glm::cross(n, t), bitan[v]) < 0.0F) ? -1.0F : 1.0F;
		result[v]			   = glm::vec4(t, handedness);
	}
	return result;
}

struct Geometry
{
	std::vector<veng::gpu::PbrVertex> vertices;
	std::vector<std::uint32_t>		  indices;
};

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
				case fastgltf::AlphaMode::Mask:
					desc.alpha_mode = AlphaMode::Mask;
					break;
				case fastgltf::AlphaMode::Blend:
					desc.alpha_mode = AlphaMode::Blend;
					break;
				default:
					desc.alpha_mode = AlphaMode::Opaque;
					break;
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

			const graph::DataHandle mesh_handle =
				m_graph->add(std::make_unique<graph::ValueData<veng::gpu::MeshRef>>(veng::gpu::MeshRef{}));
			const graph::NodeHandle node_handle = m_graph->add(std::make_unique<veng::nodes::MeshNode>(
				std::span<const veng::gpu::PbrVertex>(geometry->vertices),
				std::span<const std::uint32_t>(geometry->indices), mesh_handle));
			m_graph->set_producer(mesh_handle, node_handle);

			const graph::DataHandle model_handle = m_graph->add_source<glm::mat4>(world).handle;
			const std::uint32_t		material =
				prim.materialIndex.has_value() ? static_cast<std::uint32_t>(*prim.materialIndex) : m_default_material;
			m_model.primitives.push_back(
				GltfPrimitive{.mesh = mesh_handle, .model = model_handle, .material = material});

			for (const veng::gpu::PbrVertex& vertex : geometry->vertices)
			{
				const glm::vec3 world_pos = glm::vec3(world * glm::vec4(vertex.position, 1.0F));
				m_bounds_min			  = m_has_bounds ? glm::min(m_bounds_min, world_pos) : world_pos;
				m_bounds_max			  = m_has_bounds ? glm::max(m_bounds_max, world_pos) : world_pos;
				m_has_bounds			  = true;
			}
		}
		return {};
	}

	std::expected<Geometry, GltfError> build_geometry(const fastgltf::Primitive& prim)
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

		std::vector<glm::vec3> positions;
		fastgltf::iterateAccessor<glm::vec3>(*m_asset, m_asset->accessors[position->accessorIndex],
											 [&](glm::vec3 v) { positions.push_back(v); });
		std::vector<glm::vec3> normals;
		fastgltf::iterateAccessor<glm::vec3>(*m_asset, m_asset->accessors[normal_attr->accessorIndex],
											 [&](glm::vec3 v) { normals.push_back(v); });
		if (positions.empty() || normals.size() != positions.size())
		{
			return std::unexpected(GltfError::MissingNormal);
		}

		std::vector<glm::vec2> uvs(positions.size(), glm::vec2(0.0F));
		bool				   has_uv = false;
		if (const auto* uv = prim.findAttribute("TEXCOORD_0"); uv != prim.attributes.cend())
		{
			has_uv		   = true;
			std::size_t at = 0;
			fastgltf::iterateAccessor<glm::vec2>(*m_asset, m_asset->accessors[uv->accessorIndex],
												 [&](glm::vec2 v)
												 {
													 if (at < uvs.size())
													 {
														 uvs[at++] = v;
													 }
												 });
		}

		std::vector<std::uint32_t> indices;
		if (prim.indicesAccessor.has_value())
		{
			fastgltf::iterateAccessor<std::uint32_t>(*m_asset, m_asset->accessors[*prim.indicesAccessor],
													 [&](std::uint32_t i) { indices.push_back(i); });
		}
		else
		{
			indices.resize(positions.size());
			for (std::uint32_t i = 0; i < positions.size(); ++i)
			{
				indices[i] = i;
			}
		}

		std::vector<glm::vec4> tangents(positions.size(), glm::vec4(1.0F, 0.0F, 0.0F, 1.0F));
		if (const auto* tan = prim.findAttribute("TANGENT"); tan != prim.attributes.cend())
		{
			std::size_t at = 0;
			fastgltf::iterateAccessor<glm::vec4>(*m_asset, m_asset->accessors[tan->accessorIndex],
												 [&](glm::vec4 v)
												 {
													 if (at < tangents.size())
													 {
														 tangents[at++] = v;
													 }
												 });
		}
		else if (has_uv)
		{
			tangents = compute_tangents(positions, normals, uvs, indices);
		}

		Geometry geometry;
		geometry.vertices.reserve(positions.size());
		for (std::size_t v = 0; v < positions.size(); ++v)
		{
			geometry.vertices.push_back(veng::gpu::PbrVertex{
				.position = positions[v], .normal = normals[v], .tangent = tangents[v], .uv = uvs[v]});
		}
		geometry.indices = std::move(indices);
		return geometry;
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
} // namespace veng::assets
