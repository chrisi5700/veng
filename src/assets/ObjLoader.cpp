/**
 * @file
 * @author chris
 * @brief @ref veng::assets::load_obj implementation: tinyobjloader parse → de-index → @ref MeshData
 *        → shared geometry pipeline.
 * @ingroup assets
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <tiny_obj_loader.h>
#include <unordered_map>
#include <vector>
#include <veng/assets/ObjLoader.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/gpu/Vertex.hpp>
#include <veng/nodes/MeshNode.hpp>
#include <veng/rendergraph/data/Data.hpp>

namespace veng::assets
{
namespace
{
// OBJ indexes position/uv/normal independently, so a face corner is a (v, vt, vn) triplet; distinct
// triplets become distinct welded vertices.
struct IndexKey
{
	int			v;
	int			vt;
	int			vn;
	friend bool operator==(const IndexKey&, const IndexKey&) noexcept = default;
};
struct IndexHash
{
	std::size_t operator()(const IndexKey& k) const noexcept
	{
		std::size_t h = 1469598103934665603ULL;
		for (const int x : {k.v, k.vt, k.vn})
		{
			h = (h ^ static_cast<std::size_t>(static_cast<unsigned int>(x))) * 1099511628211ULL;
		}
		return h;
	}
};

// Parse (and triangulate) an OBJ into one merged, indexed MeshData. Normals/UVs are filled only when
// the file provides them (else left empty for the pipeline to synthesise). OBJ's V axis is flipped to
// our top-left convention.
std::expected<MeshData, ObjError> parse_to_mesh(const std::string& path)
{
	if (!std::filesystem::exists(path))
	{
		return std::unexpected(ObjError::FileUnreadable);
	}
	tinyobj::ObjReaderConfig config;
	config.triangulate	   = true;
	config.mtl_search_path = std::filesystem::path(path).parent_path().string();
	tinyobj::ObjReader reader;
	if (!reader.ParseFromFile(path, config))
	{
		return std::unexpected(ObjError::ParseFailed);
	}

	const tinyobj::attrib_t& attrib		 = reader.GetAttrib();
	const bool				 has_normals = !attrib.normals.empty();
	const bool				 has_uv		 = !attrib.texcoords.empty();

	MeshData											   data;
	std::unordered_map<IndexKey, std::uint32_t, IndexHash> dedup;
	for (const tinyobj::shape_t& shape : reader.GetShapes())
	{
		for (const tinyobj::index_t& idx : shape.mesh.indices)
		{
			const IndexKey key{.v = idx.vertex_index, .vt = idx.texcoord_index, .vn = idx.normal_index};
			const auto [it, inserted] = dedup.try_emplace(key, static_cast<std::uint32_t>(data.positions.size()));
			if (inserted)
			{
				const auto vi = static_cast<std::size_t>(idx.vertex_index) * 3;
				data.positions.emplace_back(attrib.vertices[vi], attrib.vertices[vi + 1], attrib.vertices[vi + 2]);
				if (has_normals)
				{
					if (idx.normal_index >= 0)
					{
						const auto ni = static_cast<std::size_t>(idx.normal_index) * 3;
						data.normals.emplace_back(attrib.normals[ni], attrib.normals[ni + 1], attrib.normals[ni + 2]);
					}
					else
					{
						data.normals.emplace_back(0.0F, 1.0F, 0.0F);
					}
				}
				if (has_uv)
				{
					if (idx.texcoord_index >= 0)
					{
						const auto ti = static_cast<std::size_t>(idx.texcoord_index) * 2;
						data.uvs.emplace_back(attrib.texcoords[ti], 1.0F - attrib.texcoords[ti + 1]); // flip V
					}
					else
					{
						data.uvs.emplace_back(0.0F, 0.0F);
					}
				}
			}
			data.indices.push_back(it->second);
		}
	}
	if (data.indices.empty())
	{
		return std::unexpected(ObjError::NoGeometry);
	}
	return data;
}

graph::DataHandle add_mesh_node(graph::Graph& graph, const ProcessedMesh& mesh)
{
	const graph::DataHandle handle = graph.add(std::make_unique<graph::ValueData<gpu::MeshRef>>(gpu::MeshRef{}));
	const graph::NodeHandle node   = graph.add(std::make_unique<nodes::MeshNode>(
		std::span<const gpu::PbrVertex>(mesh.vertices), std::span<const std::uint32_t>(mesh.indices), handle));
	graph.set_producer(handle, node);
	return handle;
}
} // namespace

std::expected<ObjMesh, ObjError> load_obj(graph::Graph& graph, const std::string& path, const ProcessOptions& opts)
{
	auto data = parse_to_mesh(path);
	if (!data.has_value())
	{
		return std::unexpected(data.error());
	}
	const ProcessedMesh mesh = process(*data, opts);
	if (mesh.vertices.empty() || mesh.indices.empty())
	{
		return std::unexpected(ObjError::NoGeometry);
	}
	return ObjMesh{.mesh		 = add_mesh_node(graph, mesh),
				   .vertex_count = static_cast<std::uint32_t>(mesh.vertices.size()),
				   .index_count	 = static_cast<std::uint32_t>(mesh.indices.size()),
				   .bounds_min	 = mesh.bounds_min,
				   .bounds_max	 = mesh.bounds_max};
}

std::expected<ObjLodSet, ObjError> load_obj_lods(graph::Graph& graph, const std::string& path,
												 std::span<const LodLevel> levels, const ProcessOptions& base)
{
	auto data = parse_to_mesh(path);
	if (!data.has_value())
	{
		return std::unexpected(data.error());
	}
	const LodLevel					full{};
	const std::span<const LodLevel> effective = levels.empty() ? std::span<const LodLevel>(&full, 1) : levels;

	const PreparedMesh prep = prepare(*data, base);
	if (prep.mesh.positions.empty())
	{
		return std::unexpected(ObjError::NoGeometry);
	}

	ObjLodSet set;
	set.bounds_min = prep.bounds_min;
	set.bounds_max = prep.bounds_max;
	set.meshes.reserve(effective.size());
	for (const LodLevel& level : effective)
	{
		set.meshes.push_back(add_mesh_node(graph, finalize(prep, level, base)));
	}
	return set;
}
} // namespace veng::assets
