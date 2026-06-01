/**
 * @file
 * @author chris
 * @brief @ref veng::assets::load_3mf implementation: lib3mf parse → triangle-soup @ref MeshData →
 *        shared geometry pipeline.
 * @ingroup assets
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <lib3mf_implicit.hpp>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <veng/assets/ThreemfLoader.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/gpu/Vertex.hpp>
#include <veng/nodes/MeshNode.hpp>
#include <veng/rendergraph/data/Data.hpp>

namespace veng::assets
{
namespace
{
// Read every mesh object's triangles into one flattened soup (positions in groups of three, no
// indices), so the shared pipeline welds + crease-smooths it like an STL.
std::expected<MeshData, ThreemfError> parse_to_mesh(const std::string& path)
{
	if (!std::filesystem::exists(path))
	{
		return std::unexpected(ThreemfError::FileUnreadable);
	}
	try
	{
		Lib3MF::PWrapper wrapper = Lib3MF::CWrapper::loadLibrary();
		Lib3MF::PModel	 model	 = wrapper->CreateModel();
		Lib3MF::PReader	 reader	 = model->QueryReader("3mf");
		reader->ReadFromFile(path);

		MeshData					data;
		Lib3MF::PMeshObjectIterator it = model->GetMeshObjects();
		while (it->MoveNext())
		{
			Lib3MF::PMeshObject			   mesh = it->GetCurrentMeshObject();
			std::vector<Lib3MF::sPosition> verts;
			std::vector<Lib3MF::sTriangle> tris;
			mesh->GetVertices(verts);
			mesh->GetTriangleIndices(tris);
			data.positions.reserve(data.positions.size() + (tris.size() * 3));
			for (const Lib3MF::sTriangle& tri : tris)
			{
				for (const std::uint32_t vi : tri.m_Indices)
				{
					const Lib3MF::sPosition& v = verts[vi];
					data.positions.emplace_back(v.m_Coordinates[0], v.m_Coordinates[1], v.m_Coordinates[2]);
				}
			}
		}
		if (data.positions.empty())
		{
			return std::unexpected(ThreemfError::NoGeometry);
		}
		return data;
	}
	catch (...)
	{
		return std::unexpected(ThreemfError::ParseFailed);
	}
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

std::expected<ThreemfMesh, ThreemfError> load_3mf(graph::Graph& graph, const std::string& path,
												  const ProcessOptions& opts)
{
	auto data = parse_to_mesh(path);
	if (!data.has_value())
	{
		return std::unexpected(data.error());
	}
	const ProcessedMesh mesh = process(*data, opts);
	if (mesh.vertices.empty() || mesh.indices.empty())
	{
		return std::unexpected(ThreemfError::NoGeometry);
	}
	return ThreemfMesh{.mesh		 = add_mesh_node(graph, mesh),
					   .vertex_count = static_cast<std::uint32_t>(mesh.vertices.size()),
					   .index_count	 = static_cast<std::uint32_t>(mesh.indices.size()),
					   .bounds_min	 = mesh.bounds_min,
					   .bounds_max	 = mesh.bounds_max};
}

std::expected<ThreemfLodSet, ThreemfError> load_3mf_lods(graph::Graph& graph, const std::string& path,
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
		return std::unexpected(ThreemfError::NoGeometry);
	}
	ThreemfLodSet set;
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
