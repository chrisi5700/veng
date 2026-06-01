/**
 * @file
 * @author chris
 * @brief @ref veng::assets::load_ply implementation: tinyply parse → @ref MeshData → shared pipeline.
 * @ingroup assets
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <memory>
#include <span>
#include <string>
#include <tinyply.h>
#include <vector>
#include <veng/assets/PlyLoader.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/gpu/Vertex.hpp>
#include <veng/nodes/MeshNode.hpp>
#include <veng/rendergraph/data/Data.hpp>

namespace veng::assets
{
namespace
{
// Request an optional vertex property set; returns nullptr if absent (tinyply throws on missing).
std::shared_ptr<tinyply::PlyData> try_request(tinyply::PlyFile& file, const std::string& element,
											  const std::vector<std::string>& props)
{
	try
	{
		return file.request_properties_from_element(element, props);
	}
	catch (...)
	{
		return nullptr;
	}
}

// Read N float components per element from a PlyData (FLOAT32 or FLOAT64), into a flat float vector.
// Takes a non-const ref because tinyply::Buffer::get() is non-const.
std::vector<float> read_floats(tinyply::PlyData& data, std::size_t components)
{
	std::vector<float> out(data.count * components);
	if (data.t == tinyply::Type::FLOAT32)
	{
		std::memcpy(out.data(), data.buffer.get(), out.size() * sizeof(float));
	}
	else if (data.t == tinyply::Type::FLOAT64)
	{
		const auto* src = static_cast<const double*>(static_cast<const void*>(data.buffer.get()));
		for (std::size_t i = 0; i < out.size(); ++i)
		{
			out[i] = static_cast<float>(src[i]);
		}
	}
	return out;
}

// Read a triangle index list (assumes triangulated faces) widening any integer width to uint32.
std::vector<std::uint32_t> read_indices(tinyply::PlyData& data)
{
	const std::size_t		   total = data.count * 3;
	std::vector<std::uint32_t> out(total);
	const auto*				   bytes = data.buffer.get();
	switch (data.t)
	{
		case tinyply::Type::UINT32:
		case tinyply::Type::INT32: std::memcpy(out.data(), bytes, total * sizeof(std::uint32_t)); break;
		case tinyply::Type::UINT16:
		case tinyply::Type::INT16:
		{
			const auto* src = static_cast<const std::uint16_t*>(static_cast<const void*>(bytes));
			for (std::size_t i = 0; i < total; ++i)
			{
				out[i] = src[i];
			}
			break;
		}
		case tinyply::Type::UINT8:
		case tinyply::Type::INT8:
			for (std::size_t i = 0; i < total; ++i)
			{
				out[i] = bytes[i];
			}
			break;
		default: out.clear(); break;
	}
	return out;
}

std::expected<MeshData, PlyError> parse_to_mesh(const std::string& path)
{
	std::ifstream stream(path, std::ios::binary);
	if (!stream)
	{
		return std::unexpected(PlyError::FileUnreadable);
	}
	tinyply::PlyFile file;
	try
	{
		if (!file.parse_header(stream))
		{
			return std::unexpected(PlyError::ParseFailed);
		}
	}
	catch (...)
	{
		return std::unexpected(PlyError::ParseFailed);
	}

	std::shared_ptr<tinyply::PlyData> pos = try_request(file, "vertex", {"x", "y", "z"});
	std::shared_ptr<tinyply::PlyData> nrm = try_request(file, "vertex", {"nx", "ny", "nz"});
	std::shared_ptr<tinyply::PlyData> uv  = try_request(file, "vertex", {"u", "v"});
	if (uv == nullptr)
	{
		uv = try_request(file, "vertex", {"s", "t"});
	}
	std::shared_ptr<tinyply::PlyData> col	= try_request(file, "vertex", {"red", "green", "blue"});
	std::shared_ptr<tinyply::PlyData> faces = try_request(file, "face", {"vertex_indices"});
	if (faces == nullptr)
	{
		faces = try_request(file, "face", {"vertex_index"});
	}
	if (pos == nullptr || faces == nullptr)
	{
		return std::unexpected(PlyError::ParseFailed);
	}

	try
	{
		file.read(stream);
	}
	catch (...)
	{
		return std::unexpected(PlyError::ParseFailed);
	}

	MeshData				 data;
	const std::vector<float> px = read_floats(*pos, 3);
	data.positions.resize(pos->count);
	for (std::size_t i = 0; i < pos->count; ++i)
	{
		data.positions[i] = glm::vec3(px[3 * i], px[3 * i + 1], px[3 * i + 2]);
	}
	if (nrm != nullptr && nrm->count == pos->count)
	{
		const std::vector<float> n = read_floats(*nrm, 3);
		data.normals.resize(nrm->count);
		for (std::size_t i = 0; i < nrm->count; ++i)
		{
			data.normals[i] = glm::vec3(n[3 * i], n[3 * i + 1], n[3 * i + 2]);
		}
	}
	if (uv != nullptr && uv->count == pos->count)
	{
		const std::vector<float> t = read_floats(*uv, 2);
		data.uvs.resize(uv->count);
		for (std::size_t i = 0; i < uv->count; ++i)
		{
			data.uvs[i] = glm::vec2(t[2 * i], t[2 * i + 1]);
		}
	}
	if (col != nullptr && col->count == pos->count && col->t == tinyply::Type::UINT8)
	{
		const auto* bytes = col->buffer.get();
		data.colors.resize(col->count);
		for (std::size_t i = 0; i < col->count; ++i)
		{
			data.colors[i] =
				glm::vec4(static_cast<float>(bytes[3 * i]) / 255.0F, static_cast<float>(bytes[3 * i + 1]) / 255.0F,
						  static_cast<float>(bytes[3 * i + 2]) / 255.0F, 1.0F);
		}
	}
	data.indices = read_indices(*faces);
	if (data.indices.empty())
	{
		return std::unexpected(PlyError::NoGeometry);
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

std::expected<PlyMesh, PlyError> load_ply(graph::Graph& graph, const std::string& path, const ProcessOptions& opts)
{
	auto data = parse_to_mesh(path);
	if (!data.has_value())
	{
		return std::unexpected(data.error());
	}
	const ProcessedMesh mesh = process(*data, opts);
	if (mesh.vertices.empty() || mesh.indices.empty())
	{
		return std::unexpected(PlyError::NoGeometry);
	}
	return PlyMesh{.mesh		 = add_mesh_node(graph, mesh),
				   .vertex_count = static_cast<std::uint32_t>(mesh.vertices.size()),
				   .index_count	 = static_cast<std::uint32_t>(mesh.indices.size()),
				   .bounds_min	 = mesh.bounds_min,
				   .bounds_max	 = mesh.bounds_max};
}

std::expected<PlyLodSet, PlyError> load_ply_lods(graph::Graph& graph, const std::string& path,
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
		return std::unexpected(PlyError::NoGeometry);
	}
	PlyLodSet set;
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
