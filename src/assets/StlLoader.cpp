/**
 * @file
 * @author chris
 * @brief @ref veng::assets::load_stl implementation: ASCII/binary STL parsing into a @ref MeshData,
 *        then the shared geometry pipeline (@ref process / prepare / finalize) does the rest.
 * @ingroup assets
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <veng/assets/MeshData.hpp>
#include <veng/assets/StlLoader.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/gpu/Vertex.hpp>
#include <veng/nodes/MeshNode.hpp>
#include <veng/rendergraph/data/Data.hpp>

namespace veng::assets
{
namespace
{
// One parsed triangle: three corner positions, straight from the file (winding preserved).
struct Triangle
{
	std::array<glm::vec3, 3> p;
};

// Binary STL is recognised structurally, not by header text: the 80-byte header is followed by a
// uint32 triangle count and exactly 50 bytes per triangle. A header beginning with "solid" is NOT a
// reliable ASCII signal (many exporters write "solid" into binary headers), so we trust the size.
bool looks_binary(const std::vector<std::byte>& bytes)
{
	if (bytes.size() < 84)
	{
		return false;
	}
	std::uint32_t count = 0;
	std::memcpy(&count, bytes.data() + 80, sizeof(count));
	return bytes.size() == 84U + (static_cast<std::size_t>(count) * 50U);
}

std::vector<Triangle> parse_binary(const std::vector<std::byte>& bytes)
{
	std::uint32_t count = 0;
	std::memcpy(&count, bytes.data() + 80, sizeof(count));
	std::vector<Triangle> tris;
	tris.reserve(count);
	std::size_t off = 84;
	for (std::uint32_t t = 0; t < count; ++t, off += 50)
	{
		Triangle tri{};
		// Layout: float[3] facet normal (skipped), then 3× float[3] positions, then uint16 attr.
		for (int v = 0; v < 3; ++v)
		{
			std::array<float, 3> xyz{};
			std::memcpy(xyz.data(), bytes.data() + off + 12 + (static_cast<std::size_t>(v) * 12), sizeof(xyz));
			tri.p[static_cast<std::size_t>(v)] = glm::vec3(xyz[0], xyz[1], xyz[2]);
		}
		tris.push_back(tri);
	}
	return tris;
}

// ASCII STL: scan for "vertex x y z" triplets and group every three into a triangle. Per-facet
// "normal" lines are ignored — normals are resynthesised by the pipeline with proper crease handling.
std::vector<Triangle> parse_ascii(const std::vector<std::byte>& bytes)
{
	const char* cur = static_cast<const char*>(static_cast<const void*>(bytes.data()));
	const char* end = cur + bytes.size();

	std::vector<glm::vec3> verts;
	while (cur < end)
	{
		while (cur < end && (*cur != 'v' || std::strncmp(cur, "vertex", 6) != 0))
		{
			++cur;
		}
		if (cur >= end)
		{
			break;
		}
		cur += 6; // past "vertex"
		char*		next = nullptr;
		const float x	 = std::strtof(cur, &next);
		const float y	 = std::strtof(next, &next);
		const float z	 = std::strtof(next, &next);
		verts.emplace_back(x, y, z);
		cur = next;
	}

	std::vector<Triangle> tris;
	tris.reserve(verts.size() / 3);
	for (std::size_t i = 0; i + 2 < verts.size(); i += 3)
	{
		tris.push_back(Triangle{.p = {verts[i], verts[i + 1], verts[i + 2]}});
	}
	return tris;
}

// Read the whole file as bytes; std::byte buffer (not istreambuf_iterator, which yields char).
std::expected<std::vector<std::byte>, StlError> read_file(const std::string& path)
{
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file)
	{
		return std::unexpected(StlError::FileUnreadable);
	}
	const std::streamoff size = file.tellg();
	if (size <= 0)
	{
		return std::unexpected(StlError::FileUnreadable);
	}
	file.seekg(0);
	std::vector<std::byte> bytes(static_cast<std::size_t>(size));
	file.read(static_cast<char*>(static_cast<void*>(bytes.data())), size);
	if (!file)
	{
		return std::unexpected(StlError::FileUnreadable);
	}
	return bytes;
}

// Parse an STL file into a non-indexed triangle-soup MeshData (positions only — the pipeline welds
// and synthesises normals/UVs/tangents).
std::expected<MeshData, StlError> parse_to_mesh(const std::string& path)
{
	auto bytes = read_file(path);
	if (!bytes.has_value())
	{
		return std::unexpected(bytes.error());
	}
	const std::vector<Triangle> tris = looks_binary(*bytes) ? parse_binary(*bytes) : parse_ascii(*bytes);
	if (tris.empty())
	{
		return std::unexpected(StlError::ParseFailed);
	}
	MeshData data;
	data.positions.reserve(tris.size() * 3);
	for (const Triangle& tri : tris)
	{
		for (const glm::vec3& p : tri.p)
		{
			data.positions.push_back(p);
		}
	}
	return data;
}

ProcessOptions to_process_options(const StlOptions& opts)
{
	return ProcessOptions{.repair				= opts.repair,
						  .crease_angle_deg		= opts.crease_angle_deg,
						  .texture_tiles		= opts.texture_tiles,
						  .world_units_per_tile = opts.world_units_per_tile};
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

std::expected<StlMesh, StlError> load_stl(graph::Graph& graph, const std::string& path, const StlOptions& opts)
{
	auto data = parse_to_mesh(path);
	if (!data.has_value())
	{
		return std::unexpected(data.error());
	}

	const ProcessOptions po	   = to_process_options(opts);
	const LodLevel		 level = {.target_ratio = opts.decimate_ratio,
								  .max_error	= opts.decimate_max_error,
								  .aggressive	= opts.decimate_aggressive};
	const ProcessedMesh	 mesh  = finalize(prepare(*data, po), level, po);
	if (mesh.vertices.empty() || mesh.indices.empty())
	{
		return std::unexpected(StlError::NoGeometry);
	}

	return StlMesh{.mesh		 = add_mesh_node(graph, mesh),
				   .vertex_count = static_cast<std::uint32_t>(mesh.vertices.size()),
				   .index_count	 = static_cast<std::uint32_t>(mesh.indices.size()),
				   .bounds_min	 = mesh.bounds_min,
				   .bounds_max	 = mesh.bounds_max};
}

std::expected<StlLodSet, StlError> load_stl_lods(graph::Graph& graph, const std::string& path,
												 std::span<const StlLodLevel> levels, const StlOptions& base)
{
	auto data = parse_to_mesh(path);
	if (!data.has_value())
	{
		return std::unexpected(data.error());
	}

	const StlLodLevel				   full{};
	const std::span<const StlLodLevel> effective = levels.empty() ? std::span<const StlLodLevel>(&full, 1) : levels;

	// Prepare once (weld + normals), then finalise per level — no re-parsing, no re-welding.
	const ProcessOptions po	  = to_process_options(base);
	const PreparedMesh	 prep = prepare(*data, po);
	if (prep.mesh.positions.empty())
	{
		return std::unexpected(StlError::NoGeometry);
	}

	StlLodSet set;
	set.bounds_min = prep.bounds_min;
	set.bounds_max = prep.bounds_max;
	set.meshes.reserve(effective.size());
	for (const StlLodLevel& level : effective)
	{
		set.meshes.push_back(add_mesh_node(graph, finalize(prep, level, po)));
	}
	return set;
}
} // namespace veng::assets
