/**
 * @file
 * @author chris
 * @brief @ref veng::assets::load_stl implementation: ASCII/binary STL parsing plus synthesis of the
 *        welded, crease-smoothed, box-UV'd, tangent-framed geometry a PBR pass needs.
 * @ingroup assets
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <numbers>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <veng/assets/StlLoader.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/gpu/Vertex.hpp>
#include <veng/nodes/MeshNode.hpp>
#include <veng/rendergraph/data/Data.hpp>

#include "MeshProcessing.hpp"

namespace veng::assets
{
namespace
{
// One parsed triangle: three corner positions, straight from the file (winding preserved).
struct Triangle
{
	std::array<glm::vec3, 3> p;
};

// --- Parsing ----------------------------------------------------------------------------------

// Binary STL is recognised structurally, not by header text: the 80-byte header is followed by a
// uint32 triangle count and exactly 50 bytes per triangle. A header beginning with "solid" is NOT
// a reliable ASCII signal (many exporters write "solid" into binary headers), so we trust the size.
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
// "normal" lines are ignored — normals are resynthesised below with proper crease handling.
std::vector<Triangle> parse_ascii(const std::vector<std::byte>& bytes)
{
	const char* cur = static_cast<const char*>(static_cast<const void*>(bytes.data()));
	const char* end = cur + bytes.size();

	std::vector<glm::vec3> verts;
	while (cur < end)
	{
		// Find the next "vertex" keyword.
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

// --- Geometry synthesis -----------------------------------------------------------------------

// Quantise a position to an integer grid so corners that should coincide hash equal despite
// float/text round-off. The grid step is relative to the model extent (eps · max_extent).
struct QuantKey
{
	std::int64_t x;
	std::int64_t y;
	std::int64_t z;
	friend bool	 operator==(const QuantKey&, const QuantKey&) noexcept = default;
};
struct QuantHash
{
	std::size_t operator()(const QuantKey& k) const noexcept
	{
		// FNV-ish mix of the three lanes.
		std::size_t h = 1469598103934665603ULL;
		for (const std::int64_t v : {k.x, k.y, k.z})
		{
			h = (h ^ static_cast<std::size_t>(v)) * 1099511628211ULL;
		}
		return h;
	}
};

// A smoothing cluster of incident faces at one welded position: their area-weighted normal sum,
// plus the seed normal new faces are tested against for the crease threshold.
struct NormalCluster
{
	glm::vec3 sum  = glm::vec3(0.0F);
	glm::vec3 seed = glm::vec3(0.0F);
};

struct BuiltMesh
{
	std::vector<gpu::PbrVertex> vertices;
	std::vector<std::uint32_t>	indices;
	glm::vec3					bounds_min{};
	glm::vec3					bounds_max{};
};

// Box / triplanar UV for a vertex: project the position onto whichever world-axis plane the normal
// most faces, scaled to the requested density. The sign of the dominant axis flips one coordinate
// so opposite faces don't mirror into each other.
glm::vec2 box_uv(const glm::vec3& pos, const glm::vec3& n, float scale)
{
	const glm::vec3 a = glm::abs(n);
	glm::vec2		uv;
	if (a.x >= a.y && a.x >= a.z)
	{
		uv = glm::vec2(n.x < 0.0F ? -pos.z : pos.z, pos.y);
	}
	else if (a.y >= a.z)
	{
		uv = glm::vec2(pos.x, n.y < 0.0F ? -pos.z : pos.z);
	}
	else
	{
		uv = glm::vec2(n.z < 0.0F ? -pos.x : pos.x, pos.y);
	}
	return uv * scale;
}

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
			t = std::abs(n.x) < 0.9F ? glm::cross(n, glm::vec3(1, 0, 0)) : glm::cross(n, glm::vec3(0, 1, 0));
		}
		t					   = glm::normalize(t);
		const float handedness = (glm::dot(glm::cross(n, t), bitan[v]) < 0.0F) ? -1.0F : 1.0F;
		result[v]			   = glm::vec4(t, handedness);
	}
	return result;
}

// Weld facet corners, split/smooth normals by the crease angle, then project UVs and a tangent
// frame. Returns a renderable PbrVertex buffer + index buffer (or no indices if all degenerate).
BuiltMesh build(const std::vector<Triangle>& tris, const StlOptions& opts)
{
	// 1) Bounds (drive both the weld tolerance and the returned framing box).
	glm::vec3 bmin(std::numeric_limits<float>::max());
	glm::vec3 bmax(std::numeric_limits<float>::lowest());
	for (const Triangle& tri : tris)
	{
		for (const glm::vec3& p : tri.p)
		{
			bmin = glm::min(bmin, p);
			bmax = glm::max(bmax, p);
		}
	}
	const float max_extent = std::max({bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z, 1e-6F});
	const float quant	   = max_extent * 1e-5F; // weld grid step

	// 2) Weld positions: map each corner to a shared position index.
	std::vector<glm::vec3>								   positions;
	std::unordered_map<QuantKey, std::uint32_t, QuantHash> pos_lookup;
	const auto											   to_key = [&](const glm::vec3& p)
	{
		return QuantKey{.x = static_cast<std::int64_t>(std::llround(p.x / quant)),
						.y = static_cast<std::int64_t>(std::llround(p.y / quant)),
						.z = static_cast<std::int64_t>(std::llround(p.z / quant))};
	};

	std::vector<detail::Face> raw_faces;
	raw_faces.reserve(tris.size());
	for (const Triangle& tri : tris)
	{
		detail::Face idx{};
		for (int c = 0; c < 3; ++c)
		{
			const QuantKey key		  = to_key(tri.p[static_cast<std::size_t>(c)]);
			const auto [it, inserted] = pos_lookup.try_emplace(key, static_cast<std::uint32_t>(positions.size()));
			if (inserted)
			{
				positions.push_back(tri.p[static_cast<std::size_t>(c)]);
			}
			idx[static_cast<std::size_t>(c)] = it->second;
		}
		raw_faces.push_back(idx);
	}

	// 2b) Automatic repair: drop duplicate/degenerate triangles, unify winding, orient outward — so
	//     the face normals below are trustworthy (and the caller can backface-cull). Runs in place.
	if (opts.repair)
	{
		[[maybe_unused]] const detail::RepairStats stats = detail::repair_orientation(positions, raw_faces);
	}

	// 2c) Per-face normal + area from the (repaired) winding. Geometric slivers contribute no normal.
	struct Face
	{
		std::array<std::uint32_t, 3> pos;	 ///< welded position index per corner
		glm::vec3					 normal; ///< geometric face normal
		float						 area;	 ///< triangle area (normal weight)
	};
	std::vector<Face> faces;
	faces.reserve(raw_faces.size());
	for (const detail::Face& idx : raw_faces)
	{
		const glm::vec3 p0	  = positions[idx[0]];
		const glm::vec3 cross = glm::cross(positions[idx[1]] - p0, positions[idx[2]] - p0);
		const float		len	  = glm::length(cross);
		if (len < 1e-12F)
		{
			continue;
		}
		faces.push_back(Face{.pos = idx, .normal = cross / len, .area = 0.5F * len});
	}

	// 3) Incident faces per welded position (face index + which corner touches it).
	std::vector<std::vector<std::pair<std::uint32_t, int>>> incident(positions.size());
	for (std::uint32_t f = 0; f < faces.size(); ++f)
	{
		for (int c = 0; c < 3; ++c)
		{
			incident[faces[f].pos[static_cast<std::size_t>(c)]].emplace_back(f, c);
		}
	}

	// 4) Crease-angle smoothing by normal clustering. Each welded position spawns one output vertex
	//    per cluster of incident faces whose normals lie within the crease cone; corners map to the
	//    output vertex of their cluster. Threads/curves smooth; machined edges stay hard.
	const float cos_crease = std::cos(opts.crease_angle_deg * std::numbers::pi_v<float> / 180.0F);

	std::vector<glm::vec3> out_pos;
	std::vector<glm::vec3> out_nrm;
	out_pos.reserve(positions.size());
	out_nrm.reserve(positions.size());
	// out_index[f][c] = output vertex for face f corner c.
	std::vector<std::array<std::uint32_t, 3>> out_index(faces.size());

	for (std::uint32_t p = 0; p < positions.size(); ++p)
	{
		std::vector<NormalCluster> clusters;
		std::vector<std::uint32_t> cluster_vtx; // output vertex id per cluster (assigned lazily)
		for (const auto& [f, c] : incident[p])
		{
			const glm::vec3 fn	 = faces[f].normal;
			const float		w	 = faces[f].area;
			int				slot = -1;
			for (std::size_t k = 0; k < clusters.size(); ++k)
			{
				if (glm::dot(clusters[k].seed, fn) >= cos_crease)
				{
					slot = static_cast<int>(k);
					break;
				}
			}
			if (slot < 0)
			{
				slot = static_cast<int>(clusters.size());
				clusters.push_back(NormalCluster{.sum = glm::vec3(0.0F), .seed = fn});
				cluster_vtx.push_back(static_cast<std::uint32_t>(out_pos.size()));
				out_pos.push_back(positions[p]);
				out_nrm.emplace_back(0.0F); // accumulated below, normalised after
			}
			clusters[static_cast<std::size_t>(slot)].sum += fn * w;
			out_nrm[cluster_vtx[static_cast<std::size_t>(slot)]] += fn * w;
			out_index[f][static_cast<std::size_t>(c)] = cluster_vtx[static_cast<std::size_t>(slot)];
		}
	}

	// 5) Finalise normals; fall back to +Y if a vertex somehow accumulated nothing.
	for (glm::vec3& n : out_nrm)
	{
		n = (glm::dot(n, n) < 1e-12F) ? glm::vec3(0.0F, 1.0F, 0.0F) : glm::normalize(n);
	}

	// 6) Index buffer from the per-corner output vertices.
	std::vector<std::uint32_t> indices;
	indices.reserve(faces.size() * 3);
	for (const auto& tri : out_index)
	{
		indices.push_back(tri[0]);
		indices.push_back(tri[1]);
		indices.push_back(tri[2]);
	}

	// 6b) Optional decimation — slim an over-tessellated mesh *before* UVs exist, so the box UVs and
	//     tangents below are synthesised on the final slimmed geometry (texturing unaffected). Normals
	//     feed the collapse metric to keep the crease edges. Identity ratio (1.0) is a no-op.
	if (opts.decimate_ratio < 1.0F && !indices.empty())
	{
		const detail::DecimateResult dec =
			detail::decimate(out_pos, out_nrm, indices,
							 detail::DecimateOptions{.target_ratio = opts.decimate_ratio,
													 .max_error	   = opts.decimate_max_error,
													 .aggressive   = opts.decimate_aggressive});
		out_pos = detail::apply_vertex_remap(out_pos, dec.remap, dec.vertex_count);
		out_nrm = detail::apply_vertex_remap(out_nrm, dec.remap, dec.vertex_count);
		indices = dec.indices;
	}

	// 7) Box UVs + tangents. UV density is resolved here so it stays unit-agnostic: an absolute
	//    world-units-per-tile if the caller set one, otherwise `texture_tiles` repeats across the
	//    mesh's longest extent (so the result is sane whatever units the file used).
	const float uv_scale =
		opts.world_units_per_tile > 0.0F ? 1.0F / opts.world_units_per_tile : opts.texture_tiles / max_extent;
	std::vector<glm::vec2> uvs(out_pos.size());
	for (std::size_t v = 0; v < out_pos.size(); ++v)
	{
		uvs[v] = box_uv(out_pos[v], out_nrm[v], uv_scale);
	}
	const std::vector<glm::vec4> tangents = compute_tangents(out_pos, out_nrm, uvs, indices);

	// 8) Pack into PbrVertices.
	BuiltMesh mesh;
	mesh.bounds_min = bmin;
	mesh.bounds_max = bmax;
	mesh.indices	= std::move(indices);
	mesh.vertices.reserve(out_pos.size());
	for (std::size_t v = 0; v < out_pos.size(); ++v)
	{
		mesh.vertices.push_back(
			gpu::PbrVertex{.position = out_pos[v], .normal = out_nrm[v], .tangent = tangents[v], .uv = uvs[v]});
	}
	return mesh;
}
} // namespace

std::expected<StlMesh, StlError> load_stl(graph::Graph& graph, const std::string& path, const StlOptions& opts)
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

	const std::vector<Triangle> tris = looks_binary(bytes) ? parse_binary(bytes) : parse_ascii(bytes);
	if (tris.empty())
	{
		return std::unexpected(StlError::ParseFailed);
	}

	BuiltMesh built = build(tris, opts);
	if (built.vertices.empty() || built.indices.empty())
	{
		return std::unexpected(StlError::NoGeometry);
	}

	const graph::DataHandle mesh = graph.add(std::make_unique<graph::ValueData<gpu::MeshRef>>(gpu::MeshRef{}));
	const graph::NodeHandle node = graph.add(std::make_unique<nodes::MeshNode>(
		std::span<const gpu::PbrVertex>(built.vertices), std::span<const std::uint32_t>(built.indices), mesh));
	graph.set_producer(mesh, node);

	return StlMesh{.mesh		 = mesh,
				   .vertex_count = static_cast<std::uint32_t>(built.vertices.size()),
				   .index_count	 = static_cast<std::uint32_t>(built.indices.size()),
				   .bounds_min	 = built.bounds_min,
				   .bounds_max	 = built.bounds_max};
}
} // namespace veng::assets
