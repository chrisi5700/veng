/**
 * @file
 * @author chris
 * @brief Implementation of the shared mesh pipeline (@ref veng::assets::prepare / finalize / process).
 * @ingroup assets
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <unordered_map>
#include <utility>
#include <vector>
#include <veng/assets/MeshData.hpp>

#include "MeshProcessing.hpp"

namespace veng::assets
{
namespace
{
// --- Welding (soup → shared positions) --------------------------------------------------------
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
		std::size_t h = 1469598103934665603ULL;
		for (const std::int64_t v : {k.x, k.y, k.z})
		{
			h = (h ^ static_cast<std::size_t>(v)) * 1099511628211ULL;
		}
		return h;
	}
};

// A smoothing cluster of incident faces at one welded position (area-weighted normal + seed for the
// crease test).
struct NormalCluster
{
	glm::vec3 sum  = glm::vec3(0.0F);
	glm::vec3 seed = glm::vec3(0.0F);
};

// Box / triplanar UV: project onto whichever world-axis plane the normal most faces, scaled to the
// requested density. A sign flip keeps opposite faces from mirroring into each other.
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

// Per-vertex tangents (xyz + handedness) from positions/normals/UVs — accumulate-then-orthonormalize
// (not mikktspace). Used whenever a mesh lacks authored tangents.
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

void compute_bounds(const std::vector<glm::vec3>& positions, glm::vec3& bmin, glm::vec3& bmax)
{
	bmin = glm::vec3(std::numeric_limits<float>::max());
	bmax = glm::vec3(std::numeric_limits<float>::lowest());
	for (const glm::vec3& p : positions)
	{
		bmin = glm::min(bmin, p);
		bmax = glm::max(bmax, p);
	}
}

float max_extent_of(const glm::vec3& bmin, const glm::vec3& bmax)
{
	return std::max({bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z, 1e-6F});
}

std::vector<detail::Face> faces_from_indices(const std::vector<std::uint32_t>& indices)
{
	std::vector<detail::Face> faces;
	faces.reserve(indices.size() / 3);
	for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
	{
		faces.push_back(detail::Face{indices[i], indices[i + 1], indices[i + 2]});
	}
	return faces;
}

std::vector<std::uint32_t> indices_from_faces(const std::vector<detail::Face>& faces)
{
	std::vector<std::uint32_t> indices;
	indices.reserve(faces.size() * 3);
	for (const detail::Face& f : faces)
	{
		indices.insert(indices.end(), {f[0], f[1], f[2]});
	}
	return indices;
}

// Soup → welded positions + crease-smoothed normals (splitting a position into several output
// vertices across hard edges). This is the STL path: there are no authored attributes to keep.
void prepare_soup(const MeshData& data, const ProcessOptions& opts, PreparedMesh& out)
{
	const float max_extent = max_extent_of(out.bounds_min, out.bounds_max);
	const float quant	   = max_extent * 1e-5F;

	std::vector<glm::vec3>								   positions;
	std::unordered_map<QuantKey, std::uint32_t, QuantHash> lookup;
	const auto											   to_key = [&](const glm::vec3& p)
	{
		return QuantKey{.x = static_cast<std::int64_t>(std::llround(p.x / quant)),
						.y = static_cast<std::int64_t>(std::llround(p.y / quant)),
						.z = static_cast<std::int64_t>(std::llround(p.z / quant))};
	};
	std::vector<detail::Face> raw_faces;
	raw_faces.reserve(data.positions.size() / 3);
	for (std::size_t i = 0; i + 2 < data.positions.size(); i += 3)
	{
		detail::Face idx{};
		for (int c = 0; c < 3; ++c)
		{
			const QuantKey key		  = to_key(data.positions[i + static_cast<std::size_t>(c)]);
			const auto [it, inserted] = lookup.try_emplace(key, static_cast<std::uint32_t>(positions.size()));
			if (inserted)
			{
				positions.push_back(data.positions[i + static_cast<std::size_t>(c)]);
			}
			idx[static_cast<std::size_t>(c)] = it->second;
		}
		raw_faces.push_back(idx);
	}

	if (opts.repair)
	{
		static_cast<void>(detail::repair_orientation(positions, raw_faces));
	}

	// Per-face normal + area from the (repaired) winding; drop geometric slivers.
	struct Face
	{
		detail::Face pos;
		glm::vec3	 normal;
		float		 area;
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

	std::vector<std::vector<std::pair<std::uint32_t, int>>> incident(positions.size());
	for (std::uint32_t f = 0; f < faces.size(); ++f)
	{
		for (int c = 0; c < 3; ++c)
		{
			incident[faces[f].pos[static_cast<std::size_t>(c)]].emplace_back(f, c);
		}
	}

	const float				  cos_crease = std::cos(opts.crease_angle_deg * std::numbers::pi_v<float> / 180.0F);
	std::vector<glm::vec3>	  out_pos;
	std::vector<glm::vec3>	  out_nrm;
	std::vector<detail::Face> out_index(faces.size());
	for (std::uint32_t p = 0; p < positions.size(); ++p)
	{
		std::vector<NormalCluster> clusters;
		std::vector<std::uint32_t> cluster_vtx;
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
				out_nrm.emplace_back(0.0F);
			}
			clusters[static_cast<std::size_t>(slot)].sum += fn * w;
			out_nrm[cluster_vtx[static_cast<std::size_t>(slot)]] += fn * w;
			out_index[f][static_cast<std::size_t>(c)] = cluster_vtx[static_cast<std::size_t>(slot)];
		}
	}
	for (glm::vec3& n : out_nrm)
	{
		n = (glm::dot(n, n) < 1e-12F) ? glm::vec3(0.0F, 1.0F, 0.0F) : glm::normalize(n);
	}

	out.mesh.positions = std::move(out_pos);
	out.mesh.normals   = std::move(out_nrm);
	out.mesh.indices   = indices_from_faces(out_index);
}

// Smooth per-vertex normals over an existing indexed topology (no vertex splitting), for indexed
// meshes whose format omitted normals (e.g. OBJ without `vn`).
std::vector<glm::vec3> smooth_normals(const std::vector<glm::vec3>&		positions,
									  const std::vector<std::uint32_t>& indices)
{
	std::vector<glm::vec3> normals(positions.size(), glm::vec3(0.0F));
	for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
	{
		const std::uint32_t i0 = indices[i];
		const std::uint32_t i1 = indices[i + 1];
		const std::uint32_t i2 = indices[i + 2];
		const glm::vec3		fn = glm::cross(positions[i1] - positions[i0], positions[i2] - positions[i0]);
		normals[i0] += fn;
		normals[i1] += fn;
		normals[i2] += fn;
	}
	for (glm::vec3& n : normals)
	{
		n = (glm::dot(n, n) < 1e-12F) ? glm::vec3(0.0F, 1.0F, 0.0F) : glm::normalize(n);
	}
	return normals;
}

// Indexed path (glTF/OBJ/PLY/3MF): vertices stay as authored (so UVs/colours/tangents keep aligning);
// only winding is repaired and normals are filled in when the format omitted them.
void prepare_indexed(const MeshData& data, const ProcessOptions& opts, PreparedMesh& out)
{
	std::vector<glm::vec3>	  positions = data.positions;
	std::vector<detail::Face> faces		= faces_from_indices(data.indices);
	const bool				  synth		= !data.has_normals();
	if (opts.repair && synth) // trust an authored mesh's winding; repair only when we generate normals
	{
		static_cast<void>(detail::repair_orientation(positions, faces));
	}
	std::vector<std::uint32_t> indices = indices_from_faces(faces);

	out.mesh.positions = std::move(positions);
	out.mesh.indices   = indices;
	out.mesh.normals   = synth ? smooth_normals(out.mesh.positions, indices) : data.normals;
	out.mesh.uvs	   = data.uvs; // aligned to positions — carried unchanged
	out.mesh.tangents  = data.tangents;
	out.mesh.colors	   = data.colors;
}

template <class T>
std::vector<T> remap(const std::vector<T>& src, const detail::DecimateResult& dec)
{
	return src.empty() ? src : detail::apply_vertex_remap(src, dec.remap, dec.vertex_count);
}
} // namespace

PreparedMesh prepare(const MeshData& data, const ProcessOptions& opts)
{
	PreparedMesh out;
	if (data.positions.empty())
	{
		return out;
	}
	compute_bounds(data.positions, out.bounds_min, out.bounds_max);
	if (data.indexed())
	{
		prepare_indexed(data, opts, out);
	}
	else
	{
		prepare_soup(data, opts, out);
	}
	return out;
}

ProcessedMesh finalize(const PreparedMesh& prep, const LodLevel& level, const ProcessOptions& opts)
{
	std::vector<glm::vec3>	   positions = prep.mesh.positions;
	std::vector<glm::vec3>	   normals	 = prep.mesh.normals;
	std::vector<glm::vec2>	   uvs		 = prep.mesh.uvs;	   // may be empty ⇒ synthesise below
	std::vector<glm::vec4>	   tangents	 = prep.mesh.tangents; // may be empty ⇒ compute below
	std::vector<std::uint32_t> indices	 = prep.mesh.indices;

	// 1) Decimate (feeding normals + authored UVs into the metric); remap every per-vertex array.
	if (level.target_ratio < 1.0F && !indices.empty())
	{
		const detail::DecimateResult dec = detail::decimate(positions, normals, indices,
															detail::DecimateOptions{.target_ratio = level.target_ratio,
																					.max_error	  = level.max_error,
																					.aggressive	  = level.aggressive},
															uvs);
		positions						 = remap(positions, dec);
		normals							 = remap(normals, dec);
		uvs								 = remap(uvs, dec);
		tangents						 = remap(tangents, dec);
		indices							 = dec.indices;
	}

	// 2) Synthesise box UVs on the (possibly slimmed) mesh when the format had none.
	if (uvs.empty())
	{
		const float uv_scale = opts.world_units_per_tile > 0.0F
								   ? 1.0F / opts.world_units_per_tile
								   : opts.texture_tiles / max_extent_of(prep.bounds_min, prep.bounds_max);
		uvs.resize(positions.size());
		for (std::size_t v = 0; v < positions.size(); ++v)
		{
			uvs[v] = box_uv(positions[v], normals[v], uv_scale);
		}
	}

	// 3) Compute tangents when absent.
	if (tangents.empty())
	{
		tangents = compute_tangents(positions, normals, uvs, indices);
	}

	// 4) Pack PbrVertices (per-vertex colour, if any, has no slot in PbrVertex yet — dropped).
	ProcessedMesh out;
	out.bounds_min = prep.bounds_min;
	out.bounds_max = prep.bounds_max;
	out.indices	   = std::move(indices);
	out.vertices.reserve(positions.size());
	for (std::size_t v = 0; v < positions.size(); ++v)
	{
		out.vertices.push_back(
			gpu::PbrVertex{.position = positions[v], .normal = normals[v], .tangent = tangents[v], .uv = uvs[v]});
	}
	return out;
}

ProcessedMesh process(const MeshData& data, const ProcessOptions& opts)
{
	return finalize(prepare(data, opts), LodLevel{}, opts);
}
} // namespace veng::assets
