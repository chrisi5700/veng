/**
 * @file
 * @author chris
 * @brief Shared level-of-detail types used by the asset loaders.
 * @ingroup assets
 */

#ifndef VENG_ASSETS_LOD_HPP
#define VENG_ASSETS_LOD_HPP

namespace veng::assets
{
/**
 * @brief One level of a LOD chain: the decimation applied to produce it from the base mesh.
 *
 * Format-agnostic — both @ref load_stl_lods and @ref load_gltf_lods take a sequence of these
 * (finest first; a full-detail level is just `target_ratio = 1.0`).
 *
 * @ingroup assets
 */
struct LodLevel
{
	float target_ratio = 1.0F;	///< Fraction of triangles to keep (1.0 = full detail).
	float max_error	   = 0.05F; ///< Decimation error budget (fraction of mesh size).
	bool  aggressive   = false; ///< Use the faster sloppy simplifier.
};
} // namespace veng::assets

#endif // VENG_ASSETS_LOD_HPP
