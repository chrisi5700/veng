# Findings — out-of-scope observations

Things noticed while implementing `review.md` that warrant future attention but are out of
scope for the current work. Newest entries appended at the bottom.

Format: **[area]** observation — why it matters / suggested action.

---

**[branding] Stale "[IteratedFunction]" prefix in the Vulkan debug callback.**
`src/context/Context.cpp:80` formats validation-layer messages with `"[IteratedFunction]..."`,
a leftover from a prior project (same class as the `ITERATEDFUNCTIONS_` Logger include guard
already cleaned up). Cosmetic; should read `[veng]`. Out of scope for the asset work.

**[validation] GraphicsNode stride check covers only vertex binding 0.**
The new `MeshRef::vertex_stride` assert (review.md item 2) validates binding 0, which is the
only vertex-rate binding today (instance data arrives via SSBO, not a second vertex binding).
If a future zero-copy instance-attribute path adds a per-instance vertex binding (binding 1),
the check should extend to all reflected bindings. Noted, not needed now.

**[color] sRGB swapchain shifts the existing flat-colour demos.**
review.md item 8 fix landed: the swapchain is now `B8G8R8A8_SRGB`, so the final write
hardware-encodes linear -> sRGB (correct once lighting is linear). But the pre-existing demos
(`static_shapes`, `instanced_cubes`, `phong_materials`) output already-perceptual colours that
are now sRGB-encoded again on present, so they appear brighter/washed. The proper fix is to make
those demos render in linear light and rely on the encode — i.e. convert them the way the new PBR
path is built. Cosmetic, examples only; no test covers swapchain pixels. Out of scope here.

**[color] No linear/HDR intermediate or tonemap pass.**
review.md item 8 calls for intermediate render targets to hold linear (ideally HDR
`R16G16B16A16_SFLOAT`) values with a tonemap (Reinhard/ACES) before the sRGB encode — explicitly
a follow-on. Today scene targets share the swapchain's `_SRGB` format, so: (a) there is no HDR
headroom (lighting clamps at 1.0 before encode), and (b) transparency blending happens in
sRGB-encoded space, which is not physically correct (linear blending wants a linear/HDR target).
The PhongPass transparent batches blend into an sRGB attachment under the new swapchain format.
Deferred per review.md "follow-ons"; revisit with a tonemap fullscreen pass + a linear scene
target decoupled from the swapchain format (AppLoop::scene_color_format).

**[pbr] FIXED — PbrPass now handles glTF alpha BLEND + MASK.**
MASK discards in `pbr.frag` (cutoff packed into the unused `emissive.w`); BLEND uses PhongPass's
opaque / transparent-far (eFront) / transparent-near (eBack) three-pipeline batching, blending into
the (now linear, HDR) scene target. `GltfMaterialDesc` carries `alpha_mode`/`alpha_cutoff` (a
`assets::AlphaMode` mirror of `passes::AlphaMode`, bridged by a static_cast the caller static_asserts).
Tested headless (PbrPassTests `[alpha]`). Remaining limitations, all out of scope for the happy case:
(a) **no per-object back-to-front sort** — transparent objects blend in submission order, so
overlapping transparent *objects* can mis-order (the far/near trick only fixes a single convex
object); add a centroid-distance sort when a scene needs it. (b) **no `doubleSided`** — the BLEND
trick assumes single-sided; a two-sided transparent material wants a no-cull blend pass. (c) all
fetched acceptance models are opaque, so BLEND/MASK have no live example asset yet.

**[pbr] Normal transform assumes uniform scale.**
`pbr.vert` transforms normals/tangents by the model 3x3, which is only correct for uniform-scale
(or rotation-only) transforms. A glTF node with non-uniform scale needs the inverse-transpose
normal matrix passed alongside the model (an extra push-constant mat3 or a precomputed normal
matrix). Most assets use uniform scale; flag for when one doesn't shade right.

**[pbr] LANDED — clustered-forward point lighting (+ a found/fixed pool bug).**
PbrPass now shades a directional key light + N clustered point lights. The decomposition is composable
graph nodes: `veng::culling` holds the pure froxel-AABB build + sphere-vs-AABB assignment (CPU, unit
tested — the build/analytic-mapping consistency round-trip is the linchpin); `passes::wire_clustered_lights`
wires FroxelGridCpu + LightCullCpu as `add_transform`s feeding `StorageBufferNode` uploads, so only
lights/grid/index reach the GPU (the shader reconstructs its froxel analytically). The nodes are named
for where they run so a `LightCullGpu` is a drop-in swap behind the same three SSBO edges.
Found+fixed along the way: a buffer **resize** (`ResourcePool::acquire_buffer`) destroyed copies an
in-flight frame still referenced via its descriptor set (VUID-vkDestroyBuffer-00922), because — unlike
an image resize — it isn't gated by a device-idle. The fix has two parts. (1) Resized-away copies go to
a retirement graveyard freed only past the in-flight window, and the purge MUST run during node record
(inside `acquire_buffer`), NOT in `begin_frame`: `FrameExecutor` calls `begin_frame` *before*
`swap->acquire` waits the slot's in-flight fence, so `retired_through()` is only accurate once recording
has begun — purging in `begin_frame` frees a buffer a frame still executing on the GPU uses (this was the
actual repro: on-demand drag errored, continuous didn't, because continuous races the GPU far enough
ahead that the old frame happens to be done). (2) `BufferRef` gained a `pool_id` so SSBO consumers
`pool.consume()` to retain a cached producer's copy (the buffer analogue of the image path). Reproduced
headlessly with `VENG_AUTO_ORBIT` (20 errors buggy → 0 fixed); covered by NBufferingTests `[resize]`.
Remaining follow-ons: point + directional only (**no spot/cone** lights yet — GpuLight has position+
radius, add a direction+cone-angle for spots); the cull is **O(lights × clusters)** on the CPU (fine for
the demo's handful of lights, reactive-cached when static — wire `LightCullGpu` when light counts grow);
and froxel z-slicing uses a fixed `ClusterGrid.z_near/z_far` the caller sets per scene.

**[pbr] No IBL / environment lighting.** Single hard-coded directional light + flat ambient only.
review.md defers IBL explicitly; a real PBR look wants a prefiltered environment map + irradiance.
Out of scope.

**[pbr] Per-material descriptor sets are not deduplicated.**
PbrRenderNode allocates one descriptor set per material per frame slot. A model with many
materials that share the same (e.g. default) textures allocates redundant identical sets. Could
key sets by their texture tuple. Minor; only matters for very-many-material scenes.

**[gltf] Static only — no skinning / animation / morph targets.**
The loader (review.md items 6+7) flattens the node tree to static world matrices and ignores
`asset.animations`, skins, and morph targets. review.md scopes the acceptance to a static PBR
mesh; animated/skinned glTF is a follow-on (needs joint matrices + a skinning vertex path).

**[gltf] Tangent generation is a simple accumulate, not mikktspace.**
When a primitive has a normal map but no TANGENT, `compute_tangents` accumulates per-triangle
tangents and orthonormalizes — adequate, but not the mikktspace algorithm glTF normal maps are
authored against, so normal-mapped detail can differ subtly at UV seams. All acceptance models
that use normal maps (NormalTangentTest, DamagedHelmet) ship TANGENT, so this path is the
fallback. Wire mikktspace if a no-tangent normal-mapped asset looks wrong.

**[gltf] No mesh instancing/dedup.**
Each primitive instance becomes its own MeshNode (its own GPU upload), even when several glTF
nodes reference the same mesh. A scene that reuses one mesh across many nodes re-uploads it per
node. Dedup by mesh index (upload once, many model-matrix objects) when instanced scenes matter.

**[gltf] External-image (.gltf + separate texture files) path is untested.**
`image_bytes` handles inline (Array/Vector/ByteView) and buffer-view (GLB) image sources; external
URI images rely on `Options::LoadExternalImages` turning them into `sources::Array`. All five
acceptance models are self-contained `.glb`, so the separate-file `.gltf` case has no test
coverage. Verify with a `.gltf`+PNG asset before relying on it; a surviving `sources::URI` returns
empty bytes -> TextureLoad error.

**[pbr] FIXED — PbrPass front-face winding is eCounterClockwise (opposite PhongPass).**
The windowed `gltf_viewer` showed only the *inside* of the helmet: PbrPass had copied PhongPass's
`vk::FrontFace::eClockwise`, but that winding is for the engine's hand-authored meshes, which wind
their outward faces CW (`example/common/Geometry.cpp` — the `-z` cube face triangle `(0,1,2)` has a
geometric normal of `+z`, opposite its outward `-z`). glTF defines front faces as CCW-wound (spec
3.7.2.1), the opposite, so `eBack` was culling the model's outer shell. Now `eCounterClockwise`
(`src/passes/PbrPass.cpp` `ensure_pipeline`). The integration test uses `cull_mode = eNone`, so it
never caught the facing bug — a windowed-only regression. Worth a headless facing test (render a
closed glTF mesh with `eBack` and assert the silhouette is filled, not the backface interior) if
PbrPass culling regresses again.

**[examples] Cosmetic "vertex attribute not consumed" pipeline warnings (phong/picking).**
`vkCreateGraphicsPipelines` warns `Vertex attribute at location N not consumed by vertex shader` in
phong_materials (loc 2) and picking_outline (loc 1, 2): the engine's `example::Vertex` carries
position/normal/colour, but those passes' vertex shaders read a subset. Harmless (the attribute is
just ignored), pre-existing, and orthogonal to the PBR work — silence by trimming the unused inputs
from the shader's declared `VSIn` or by giving those examples a slimmer vertex type. Out of scope.

**[examples] FIXED — empty-SSBO zero range + teardown image-view destroy.**
Two validation errors surfaced while testing all examples after the lighting work, both fixed at the
root. (1) `StorageBufferNode` published `BufferRef.size = 0` for an empty array (phong_materials' three
Transform SSBOs start empty until the sim spawns objects), and a 0 descriptor range is invalid
(VUID-VkDescriptorBufferInfo-range-00341); it now publishes the allocated size (>= one stride), with
`count = 0` still driving zero-instance draws. (2) `AppLoop::run()` returned with the last frame still
in flight, so an example's `assets::Texture`/`GltfModel` (declared after `app`, destroyed before
`~AppLoop`'s waitIdle) freed image views the in-flight descriptor sets referenced
(VUID-vkDestroyImageView-01026); `run()` now waits the device idle before returning. Added headless
test affordances `VENG_AUTO_ORBIT` (spin the camera each frame) and `VENG_RUN_SECONDS` (auto-close) so
the camera-motion and clean-shutdown paths are reproducible without a mouse/CI.

**[examples] Windowed `pbr_materials` re-renders a static scene instead of idling.**
The `pbr_materials` example presents continuously (~220 fps) while the camera and scene are
static, where the engine's on-demand caching should idle it to ~0 (as `static_shapes` does).
This is NOT a PbrPass/core caching bug: the headless test "PbrPass caches a static multi-material
scene" proves a second `resolve()` returns an empty plan. So the cause is in the windowed driver /
example wiring (the single-material `gltf_viewer` idles to ~1 fps, the 36-material grid does not) —
likely the `FrameExecutor` OnDemand path or a per-frame source touch the example introduces.
Visual output is correct; this is an efficiency regression to chase in the windowed loop, not the
reactive core. Investigate with a frame-by-frame plan dump on the windowed path.
