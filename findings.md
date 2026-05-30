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

**[pbr] PbrPass is opaque-only; no glTF alpha BLEND / MASK.**
The pass builds a single opaque pipeline (depth write, no blend). glTF materials with
`alphaMode = BLEND` (transparency) or `MASK` (alpha cutoff) are rendered fully opaque. PhongPass
already has the opaque/transparent-back/transparent-front batching pattern to copy. Follow-on; the
acceptance model (DamagedHelmet) is opaque.

**[pbr] Normal transform assumes uniform scale.**
`pbr.vert` transforms normals/tangents by the model 3x3, which is only correct for uniform-scale
(or rotation-only) transforms. A glTF node with non-uniform scale needs the inverse-transpose
normal matrix passed alongside the model (an extra push-constant mat3 or a precomputed normal
matrix). Most assets use uniform scale; flag for when one doesn't shade right.

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

**[examples] Windowed `pbr_materials` re-renders a static scene instead of idling.**
The `pbr_materials` example presents continuously (~220 fps) while the camera and scene are
static, where the engine's on-demand caching should idle it to ~0 (as `static_shapes` does).
This is NOT a PbrPass/core caching bug: the headless test "PbrPass caches a static multi-material
scene" proves a second `resolve()` returns an empty plan. So the cause is in the windowed driver /
example wiring (the single-material `gltf_viewer` idles to ~1 fps, the 36-material grid does not) —
likely the `FrameExecutor` OnDemand path or a per-frame source touch the example introduces.
Visual output is correct; this is an efficiency regression to chase in the windowed loop, not the
reactive core. Investigate with a frame-by-frame plan dump on the windowed path.
