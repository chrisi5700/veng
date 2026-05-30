# Roadmap: displaying industry-standard scenes & objects

Goal: load and correctly render a standard asset (a textured, PBR glTF model with a
node hierarchy) instead of only procedural, flat-shaded geometry.

## Implementation status — ✅ DONE

All eight items below are implemented and tested (110/110 tests pass under the `llm-vcpkg`
preset; the acceptance gate — DamagedHelmet loads + renders — is green). Cut edges and
follow-ons are tracked in `findings.md`.

| # | Item | Landed |
|---|------|--------|
| 2 | MeshRef stride + draw-time validation | `gpu::MeshRef::vertex_stride`; GraphicsNode/PbrPass assert vs reflected binding 0 → typed error + log. Test: `MeshStrideTests`. |
| 1 | Vertex attributes (UV + tangent) | `gpu::PbrVertex` (pos/normal/tangent/uv, 48B) + `shaders/passes/pbr.vert`. Reflection-driven, zero pipeline change. |
| 3 | Texture-from-file loader + mips | `veng_assets`: `assets::Texture` (stb decode → staging → device-local → blit-down mip chain). `Image::create` gained `mip_levels`; `CommandManager::image_barrier_range`. Tests: `TextureTests`. |
| 4 | Configurable sampler | `gpu::SamplerConfig` (render_target/texture presets); `GraphicsNode::set_sampler`; PbrPass uses the texture preset. `samplerAnisotropy` enabled in Context. |
| 8 | Color management (sRGB) | Per-texture sRGB/UNORM format in the loader; swapchain → `B8G8R8A8_SRGB` for the output encode. Test: sRGB-decode case in `TextureTests`. HDR-intermediate + tonemap deferred (findings). |
| 5 | PBR material + shader | `passes::PbrPass` + `shaders/passes/pbr.{vert,frag}` (Cook-Torrance metallic-roughness, per-material descriptor sets, directional light). Test: `PbrPassTests`. |
| 6 | glTF loader | `assets::load_gltf` (fastgltf): primitives → MeshNodes, materials/textures → SSBO-free PBR materials, tangent gen fallback. Test: `GltfLoaderTests`. |
| 7 | Scene-graph flatten | Node tree → per-primitive world matrices (`fastgltf::getTransformMatrix`), fed as model-matrix sources. Covered by `GltfLoaderTests` (MetalRoughSpheres multi-mesh). |

Test infra: Khronos sample models fetched at configure time (SHA256-pinned, gitignored,
SKIP-if-offline) via `tests/assets/models.cmake`.

---


The headline finding from reviewing the code: **almost everything here is additive, not
architectural.** The reflection-driven pipeline, byte-erased upload nodes, the reactive
graph, and the N-buffered `ResourcePool` are exactly the load-bearing pieces, and they
already exist. The work is mostly new *producers* feeding the graph new `ImageRef` /
`MeshRef` / SSBO values, plus two small correctness fixes.

---

## Foundations already in place (no change needed)

- **Vertex layout is reflection-driven.** `GraphicsPipelineBuilder` reflects bindings,
  attributes, stride, offsets and formats from the vertex shader's input struct via Slang
  (`src/pipelines/GraphicsPipeline.cpp:199-211`, `include/veng/shader/Shader.hpp:54-86`).
  The pipeline takes **no** vertex-layout argument — the `.vert.slang` input struct *is*
  the layout declaration.
- **`MeshNode` is already a vertex-type template.** `template <class Vertex>`
  (`include/veng/nodes/MeshNode.hpp:45`) memcpys the bytes and type-erases to
  `std::vector<std::byte>`; indices are fixed `uint32`. Any vertex struct works today.
- **Per-draw materials & transforms** already exist: `phong_materials` indexes a material
  SSBO by draw; `MultiDraw` feeds per-draw model matrices. This is the glTF
  primitive→draw / material / node-transform pattern.
- **Reactive graph + ResourcePool + N-buffering** — the hard runtime parts are done. New
  textures/meshes are just new values flowing on graph edges.

---

## Work items

### 1. Vertex attributes (UV + tangent) — content only
**Status: no engine change required.**

- Define a richer vertex struct, e.g. `struct PbrVertex { glm::vec3 position, normal,
  tangent; glm::vec2 uv; glm::vec3 color; };`.
- Add a matching `mesh_pbr.vert.slang` with the corresponding `[[vk::location(N)]]`
  inputs. Reflection picks up the new stride/attributes automatically.

> Rejected design: a "vertex size argument" on the pipeline constructor. The stride is
> already reflected from the shader; a manual size would be a second source of truth that
> can silently disagree with the shader. Do not add it.

### 2. `MeshRef` stride + draw-time validation — correctness fix (do first)
**Status: latent bug. Small, self-contained (~20 lines). De-risks all of glTF.**

Today nothing checks that the bytes a `MeshNode` uploaded match the stride the bound
shader expects — upload 36-byte verts, bind a 56-byte-input shader, and you get garbage
strided through the buffer with **no error**. `MeshRef` doesn't even carry the stride
(`include/veng/gpu/MeshRef.hpp:26-36` has `vertex_count`, no byte stride).

- Add `std::uint32_t vertex_stride` to `MeshRef` (`MeshNode` knows it: `sizeof(Vertex)`).
- In `GraphicsNode`'s draw, assert `mesh.vertex_stride == reflected_binding.stride` →
  return a typed `ExecError` instead of corrupting silently.

### 3. Texture-from-file loader — the one genuinely new component
**Status: new machinery, but produces a plain `gpu::ImageRef` so nothing downstream
changes.**

- Add an image-decode dependency (`stb_image`; consider KTX2 later for compressed +
  pre-baked mips). Currently `vcpkg.json` has none.
- New upload path: staging buffer → device-local `Image` → **mip generation** (blit down
  the chain). There is no mip support anywhere today. `ResourcePool::constant_image`
  (`src/resources/ResourcePool.cpp:207`) is the template for the allocate + immediate-
  submit transition, but it is clear-color only.
- **Per-texture format choice (sRGB vs UNORM)** — see item 7; the loader is where this
  lives.

### 4. Configurable sampler
**Status: small.**

The single sampler (`src/nodes/GraphicsNode.cpp:121`) is `ClampToEdge`, linear, **no mip
LOD, no anisotropy** — built only for sampling render targets. Assets need `Repeat`,
trilinear mip sampling, anisotropy, and per-material configuration.

### 5. PBR material + shader
**Status: extends the existing Phong path.**

`phong.frag.slang` is fixed Blinn-Phong with hardcoded specular and samples zero textures
(`shaders/passes/phong.frag.slang:40-46`). Add a metallic-roughness `pbr.frag` sampling
baseColor / metallic-roughness / normal / emissive / occlusion, with material factors in
the SSBO. Single directional light first; defer IBL.

### 6. glTF loader — integration work
**Status: wiring existing nodes, not new abstractions.**

- Add a parser (`fastgltf` preferred, or `tinygltf`).
- Walk primitives → `MeshNode`s; materials + textures → the material SSBO + texture
  `ImageRef`s (reuse the `phong_materials` SSBO-indexed-by-draw pattern).
- Generate tangents (mikktspace) when the asset doesn't provide them.

### 7. Scene graph / node hierarchy
**Status: a CPU transform node.**

glTF is a tree of TRS transforms. Flatten it to world matrices and feed them as the
per-draw model matrices `MultiDraw` already consumes. No new core concept.

### 8. Color management (sRGB / linear) — correctness, partly already wrong
**Status: the output half is currently incorrect.**

Lighting math is only correct on **linear** light values, but image files store color
gamma-encoded (sRGB). Workflow: decode sRGB→linear on read, light in linear, encode
linear→sRGB on write. Vulkan does both conversions in hardware *if the formats are right*.

- **Color textures** (baseColor, emissive) → `_SRGB` view (sampler auto-decodes).
- **Data textures** (normal, metallic-roughness, AO, height) → must stay `_UNORM`
  (linear). Decoding these as sRGB corrupts them — the most common glTF rendering bug.
- **Output is wrong today:** the swapchain is `VK_FORMAT_B8G8R8A8_UNORM`
  (`src/managers/SwapchainManager.cpp:68`), so there is **no automatic linear→sRGB encode
  on present**. Flat demos hide it; real linear-space lighting will look too dark. Fix:
  switch the desired surface format to `B8G8R8A8_SRGB` (free encode) or encode manually in
  the final write.
- **Intermediate targets** (already used via `BlitNode`, `OutlinePass`) should hold linear
  values, ideally HDR (`R16G16B16A16_SFLOAT`) so highlights aren't clamped; convert to
  sRGB only at the last write.
- **Tonemapping** (Reinhard / ACES) before the sRGB encode maps HDR back into [0,1].
  Optional first pass (clamp instead), but needed to "look right." Fits the existing
  `fullscreen.vert` + frag fullscreen-pass pattern.

---

## Recommended sequence

1. **`MeshRef` stride + draw assert** (item 2) — pure safety, tiny, de-risks everything
   below.
2. **Vertex attributes** (item 1) + **texture loader** (item 3) + **sampler** (item 4) +
   **sRGB-aware format choice** (item 8) — independent and high-leverage; yields textured,
   normal-mapped meshes (with correct output color) before any glTF parsing exists.
   Testable headless with a checkerboard texture.
3. **PBR shader** (item 5).
4. **glTF loader** (item 6) + **scene graph flatten** (item 7).
5. Follow-ons: IBL, KTX2 / compressed textures, tonemapping curve, skinning/animation.

## Out of scope (explicitly deferred)
IBL / environment lighting, compressed texture formats, skeletal animation, and a full
material editor. None of these change the architecture; they're later additions on the
same data-flow.
