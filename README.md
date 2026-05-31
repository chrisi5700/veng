# veng

[![CI](https://github.com/chrisi5700/veng/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/chrisi5700/veng/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![C++](https://img.shields.io/badge/C%2B%2B-23%2F26-blue.svg)
![Vulkan](https://img.shields.io/badge/Vulkan-1.3-red.svg)

**veng** is an experimental Vulkan rendering engine built around a *reactive render
graph*: a persistent, demand-driven dependency graph that only recomputes the work an
output actually depends on. A still camera over a static scene costs almost nothing; a
changed uniform re-runs exactly the nodes downstream of it — no per-frame rebuild, no
manual dirty-tracking.

> ⚠️ **Status: early / experimental.** The architecture and public API are still moving.
> It targets Linux + Vulkan 1.3 and is a personal engine, not a production library.

## Why a reactive graph?

Traditional frame graphs rebuild and re-record every frame. veng instead models rendering
as incremental computation over a bipartite DAG of **nodes** (compute) and **values**
(buffers, images, uniforms, meshes):

- **Pull-based** — you *demand* a sink (e.g. the swapchain image); only its transitive
  inputs are resolved.
- **Memoized** — a node whose inputs did not change is skipped (change-cutoff), so cached
  results flow straight through.
- **Temporal edges** — a first-class one-frame-delayed *history* edge expresses
  ping-pong / TAA / progressive accumulation without hand-managed double buffers.
- **Scheduler-injected** — the core is Vulkan-agnostic and runs through an injected
  `Scheduler` (inline for tests, a thread pool for the real loop).

### The core in a nutshell

```cpp
#include <veng/rendergraph/Graph.hpp>
using namespace veng::graph;

Graph g;
auto width  = g.add_source<int>(1280);
auto height = g.add_source<int>(720);
auto aspect = g.add_transform(
    [](const int& w, const int& h) { return static_cast<float>(w) / static_cast<float>(h); },
    width, height);

InlineScheduler sched;                  // runs tasks inline; swap for ThreadPoolScheduler
auto plan = g.frame(aspect, sched);     // resolve + execute the demanded sub-graph

g.set(width, 1920);                     // queued, applied at the next frame boundary
plan = g.frame(aspect, sched);          // only `aspect` re-runs; unchanged sources are cut
```

GPU nodes (`MeshNode`, `UniformNode`, `GraphicsNode`, …) are the same idea with Vulkan
resources flowing on the edges. See [`example/`](example/) for the rendering API in use.

## Features

- Clustered-forward **PBR** (metallic-roughness) with HDR + tonemapping
- **glTF 2.0** scene loading via [fastgltf](https://github.com/spnda/fastgltf) (static
  meshes, materials, node hierarchy, textures)
- Phong materials, transparency ordering, hardware **instancing**
- Screen-space **outline** post-process and GPU **object picking** passes
- Versioned, **N-buffered** transient resource pool (frames-in-flight safe)
- Reflection-driven pipeline + descriptor builders (shaders authored in **Slang**)
- Multi-threaded, height-batched graph scheduler

## Examples

Each demo under [`example/`](example/) builds to its own executable:

| Example | Shows |
|---|---|
| `static_shapes` | minimal lit scene; the reactive idle (cached% → ~100% when still) |
| `instanced_cubes` | 125 instanced cubes under an orbit camera |
| `phong_materials` | Phong shading + transparency |
| `pbr_materials` | metallic-roughness PBR spheres |
| `gltf_viewer` | load + render a glTF model (e.g. DamagedHelmet) |
| `picking_outline` | GPU picking + screen-space outline |

## Getting started

### Prerequisites

- **Linux** with a **Vulkan 1.3** driver (a discrete/integrated GPU, or
  [lavapipe](https://docs.mesa3d.org/drivers/llvmpipe.html) for software rendering)
- A C++23/26 compiler — **GCC 14+** or **Clang 19+** (developed against GCC 16). Clang
  18 is too old: its preliminary C++26 mode leaves libstdc++'s `std::expected` gated off.
- **CMake ≥ 3.28** (3.30+ recommended for C++26) and **Ninja**
- **[vcpkg](https://github.com/microsoft/vcpkg)** — set `VCPKG_ROOT` to your checkout
- **[Slang](https://github.com/shader-slang/slang)** — installed system-wide or via the
  Vulkan SDK (provides `slang::slang` / `slangConfig.cmake`)

> All other dependencies (GLFW, glm, spdlog, Dear ImGui, vk-bootstrap, VMA-Hpp, stb,
> fastgltf, Catch2, Google Benchmark) are resolved by vcpkg from [`vcpkg.json`](vcpkg.json).

### Build

```bash
export VCPKG_ROOT=/path/to/vcpkg
cmake --preset dev-vcpkg          # Debug; use release-vcpkg for optimized builds
cmake --build --preset dev-vcpkg
```

Run an example (built into `build/dev-vcpkg/example/<name>/`):

```bash
./build/dev-vcpkg/example/gltf_viewer/gltf_viewer
```

> **Windowed examples & WSI:** if you built the Vulkan loader through vcpkg it may lack
> the window-system (WSI) surface extensions GLFW needs. Prefer your **system** Vulkan
> loader (`libvulkan`) when running the windowed examples.

### Available CMake presets

| Preset | Purpose |
|---|---|
| `dev-vcpkg` | Debug, warnings on, no sanitizers |
| `release-vcpkg` | Optimized + LTO |
| `llm-vcpkg` | Debug + warnings-as-errors + Address/UB sanitizers (the safety gate) |
| `ci-coverage` | Debug + gcov coverage instrumentation (used by CI) |

## Testing

Tests use **Catch2 v3** and register with CTest. They exercise the reactive core on the
CPU and the GPU layers end-to-end (rendering + pixel readback) on a real or software
Vulkan device.

```bash
# Coverage build + run + lcov HTML report (writes build/coverage/html/index.html)
./run_test.sh

# The safety gate: warnings-as-errors + ASan/UBSan
cmake --build --preset llm-vcpkg
ASAN_OPTIONS=detect_leaks=0 ctest --preset llm-vcpkg
```

`./fix_and_format.sh` applies clang-format and runs clang-tidy + cppcheck (advisory).

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the suite on **lavapipe**
(software Vulkan — the runners have no GPU) across four lanes:

| Lane | Compiler | Build | What it guards |
|---|---|---|---|
| **coverage** | GCC 14 | `ci-coverage` | full suite + lcov report (artifact + run summary) |
| **asan+ubsan** | GCC 14 | `llm-vcpkg` | the safety gate: warnings-as-errors + Address/UB sanitizers |
| **clang** | Clang 19 | `dev-vcpkg` (`-Werror`) | the suite compiles + passes under a second compiler |
| **tsan** | GCC 14 | `tsan-vcpkg` | ThreadSanitizer over the Vulkan-free reactive core (the threaded scheduler) |

TSan is scoped to the CPU-only core tests (`SchedulerTests`, `GraphTests`,
`TemporalTests`) — running it over the GPU suite would only surface noise from the
uninstrumented Mesa driver. vcpkg dependencies are pinned and binary-cached per compiler
for fast reruns.

## Documentation

The API reference is generated from the in-source Doxygen comments using the clean,
dark-mode [doxygen-awesome-css](https://github.com/jothepro/doxygen-awesome-css) theme.
When `doxygen` is installed it is regenerated into `docs/html/` (gitignored) at CMake
configure time — open `docs/html/index.html`. Regenerate on demand with:

```bash
cmake --build --preset dev-vcpkg --target docs
```

Pass `-DVENG_BUILD_DOCS=OFF` to skip documentation generation.

## Project layout

```
include/veng/   Public headers (mirrors src/ by subsystem)
src/            Implementation, layered:
  context/        L0  Vulkan device/context (vk-bootstrap, VMA)
  resources/      L1  RAII Buffer/Image + versioned N-buffered ResourcePool
  shader/         L2  Slang runtime compilation + reflection
  pipelines/      L2  reflection-driven compute/graphics pipeline builders
  descriptors/    L2  growable descriptor allocator
  managers/       L2  command pools, swapchain, per-frame executor
  rendergraph/    L3  the reactive core (Vulkan-agnostic) + thread-pool scheduler
  nodes/          L4  concrete GPU graph nodes (mesh, uniform, graphics, blit, present…)
  culling/        clustered-shading froxel/light-cull math (pure CPU)
  passes/         L5  reusable effects (Phong, PBR, Outline, Picking, ClusteredLights)
  assets/         texture + glTF loaders
tests/          Catch2 unit + integration tests (mirrors src/)
example/        Runnable demos
shaders/        Slang shader sources (compiled at runtime)
cmake/          Target helpers + compiler settings
```

## Contributing

This is a personal project, but issues and PRs are welcome. Please run
`./fix_and_format.sh` and make sure `ctest` passes (the `llm-vcpkg` gate is the bar)
before opening a PR. See [`CLAUDE.md`](CLAUDE.md) for the coding conventions.

## License

[MIT](LICENSE) © 2026 Chris (chrisi5700)

## Acknowledgements

The reactive-graph design draws on incremental/self-adjusting computation (Adapton,
Salsa) and the Frostbite frame-graph talk (GDC 2017), with shaders authored in
[Slang](https://github.com/shader-slang/slang).
