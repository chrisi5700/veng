# Consuming veng

veng is a Vulkan reactive render-graph engine. This is how a downstream project links and uses it.
For runnable code, see `example/` (built when veng is the top-level project) and the geng project,
which consumes veng as a submodule.

## Build integration

veng exposes one umbrella CMake target, **`veng::veng`**, that carries every public library plus all
usage requirements (the Vulkan/glm/… links *and* the required compile definitions such as
`GLM_FORCE_DEPTH_ZERO_TO_ONE` and the `VULKAN_HPP_*` flags — you never re-list them). Per-library
targets (`veng::veng_context`, `veng::veng_nodes`, …) are available if you want to link narrowly.

### As a submodule (add_subdirectory)

```cmake
add_subdirectory(libs/veng)          # veng sees PROJECT_IS_TOP_LEVEL=OFF -> libraries only
                                     # (its examples/tests/install rules stay off)
target_link_libraries(my_app PRIVATE veng::veng)
```

### As an installed package (find_package)

```sh
cmake --preset llm-vcpkg
cmake --build --preset llm-vcpkg -j 4
cmake --install build/llm-vcpkg --prefix /your/prefix
```

```cmake
find_package(veng CONFIG REQUIRED)   # re-finds veng's dependencies via find_dependency
target_link_libraries(my_app PRIVATE veng::veng)
```

`vengConfig.cmake` also sets `veng_SHADER_DIR` to the installed shader directory (see *Shaders* below).

## Includes

Pull in the public API with the umbrella header, or include the narrower per-subsystem headers to
keep a translation unit's include cost down:

```cpp
#include <veng/veng.hpp>                 // everything
// or, e.g.:
#include <veng/context/Context.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/managers/HeadlessExecutor.hpp>
```

## Two ways to drive it

**Headless / offscreen** — veng creates its own device; you build a graph and run it, e.g. reading the
result back through a `ScreenshotNode`:

```cpp
auto ctx = veng::Context::create("my_app").value();          // no surface == headless
veng::ResourcePool   pool(ctx.device(), ctx.rhi(), ctx.allocator(), /*frames_in_flight*/ 2);
veng::CommandManager commands(ctx);
veng::graph::InlineScheduler scheduler;
veng::HeadlessExecutor       executor(ctx, pool, commands, scheduler);

veng::graph::Graph graph;
// ... add_source / add nodes / set_producer ... then:
executor.run_once(graph, sinks);
```

**Embedded** — a host (e.g. Qt's `QVulkanWindow`) owns the Vulkan device and swapchain; veng adopts
them and renders into an offscreen image you copy into the image you present:

```cpp
auto ctx = veng::Context::adopt(instance, physical_device, device, graphics_queue, graphics_family).value();
// ... build + execute the graph into an offscreen ImageRef `scene`, then, on your command buffer:
veng::gpu::blit_into_host_image(gpu_ctx, scene, host_image, host_extent);   // does the barriers + blit
```

The host keeps ownership of acquisition, submission, and synchronization; veng only records the work.

## Shaders

veng compiles its shipped shaders (passes, etc.) from `SHADER_DIR` at runtime. For a build-tree or
submodule consumer this is veng's source `shaders/`. For an **installed** package, set the
`VENG_SHADER_DIR` environment variable to the installed location (`vengConfig.cmake` exposes it as the
`veng_SHADER_DIR` CMake variable), or pass your own directories to `Context::create` /
`Context::adopt` — those are always searched after veng's own.

## Requirements

A C++23 compiler and the dependencies veng finds via vcpkg (Vulkan, glfw, glm, VulkanMemoryAllocator,
slang, spdlog, fastgltf, meshoptimizer, …). The `llm-vcpkg` preset wires the vcpkg toolchain; a
consumer using its own vcpkg toolchain gets veng's transitive `find_dependency` calls for free.
