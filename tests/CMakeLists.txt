include(CTest)
find_package(Catch2 3 REQUIRED)
include(Catch)

# Fetch the Khronos glTF sample models the integration tests render (gitignored, SHA256-pinned).
include(${CMAKE_CURRENT_SOURCE_DIR}/assets/models.cmake)

function(add_test_executable TARGET_NAME SRC_FILES)
    target_add_executable(${TARGET_NAME} ${SRC_FILES})
    target_link_libraries(${TARGET_NAME} PRIVATE Catch2::Catch2WithMain)
    target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
    catch_discover_tests(${TARGET_NAME})
endfunction()

# VulkanContext tests
add_executable(ContextTests context/VulkanContextTests.cpp)
target_link_libraries(ContextTests PRIVATE veng_context Catch2::Catch2WithMain)


add_executable(ShaderLoadingTests shader/ShaderLoadingTests.cpp)
target_link_libraries(ShaderLoadingTests PRIVATE veng_shader Catch2::Catch2WithMain)
target_compile_options(ShaderLoadingTests PRIVATE -O0)

add_executable(ShaderValidationTests shader/ShaderValidationTests.cpp)
target_link_libraries(ShaderValidationTests PRIVATE veng_shader Catch2::Catch2WithMain)

# L3 Reactive Core tests — Vulkan-free, the priority for coverage (design.md §10).
add_executable(GraphTests rendergraph/GraphTests.cpp)
target_link_libraries(GraphTests PRIVATE veng_render_graph Catch2::Catch2WithMain)

# L5 job-system tests — multi-threaded scheduler driving the same core.
add_executable(SchedulerTests rendergraph/SchedulerTests.cpp)
target_link_libraries(SchedulerTests PRIVATE veng_render_graph Catch2::Catch2WithMain)

# L3 temporal / accumulation tests — demand-refresh + temporal self-edge (design §L4).
add_executable(TemporalTests rendergraph/TemporalTests.cpp)
target_link_libraries(TemporalTests PRIVATE veng_render_graph Catch2::Catch2WithMain)

# Clustered-shading culling math — Vulkan-free (pure glm): froxel build + light assignment, and the
# load-bearing build/analytic-mapping consistency round-trip.
add_executable(ClusterTests culling/ClusterTests.cpp)
target_link_libraries(ClusterTests PRIVATE veng_culling Catch2::Catch2WithMain)

# L2 ComputePipeline tests — reflection-driven builder on a headless device.
add_executable(ComputePipelineTests pipelines/ComputePipelineTests.cpp)
target_link_libraries(ComputePipelineTests PRIVATE veng_pipelines Catch2::Catch2WithMain)

# L2.1 GraphicsPipeline tests — dynamic-rendering builder on a headless device.
add_executable(GraphicsPipelineTests pipelines/GraphicsPipelineTests.cpp)
target_link_libraries(GraphicsPipelineTests PRIVATE veng_pipelines Catch2::Catch2WithMain)

# L2.2 DescriptorAllocator tests — pool growth + recycling on a headless device.
add_executable(DescriptorAllocatorTests descriptors/DescriptorAllocatorTests.cpp)
target_link_libraries(DescriptorAllocatorTests PRIVATE veng_descriptors Catch2::Catch2WithMain)

# L1 resource tests — Buffer/Image RAII over VMA on a headless device.
add_executable(ResourceTests resources/ResourceTests.cpp)
target_link_libraries(ResourceTests PRIVATE veng_resources Catch2::Catch2WithMain)

# L1.5 — the versioned, N-buffered ResourcePool's reuse + retention logic (N-buffering core).
add_executable(ResourcePoolTests resources/ResourcePoolTests.cpp)
target_link_libraries(ResourcePoolTests PRIVATE veng_resources Catch2::Catch2WithMain)

# GPU bridge tests — GpuExecContext/GpuNode seam, command recorded + executed on-GPU.
add_executable(GpuBridgeTests gpu/GpuBridgeTests.cpp)
target_link_libraries(GpuBridgeTests PRIVATE veng_gpu Catch2::Catch2WithMain)

# L2.4 manager tests — CommandManager + SyncManager on a headless device.
add_executable(ManagerTests managers/ManagerTests.cpp)
target_link_libraries(ManagerTests PRIVATE veng_managers veng_resources Catch2::Catch2WithMain)

# L4 slice test — RasterTriangleNode renders a triangle, verified by readback.
add_executable(RasterTriangleTests nodes/RasterTriangleTests.cpp)
target_link_libraries(RasterTriangleTests PRIVATE veng_nodes veng_resources Catch2::Catch2WithMain)

# L4/L5 slice proof — offscreen driver loop: static scene caches raster, present every frame.
add_executable(SliceDriverTests nodes/SliceDriverTests.cpp)
target_link_libraries(SliceDriverTests PRIVATE veng_nodes veng_resources Catch2::Catch2WithMain)

# L4 generic-graphics test — cube drawn by GraphicsNode from an MVP edge; readback guards
# the MVP convention and the node-level caching.
add_executable(GraphicsNodeTests nodes/GraphicsNodeTests.cpp)
target_link_libraries(GraphicsNodeTests PRIVATE veng_nodes veng_resources Catch2::Catch2WithMain)

# Real buffer-backed geometry: a MeshNode uploads vertex/index buffers an indexed
# GraphicsNode draws — proving "actual verts" and the upload-once caching.
add_executable(MeshNodeTests nodes/MeshNodeTests.cpp)
target_link_libraries(MeshNodeTests PRIVATE veng_nodes veng_resources Catch2::Catch2WithMain)

# MeshRef vertex-stride validation — a layout mismatch fails the frame, a match still renders.
add_executable(MeshStrideTests nodes/MeshStrideTests.cpp)
target_link_libraries(MeshStrideTests PRIVATE veng_nodes veng_resources Catch2::Catch2WithMain)

# Texture loader, mip generation, configurable sampler, sRGB decode — analytic readback.
add_executable(TextureTests assets/TextureTests.cpp)
target_link_libraries(TextureTests PRIVATE veng_assets veng_nodes veng_resources Catch2::Catch2WithMain)

# Metallic-roughness PBR pass — analytic shading invariants (base colour, emissive, N.L).
add_executable(PbrPassTests passes/PbrPassTests.cpp)
target_link_libraries(PbrPassTests PRIVATE veng_passes veng_assets veng_nodes veng_resources Catch2::Catch2WithMain)

# glTF loader integration — loads + renders the fetched Khronos sample models (SKIP if absent).
add_executable(GltfLoaderTests assets/GltfLoaderTests.cpp)
target_link_libraries(GltfLoaderTests PRIVATE veng_assets veng_passes veng_nodes veng_resources Catch2::Catch2WithMain)
target_compile_definitions(GltfLoaderTests PRIVATE VENG_MODELS_DIR="${VENG_MODELS_DIR}")

# Descriptor-set uniforms: a UniformNode feeds a GraphicsNode descriptor by reflected name,
# and a new value re-invalidates the upload + draw (a reactive uniform).
add_executable(UniformNodeTests nodes/UniformNodeTests.cpp)
target_link_libraries(UniformNodeTests PRIVATE veng_nodes veng_resources Catch2::Catch2WithMain)

# Array-of-T descriptor: a StorageBufferNode uploads a vector<T> into an SSBO and publishes a
# BufferRef carrying stride/count/name; a reactive re-upload re-invalidates downstream draws.
add_executable(StorageBufferNodeTests nodes/StorageBufferNodeTests.cpp)
target_link_libraries(StorageBufferNodeTests PRIVATE veng_nodes veng_resources Catch2::Catch2WithMain)

# Instanced draw end-to-end: a StorageBufferNode + a GraphicsNode with add_storage_buffer +
# set_instances_from render N triangles whose per-instance offsets came from the SSBO; the
# count of instances ALSO came from the SSBO's published ref. Readback proves N landed.
add_executable(InstancedDrawTests nodes/InstancedDrawTests.cpp)
target_link_libraries(InstancedDrawTests PRIVATE veng_nodes veng_resources Catch2::Catch2WithMain)

# Line-topology + dynamic vertex upload: a DynamicMeshNode feeds a GraphicsNode in
# eLineList topology — proves both the new topology() setter and reactive VB upload work.
add_executable(DebugLineTests nodes/DebugLineTests.cpp)
target_link_libraries(DebugLineTests PRIVATE veng_nodes veng_resources Catch2::Catch2WithMain)

# One pass sampling another's output: the sampled-image descriptor path + the demand cascade
# when a sampled source is rebound (the basis of the multi-pass outline effect).
add_executable(SampledImageTests nodes/SampledImageTests.cpp)
target_link_libraries(SampledImageTests PRIVATE veng_nodes veng_resources Catch2::Catch2WithMain)

# Pass/Draw split — one GraphicsNode hosts several draws (two meshes, per-draw model matrices)
# into one target; readback proves both draws landed with their own mesh + push constant.
add_executable(MultiDrawTests nodes/MultiDrawTests.cpp)
target_link_libraries(MultiDrawTests PRIVATE veng_nodes veng_resources Catch2::Catch2WithMain)

# N-buffering integration — two frames in flight render into distinct pooled target copies.
add_executable(NBufferingTests nodes/NBufferingTests.cpp)
target_link_libraries(NBufferingTests PRIVATE veng_nodes veng_resources Catch2::Catch2WithMain)

# ScreenshotNode + on_retired — proves peer-sink + post-fence readback work without PresentNode.
add_executable(ScreenshotNodeTests nodes/ScreenshotNodeTests.cpp)
target_link_libraries(ScreenshotNodeTests PRIVATE veng_nodes veng_resources Catch2::Catch2WithMain)

# PhongPass (L5) — locks the face-winding fix (eClockwise under the camera's Y-flip) and the
# back-then-front transparency ordering: front renders, back is culled, translucent blends over opaque.
add_executable(PhongPassTests nodes/PhongPassTests.cpp)
target_link_libraries(PhongPassTests PRIVATE veng_passes veng_nodes veng_resources Catch2::Catch2WithMain)

catch_discover_tests(ContextTests)
catch_discover_tests(ShaderLoadingTests)
catch_discover_tests(ShaderValidationTests)
catch_discover_tests(GraphTests)
catch_discover_tests(SchedulerTests)
catch_discover_tests(TemporalTests)
catch_discover_tests(ComputePipelineTests)
catch_discover_tests(GraphicsPipelineTests)
catch_discover_tests(DescriptorAllocatorTests)
catch_discover_tests(ResourceTests)
catch_discover_tests(ResourcePoolTests)
catch_discover_tests(GpuBridgeTests)
catch_discover_tests(ManagerTests)
catch_discover_tests(RasterTriangleTests)
catch_discover_tests(SliceDriverTests)
catch_discover_tests(GraphicsNodeTests)
catch_discover_tests(MeshNodeTests)
catch_discover_tests(MeshStrideTests)
catch_discover_tests(TextureTests)
catch_discover_tests(PbrPassTests)
catch_discover_tests(GltfLoaderTests)
catch_discover_tests(UniformNodeTests)
catch_discover_tests(StorageBufferNodeTests)
catch_discover_tests(InstancedDrawTests)
catch_discover_tests(DebugLineTests)
catch_discover_tests(SampledImageTests)
catch_discover_tests(MultiDrawTests)
catch_discover_tests(NBufferingTests)
catch_discover_tests(ScreenshotNodeTests)
catch_discover_tests(PhongPassTests)

