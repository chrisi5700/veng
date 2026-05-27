include(CTest)
find_package(Catch2 3 REQUIRED)
include(Catch)

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

# Descriptor-set uniforms: a UniformNode feeds a GraphicsNode descriptor by reflected name,
# and a new value re-invalidates the upload + draw (a reactive uniform).
add_executable(UniformNodeTests nodes/UniformNodeTests.cpp)
target_link_libraries(UniformNodeTests PRIVATE veng_nodes veng_resources Catch2::Catch2WithMain)

# One pass sampling another's output: the sampled-image descriptor path + the demand cascade
# when a sampled source is rebound (the basis of the multi-pass outline effect).
add_executable(SampledImageTests nodes/SampledImageTests.cpp)
target_link_libraries(SampledImageTests PRIVATE veng_nodes veng_resources Catch2::Catch2WithMain)

# Pass/Draw split — one GraphicsNode hosts several draws (two meshes, per-draw model matrices)
# into one target; readback proves both draws landed with their own mesh + push constant.
add_executable(MultiDrawTests nodes/MultiDrawTests.cpp)
target_link_libraries(MultiDrawTests PRIVATE veng_nodes veng_resources Catch2::Catch2WithMain)

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
catch_discover_tests(UniformNodeTests)
catch_discover_tests(SampledImageTests)
catch_discover_tests(MultiDrawTests)

