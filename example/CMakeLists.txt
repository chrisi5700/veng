
# Examples that exercise the engine end-to-end (window + swapchain + sim thread). They
# share `veng_example_common` for non-engine boilerplate (Window, OrbitCamera, ...) so
# each example's main.cpp can focus on the graph wiring that's unique to it.

add_subdirectory(common)
add_subdirectory(instanced_cubes)
add_subdirectory(static_shapes)
add_subdirectory(phong_materials)
add_subdirectory(picking_outline)
add_subdirectory(pbr_materials)
add_subdirectory(gltf_viewer)
add_subdirectory(stl_viewer)
