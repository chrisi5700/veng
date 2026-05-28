
# veng_example_common — shared infrastructure for example apps:
#   * Window         : GLFW window + Vulkan surface
#   * OrbitCamera    : owns a `view_proj` graph source, drag/scroll-driven
#   * Geometry       : Vertex + Mesh + factory functions (cube, sphere, torus, ...)
#   * AppLoop        : Vulkan context + swapchain + graph + frame-closer + main loop
#
# An example app links this and writes only the scene-specific reactive nodes — see
# example/instanced_cubes/main.cpp for the pattern.

add_library(veng_example_common STATIC
        Window.cpp
        Geometry.cpp
        AppLoop.cpp)
target_include_directories(veng_example_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(veng_example_common PUBLIC cxx_std_26)
target_link_libraries(veng_example_common PUBLIC
        Vulkan::Vulkan glfw glm::glm
        veng_render_graph veng_nodes veng_managers veng_gpu veng_pipelines veng_shader veng_context veng_resources)
