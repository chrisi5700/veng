# rhi_triangle — a node-free, graph-free example that presents one triangle straight through the RHI
# (rhi::Device + rhi::Swapchain + rhi::CommandEncoder), for reviewing the RHI surface itself. Unlike
# the other examples it does NOT link veng_example_common (AppLoop + the node/manager/graph layers) —
# it stays a pure-RHI demonstration. It links only veng_pipelines (GraphicsPipeline + Shader), which
# transitively brings veng_context (rhi::Device/Swapchain + the RHI headers), plus glfw for the
# window. Window.cpp is compiled in directly (reusing its GLFW + surface glue, including the XWayland
# extension fallback) instead of being duplicated here.

target_add_executable(rhi_triangle
        main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/Window.cpp)
target_compile_features(rhi_triangle PRIVATE cxx_std_26)
target_include_directories(rhi_triangle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(rhi_triangle PRIVATE veng_pipelines glfw)
# Clear the runpath so libvulkan.so.1 resolves to the SYSTEM loader (which has WSI surface support);
# vcpkg's bundled loader is built without it, so a windowed surface can't be created against it.
set_target_properties(rhi_triangle PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "")
