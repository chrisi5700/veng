
target_add_executable(playground main.cpp Window.cpp)
target_compile_features(playground PRIVATE cxx_std_26)
target_link_libraries(playground PRIVATE
        veng_render_graph veng_nodes veng_managers veng_gpu veng_pipelines veng_shader veng_context)

# Use the SYSTEM Vulkan loader, not the vcpkg-bundled one. vcpkg's libvulkan is built
# without WSI (no VK_KHR_xcb/xlib/wayland_surface), so a windowed surface can't be
# created with it. libvulkan is the only shared library the playground pulls from vcpkg
# (everything else is static), so clearing the vcpkg runpath lets the dynamic linker
# resolve libvulkan.so.1 via ldconfig to the system loader (which has WSI). This removes
# the need for any LD_LIBRARY_PATH override at runtime.
set_target_properties(playground PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "")
