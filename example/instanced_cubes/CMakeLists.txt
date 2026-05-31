
# instanced_cubes — the rigid-body viewer slice. 125 instanced
# cubes orbiting the origin, driven by a CPU "physics" thread, lit by a directional light,
# viewed through an interactive orbit camera. See main.cpp for the reactive-graph wiring;
# the window/context/swapchain/camera/loop boilerplate all lives in veng_example_common.

target_add_executable(instanced_cubes main.cpp)
target_compile_features(instanced_cubes PRIVATE cxx_std_26)
target_link_libraries(instanced_cubes PRIVATE veng_example_common)

# Use the SYSTEM Vulkan loader, not the vcpkg-bundled one. vcpkg's libvulkan is built
# without WSI (no VK_KHR_xcb/xlib/wayland_surface), so a windowed surface can't be
# created with it. libvulkan is the only shared library this example pulls from vcpkg
# (everything else is static), so clearing the vcpkg runpath lets the dynamic linker
# resolve libvulkan.so.1 via ldconfig to the system loader (which has WSI). This removes
# the need for any LD_LIBRARY_PATH override at runtime.
set_target_properties(instanced_cubes PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "")
