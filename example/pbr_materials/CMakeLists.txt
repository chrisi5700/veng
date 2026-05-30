
# pbr_materials — a metallic×roughness grid of spheres rendered through veng::passes::PbrPass.
# Links the pass library + the asset library (for the 1x1 default textures) on top of the shared
# example scaffolding.

target_add_executable(pbr_materials main.cpp)
target_compile_features(pbr_materials PRIVATE cxx_std_26)
target_link_libraries(pbr_materials PRIVATE veng_example_common veng_passes veng_assets glfw)
set_target_properties(pbr_materials PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "")
