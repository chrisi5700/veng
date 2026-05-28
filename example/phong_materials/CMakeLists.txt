
# phong_materials — five spheres with per-instance Phong materials (per-object ambient /
# diffuse / specular + Ka/Kd/Ks + shininess + mode), demonstrating the UniformNode pattern
# for scene-wide state (light, view) alongside a StorageBufferNode for the per-object array.

target_add_executable(phong_materials main.cpp)
target_compile_features(phong_materials PRIVATE cxx_std_26)
target_link_libraries(phong_materials PRIVATE veng_example_common)
set_target_properties(phong_materials PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "")
