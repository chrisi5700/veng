
# gltf_lod — loads a glTF model as a decimated LOD chain (veng_assets::load_gltf_lods) and selects
# the level at runtime from screen coverage. VENG_MODELS_DIR lets the no-argument run default to the
# DamagedHelmet sample fetched for the integration tests; any .glb/.gltf path can be passed instead.

target_add_executable(gltf_lod main.cpp)
target_compile_features(gltf_lod PRIVATE cxx_std_26)
target_link_libraries(gltf_lod PRIVATE veng_example_common veng_passes veng_assets glfw)
target_compile_definitions(gltf_lod PRIVATE VENG_MODELS_DIR="${CMAKE_SOURCE_DIR}/tests/assets/models")
set_target_properties(gltf_lod PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "")
