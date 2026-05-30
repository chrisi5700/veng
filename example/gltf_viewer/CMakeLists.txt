
# gltf_viewer — loads a glTF model via veng_assets and renders it with veng::passes::PbrPass.
# VENG_MODELS_DIR lets the no-argument run default to the DamagedHelmet sample fetched for the
# integration tests; any .glb/.gltf path can be passed on the command line instead.

target_add_executable(gltf_viewer main.cpp)
target_compile_features(gltf_viewer PRIVATE cxx_std_26)
target_link_libraries(gltf_viewer PRIVATE veng_example_common veng_passes veng_assets glfw)
target_compile_definitions(gltf_viewer PRIVATE VENG_MODELS_DIR="${CMAKE_SOURCE_DIR}/tests/assets/models")
set_target_properties(gltf_viewer PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "")
