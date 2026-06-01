
# stl_viewer — loads an STL mesh (no UVs/normals worth trusting), synthesises welded geometry +
# crease-smoothed normals + box UVs + tangents via veng_assets::load_stl, and renders it with a
# CC0 bare-steel PBR material through veng::passes::PbrPass. Pass an .stl path as argv[1]; with no
# argument it loads example/screw.stl. The steel texture set is fetched at configure (see
# steel.cmake) into a gitignored assets/ dir; if absent the example falls back to flat factor steel.

include(${CMAKE_CURRENT_SOURCE_DIR}/steel.cmake)

target_add_executable(stl_viewer main.cpp)
target_compile_features(stl_viewer PRIVATE cxx_std_26)
target_link_libraries(stl_viewer PRIVATE veng_example_common veng_passes veng_assets glfw)
target_compile_definitions(stl_viewer PRIVATE
    VENG_STL_MESH="${CMAKE_CURRENT_SOURCE_DIR}/resources/screw.stl"
    VENG_STL_ASSETS_DIR="${VENG_STL_ASSETS_DIR}")
set_target_properties(stl_viewer PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "")
