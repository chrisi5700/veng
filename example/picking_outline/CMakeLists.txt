
# picking_outline — composes PickingPass + OutlinePass: left-click shapes to toggle a glow
# outline. Links veng_passes (the pass library) on top of the shared example scaffolding.

target_add_executable(picking_outline main.cpp)
target_compile_features(picking_outline PRIVATE cxx_std_26)
target_link_libraries(picking_outline PRIVATE veng_example_common veng_passes glfw)
set_target_properties(picking_outline PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "")
