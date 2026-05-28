
# static_shapes — three lit primitives (cube / sphere / torus) drawn with one GraphicsNode
# hosting three add_draw() calls, no instancing. The only reactive surface is the orbit
# camera; OnDemand pacing idles the loop the moment dragging stops, which the cached% stat
# line makes obvious.

target_add_executable(static_shapes main.cpp)
target_compile_features(static_shapes PRIVATE cxx_std_26)
target_link_libraries(static_shapes PRIVATE veng_example_common)
set_target_properties(static_shapes PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "")
