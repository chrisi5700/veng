# Fetch-at-configure of a CC0 bare-steel PBR texture set for the stl_viewer example. The texture
# files are gitignored (example/stl_viewer/assets/) and the zip is SHA256-pinned. Robust offline /
# without Python: a failed download, hash mismatch, or missing packer leaves the assets absent and
# the example falls back to a flat factor-only steel material (it checks the files at run time), so
# configure never fails on it.
#
# Source: ambientCG "Metal 032" (https://ambientcg.com/view?id=Metal032), CC0. ambientCG ships the
# metallic-roughness channels as separate greyscale PNGs, so we pack them into the single glTF-style
# metal_rough texture veng's shader expects (G=roughness, B=metallic) via pack_mr.py.

set(VENG_STL_ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")
set(_veng_steel_url "https://ambientcg.com/get?file=Metal032_1K-PNG.zip")
set(_veng_steel_sha "25acbc15cd3557801c7a45392796b5570e41799a80fb569db0301587ea91292f")

option(VENG_FETCH_STL_TEXTURES "Download the CC0 steel PBR texture set for the stl_viewer example" ON)

# The canonical files the example loads. If all three already exist we are done.
set(_veng_steel_base "${VENG_STL_ASSETS_DIR}/steel_basecolor.png")
set(_veng_steel_normal "${VENG_STL_ASSETS_DIR}/steel_normal.png")
set(_veng_steel_mr "${VENG_STL_ASSETS_DIR}/steel_mr.png")

if(EXISTS "${_veng_steel_base}" AND EXISTS "${_veng_steel_normal}" AND EXISTS "${_veng_steel_mr}")
    return()
endif()
if(NOT VENG_FETCH_STL_TEXTURES)
    return()
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(NOT Python3_Interpreter_FOUND)
    message(STATUS "veng: stl_viewer steel textures need Python3 to pack — skipping (example uses flat steel)")
    return()
endif()

set(_veng_steel_zip "${CMAKE_CURRENT_BINARY_DIR}/Metal032.zip")
set(_veng_steel_extract "${CMAKE_CURRENT_BINARY_DIR}/Metal032")

message(STATUS "veng: fetching CC0 steel texture set 'Metal032' for stl_viewer")
file(DOWNLOAD "${_veng_steel_url}" "${_veng_steel_zip}"
     EXPECTED_HASH SHA256=${_veng_steel_sha}
     STATUS _veng_steel_st TIMEOUT 180)
list(GET _veng_steel_st 0 _veng_steel_code)
if(NOT _veng_steel_code EQUAL 0)
    message(STATUS "veng:   steel set download failed (offline?) — example uses flat steel (${_veng_steel_st})")
    file(REMOVE "${_veng_steel_zip}")
    return()
endif()

file(MAKE_DIRECTORY "${_veng_steel_extract}" "${VENG_STL_ASSETS_DIR}")
file(ARCHIVE_EXTRACT INPUT "${_veng_steel_zip}" DESTINATION "${_veng_steel_extract}")

# Base colour (sRGB) and the GL-convention normal map copy straight across.
file(COPY_FILE "${_veng_steel_extract}/Metal032_1K-PNG_Color.png" "${_veng_steel_base}")
file(COPY_FILE "${_veng_steel_extract}/Metal032_1K-PNG_NormalGL.png" "${_veng_steel_normal}")

# Pack roughness + metalness into one metal_rough texture (G=roughness, B=metallic).
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/pack_mr.py"
            "${_veng_steel_extract}/Metal032_1K-PNG_Roughness.png"
            "${_veng_steel_extract}/Metal032_1K-PNG_Metalness.png"
            "${_veng_steel_mr}"
    RESULT_VARIABLE _veng_steel_pack)
if(NOT _veng_steel_pack EQUAL 0)
    message(STATUS "veng:   metal-rough packing failed (missing Pillow?) — example uses flat steel")
    file(REMOVE "${_veng_steel_base}" "${_veng_steel_normal}")
endif()
