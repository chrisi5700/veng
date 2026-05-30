# Fetch-at-configure of the Khronos glTF Sample Models the integration tests render
# (review.md item 6 acceptance set). The .glb files are gitignored (tests/assets/models/) and
# SHA256-pinned here. Robust offline: a failed download or hash mismatch removes the file and the
# corresponding test SKIPs (it checks the file exists at run time) — configure never fails on it.
#
# GLB (glTF-Binary) variants are used because each is a single self-contained file.

set(VENG_MODELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets/models")
set(_veng_models_base "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models")

option(VENG_FETCH_MODELS "Download Khronos glTF sample models for the integration tests" ON)

function(veng_fetch_model name path sha)
    set(out "${VENG_MODELS_DIR}/${name}.glb")
    if(EXISTS "${out}")
        file(SHA256 "${out}" have)
        if(have STREQUAL "${sha}")
            return() # already present and verified
        endif()
    endif()
    if(NOT VENG_FETCH_MODELS)
        return()
    endif()
    message(STATUS "veng: fetching glTF sample model '${name}'")
    file(DOWNLOAD "${_veng_models_base}/${path}" "${out}" STATUS st TIMEOUT 120)
    list(GET st 0 code)
    if(NOT code EQUAL 0)
        message(STATUS "veng:   '${name}' download failed (offline?) — its test will SKIP (${st})")
        file(REMOVE "${out}")
        return()
    endif()
    file(SHA256 "${out}" have)
    if(NOT have STREQUAL "${sha}")
        message(WARNING "veng:   '${name}' sha256 mismatch — removing; its test will SKIP")
        file(REMOVE "${out}")
    endif()
endfunction()

veng_fetch_model(Box "Box/glTF-Binary/Box.glb"
        "ed52f7192b8311d700ac0ce80644e3852cd01537e4d62241b9acba023da3d54e")
veng_fetch_model(BoxTextured "BoxTextured/glTF-Binary/BoxTextured.glb"
        "b510eca2e2ef33f62f9ed57d6e7ce2d10ebb2bdebc4a8e59d347719ba81abdf4")
veng_fetch_model(NormalTangentTest "NormalTangentTest/glTF-Binary/NormalTangentTest.glb"
        "5ac0932355ae1ea05a7485eeddcc3f1fbe56c678e00b519075feced80e9e9d6a")
veng_fetch_model(MetalRoughSpheres "MetalRoughSpheres/glTF-Binary/MetalRoughSpheres.glb"
        "450c05557b0823a10835e32c05caf77a7b175eba94516555c1c9bb084f563f01")
veng_fetch_model(DamagedHelmet "DamagedHelmet/glTF-Binary/DamagedHelmet.glb"
        "a1e3b04de97b11de564ce6e53b95f02954a297f0008183ac63a4f5974f6b32d8")
