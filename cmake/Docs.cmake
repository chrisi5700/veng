# Docs.cmake — optional Doxygen API documentation with the doxygen-awesome-css theme.
#
# When VENG_BUILD_DOCS is ON and Doxygen is installed, the HTML docs are (re)generated at
# configure time into docs/html (gitignored) using a clean, dark-mode-capable theme. A
# `docs` target is also provided to regenerate on demand. If Doxygen is absent (e.g. on CI),
# this module quietly does nothing, so it never blocks a build.

option(VENG_BUILD_DOCS "Generate Doxygen API documentation at configure time (requires doxygen)" ON)

if(NOT VENG_BUILD_DOCS)
    return()
endif()

find_package(Doxygen OPTIONAL_COMPONENTS dot)
if(NOT DOXYGEN_FOUND)
    message(STATUS "veng: doxygen not found — skipping API docs (set VENG_BUILD_DOCS=OFF to silence)")
    return()
endif()

# Fetch the theme (CSS/JS only). SOURCE_SUBDIR points at a directory that does not exist so
# FetchContent_MakeAvailable populates the files without add_subdirectory'ing the theme's own
# CMake project (which we do not want to build).
include(FetchContent)
FetchContent_Declare(
    doxygen-awesome-css
    GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
    GIT_TAG        v2.4.2
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  __veng_theme_no_cmake__
)
FetchContent_MakeAvailable(doxygen-awesome-css)
FetchContent_GetProperties(doxygen-awesome-css SOURCE_DIR VENG_AWESOME_DIR)

# --- Doxyfile.in substitution variables -------------------------------------------------
set(VENG_DOXY_VERSION    "0.1.0")
set(VENG_DOXY_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/docs")
set(VENG_DOXY_STRIP_PATH "${CMAKE_SOURCE_DIR}")
set(VENG_DOXY_MAINPAGE   "${CMAKE_SOURCE_DIR}/README.md")
set(VENG_DOXY_INPUT      "\"${CMAKE_SOURCE_DIR}/include\" \"${CMAKE_SOURCE_DIR}/src\" \"${CMAKE_SOURCE_DIR}/README.md\" \"${CMAKE_SOURCE_DIR}/CLAUDE.md\"")

if(DOXYGEN_DOT_FOUND)
    set(VENG_DOXY_HAVE_DOT "YES")
else()
    set(VENG_DOXY_HAVE_DOT "NO")
endif()

set(VENG_DOXY_EXTRA_STYLESHEETS "\"${VENG_AWESOME_DIR}/doxygen-awesome.css\" \"${VENG_AWESOME_DIR}/doxygen-awesome-sidebar-only.css\" \"${VENG_AWESOME_DIR}/doxygen-awesome-sidebar-only-darkmode-toggle.css\"")
set(VENG_DOXY_EXTRA_FILES       "\"${VENG_AWESOME_DIR}/doxygen-awesome-darkmode-toggle.js\" \"${VENG_AWESOME_DIR}/doxygen-awesome-fragment-copy-button.js\" \"${VENG_AWESOME_DIR}/doxygen-awesome-paragraph-link.js\" \"${VENG_AWESOME_DIR}/doxygen-awesome-interactive-toc.js\"")

set(_docs_bin "${CMAKE_BINARY_DIR}/docs")
file(MAKE_DIRECTORY "${_docs_bin}")
set(VENG_DOXY_HEADER "${_docs_bin}/header.html")

set(_doxyfile "${_docs_bin}/Doxyfile")
configure_file("${CMAKE_SOURCE_DIR}/docs/Doxyfile.in" "${_doxyfile}" @ONLY)

# Generate Doxygen's version-correct stock header, then inject the doxygen-awesome scripts so
# the dark-mode toggle, fragment copy button, paragraph links and interactive ToC are wired
# up. Done once (the header is stable); delete build/docs/header.html to regenerate.
if(NOT EXISTS "${VENG_DOXY_HEADER}")
    execute_process(
        COMMAND "${DOXYGEN_EXECUTABLE}" -w html "${VENG_DOXY_HEADER}"
                "${_docs_bin}/footer.html" "${_docs_bin}/doxygen-extra.css" "${_doxyfile}"
        WORKING_DIRECTORY "${_docs_bin}"
        RESULT_VARIABLE _hdr_rc)
    if(_hdr_rc EQUAL 0 AND EXISTS "${VENG_DOXY_HEADER}")
        file(READ "${VENG_DOXY_HEADER}" _hdr)
        string(REPLACE "</head>"
"<script type=\"text/javascript\" src=\"$relpath^doxygen-awesome-darkmode-toggle.js\"></script>
<script type=\"text/javascript\" src=\"$relpath^doxygen-awesome-fragment-copy-button.js\"></script>
<script type=\"text/javascript\" src=\"$relpath^doxygen-awesome-paragraph-link.js\"></script>
<script type=\"text/javascript\" src=\"$relpath^doxygen-awesome-interactive-toc.js\"></script>
<script type=\"text/javascript\">
  DoxygenAwesomeDarkModeToggle.init();
  DoxygenAwesomeFragmentCopyButton.init();
  DoxygenAwesomeParagraphLink.init();
  DoxygenAwesomeInteractiveToc.init();
</script>
</head>" _hdr "${_hdr}")
        file(WRITE "${VENG_DOXY_HEADER}" "${_hdr}")
    else()
        message(WARNING "veng: could not generate the Doxygen HTML header (rc=${_hdr_rc}); docs will use the default theme")
        set(VENG_DOXY_HEADER "")
        configure_file("${CMAKE_SOURCE_DIR}/docs/Doxyfile.in" "${_doxyfile}" @ONLY)
    endif()
endif()

# On-demand regeneration.
add_custom_target(docs
    COMMAND "${DOXYGEN_EXECUTABLE}" "${_doxyfile}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "veng: regenerating API docs -> docs/html/index.html"
    VERBATIM)

# Regenerate at configure time (best-effort: a docs failure never breaks configure).
execute_process(
    COMMAND "${DOXYGEN_EXECUTABLE}" "${_doxyfile}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE _doc_rc
    OUTPUT_QUIET)
if(_doc_rc EQUAL 0)
    message(STATUS "veng: API docs generated -> ${CMAKE_SOURCE_DIR}/docs/html/index.html")
else()
    message(WARNING "veng: doxygen exited ${_doc_rc}; docs may be incomplete")
endif()
