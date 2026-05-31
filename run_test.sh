#!/usr/bin/env bash
#
# Build and run the test suite with coverage, per CLAUDE.md. Produces lcov coverage
# information for the veng libraries (the L3 core is the priority).
#
# A separate gcov-instrumented build is used (not the sanitized llm-vcpkg preset):
# coverage and ASan are orthogonal, and a clean coverage build keeps counts accurate.
# For the safety gate (warnings-as-errors + ASan/UBSan) run instead:
#     cmake --build --preset llm-vcpkg && \
#       ASAN_OPTIONS=detect_leaks=0 ctest --preset llm-vcpkg
# (leak detection is disabled because the GPU driver/loader leak via libdbus at exit).
#
set -euo pipefail
cd "$(dirname "$0")"

: "${VCPKG_ROOT:?set VCPKG_ROOT to your vcpkg checkout}"

build_dir="build/coverage"

echo "==> Configuring coverage build ($build_dir)"
# VENG_TEST_VULKAN_ICD (optional): point the tests at a specific Vulkan ICD so the swapchain /
# present / FrameExecutor tests (which need a surface) run instead of SKIPping — set it to Mesa
# lavapipe's manifest to include that tier in coverage, e.g.
#   VENG_TEST_VULKAN_ICD=/usr/share/vulkan/icd.d/lvp_icd.json ./run_test.sh
# -fprofile-update=atomic so the multi-threaded SchedulerTests don't corrupt counters.
cmake -S . -B "$build_dir" -G Ninja \
	-DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
	-DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_STANDARD=26 \
	-DVENG_TEST_VULKAN_ICD="${VENG_TEST_VULKAN_ICD:-}" \
	-DCMAKE_CXX_FLAGS="--coverage -fprofile-update=atomic" \
	-DCMAKE_EXE_LINKER_FLAGS="--coverage"

echo "==> Building"
cmake --build "$build_dir"

echo "==> Running tests"
(cd "$build_dir" && ctest --output-on-failure)

if ! command -v lcov >/dev/null 2>&1; then
	echo "==> lcov not installed; skipping coverage report"
	exit 0
fi

# lcov 2.x is strict by default; tolerate the usual gcov/source mismatches.
ignore="--ignore-errors=mismatch,negative,unused,gcov,source,empty,inconsistent,range"

echo "==> Capturing coverage"
lcov --capture --directory "$build_dir" --output-file "$build_dir/coverage.info" $ignore
lcov --remove "$build_dir/coverage.info" \
	'*/vcpkg_installed/*' '/usr/*' '*/tests/*' '*/bench/*' \
	--output-file "$build_dir/coverage.filtered.info" $ignore
# Headline from --summary (global, de-duped). Avoid `lcov --list`: it double-counts
# header lines instrumented across TUs and reports impossible >100% per-file rates.
lcov --summary "$build_dir/coverage.filtered.info" $ignore
# genhtml takes a narrower --ignore-errors set than lcov (no 'gcov'); give it its own.
genhtml "$build_dir/coverage.filtered.info" --output-directory "$build_dir/html" \
	--ignore-errors source,unused,empty || true

echo "==> Coverage HTML: $build_dir/html/index.html"
