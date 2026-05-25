#!/usr/bin/env bash
#
# Auto-format the source tree and run static analysis, per CLAUDE.md.
#
# Formatting (clang-format) is applied in place. clang-tidy and cppcheck are run as
# advisory reports — the enforcing gate is the `llm-vcpkg` preset, which builds the
# library/app targets with warnings-as-errors and ASan/UBSan (see run_test.sh).
#
set -uo pipefail
cd "$(dirname "$0")"

echo "==> clang-format (in place)"
find include src tests playground bench -type f \( -name '*.hpp' -o -name '*.cpp' \) -print0 |
	xargs -0 -r clang-format -i
echo "    formatted."

# clang-tidy needs a compile database; reuse whichever build dir already has one.
build_dir=""
for candidate in build/dev-vcpkg build/llm-vcpkg build/release-vcpkg; do
	if [ -f "$candidate/compile_commands.json" ]; then
		build_dir="$candidate"
		break
	fi
done

if command -v clang-tidy >/dev/null 2>&1 && [ -n "$build_dir" ]; then
	echo "==> clang-tidy (advisory, using $build_dir)"
	# Scope to our translation units; checks kept focused to avoid third-party noise.
	find src -name '*.cpp' -print0 |
		xargs -0 -r clang-tidy -p "$build_dir" --quiet \
			-checks='bugprone-*,performance-*,modernize-*,-modernize-use-trailing-return-type,clang-analyzer-*' ||
		true
else
	echo "==> clang-tidy skipped (need clang-tidy + a configured build dir with compile_commands.json)"
fi

if command -v cppcheck >/dev/null 2>&1; then
	echo "==> cppcheck (advisory)"
	cppcheck --enable=warning,performance,portability --std=c++23 --language=c++ \
		--quiet --inline-suppr -I include src || true
else
	echo "==> cppcheck skipped (not installed)"
fi

echo "==> done."
