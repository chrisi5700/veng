# veng — Code Review Feedback

> Critique companion to `progress.md`, produced by the `/loop` review pass against
> `design.md`. Item IDs (C/H/M/L#) are **stable across passes** so `progress.md` can
> cross-reference them; resolved items move to the changelog at the bottom.

---

## ⚠ MAINTAINER / ARCHITECT DIRECTIVE (2026-05-25, post pass 10) — SUPERSEDES THE PRIORITIES BELOW

An architect reviewed the **working tree** (not the stale committed snapshot the
automated passes grade). Two findings and a change of mode. Act on this before
anything in the pass-10 section.

### Finding A — core diverges from `design.md §2`, but this is RESOLVED BY DECISION — do **not** "fix" it

The L3 core implements the **revision model** (`Revision`, `Data::changed_at`,
`Node::verified_at`, recursive `visit` with explicit dirty propagation) — i.e.
`design.md`'s *earlier* draft. `design.md §2`/`§L3` were later rewritten to a
dirty-flag + forward-topological-sweep model (`m_dirty`/`m_changed`, `FrameIndex`,
typed `Data<T>*` edges). **The code does not match that text.**

**Decision: the implemented revision core is canonical and correct.** It independently
fixed (C1/M7) the exact async-staleness bug the rewrite was meant to prevent. **Do NOT
rewrite the core to match `design.md §2`** — the maintainer will reconcile the doc to
the code. For the L3 core, **the code is the source of truth.** (`design.md §L4`/`§L5`
— swapchain/present/driver — are *unbuilt* and remain the spec for the new work below.)

### Finding B — there is no rendering path. This is now the only priority.

The engine **cannot present a pixel.** No driver loop, no swapchain, no present node,
no L1 GPU resources, no GPU `ExecContext`. "Driving" today = manual
`graph.frame(sink, scheduler)` calls in the playground demo and tests. The reactive
mechanics (demand via sink, invalidation via `set`, caching/cutoff) are done and
tested; the harness that ties them to a swapchain/present cadence is the unbuilt
L4/L5, and it is what turns "a tested reactive compute graph" into "an engine that
draws."

### DIRECTIVE — change of mode: stop polishing, build the vertical slice

Ten iterations went to findings + L2/L3 polish + re-confirming already-fixed items.
**From now on: feature-driven forward progress, larger steps per iteration, toward a
rendered frame.** **Defer all open LOW/MEDIUM findings (M5, M6, M8, L1–L7)** and any
further L2 breadth unless a step below directly needs it. This is the roadmap now
(ordered; each item is one meaningful iteration, not a one-line fix):

1. **L1 resources** — `Buffer`/`Image` RAII over VMA; make `get_node`/`get_data`
   honor `Handle::generation`. Minimal transient/persistent split; aliasing stays
   deferred.
2. **GPU bridge (the L3→Vulkan seam)** — `GpuExecContext : graph::ExecContext`
   carrying command buffer + resource/sync managers + frame slot, and a `GpuNode`
   base that does the single localized `static_cast` to it (`design.md §L4`).
3. **L2 managers** — `SwapchainManager` (over vk-bootstrap; exposes `extent()` and
   acquire/present), `CommandManager`, `SyncManager` (one timeline per queue).
4. **L4 nodes for the slice** — `ScreenSize` source; one GPU raster node drawing a
   triangle into a **persistent scene-color target** sized from `ScreenSize`; a
   **swapchain source** invalidated on every `acquire()`; a **present/blit node**
   depending on **both** the scene-color target and the swapchain source.
5. **L5 driver loop** — `acquire()` → `set()` the swapchain source → `graph.frame(
   present_sink, scheduler)` → blit the cached scene into the acquired image →
   `vkQueuePresentKHR`. Frames-in-flight bookkeeping; conservative cadence (present
   every frame).
6. **Proof** — a window showing a triangle, plus an assertion that a *static* scene
   runs the present/blit every frame but re-runs the raster node **0 times** (caching),
   measured via node-execution counts / `to_dot`.

### Wiring rule (do not lose the thesis)

The scene subgraph must **not** depend on the swapchain source. Only the swapchain
source is dirtied each frame (on acquire), so only the present/blit re-runs; the scene
re-renders **only** when a scene source changes. A loop with no swapchain source would
produce **empty plans forever** — correct caching, but nothing ever presented. The
swapchain-source-as-clock is the one input deliberately dirty every frame.

### Process fixes

- **Commit the working tree on a branch at the end of every iteration.** All ten
  iterations are uncommitted, so each review pass grades a stale snapshot and
  re-flags fixed items — that lag is the main reason the loop feels slow. Commit so
  the review grades reality.
- The automated review's "remaining work is mostly NEW design scope" is right — build
  the slice above; do not hunt for more findings.

---

## Review context (2026-05-25, pass 11) — operating under the architect directive above

Acknowledged and adopted. This pass conforms to it:

- **Finding A (core vs `design.md §2`): not flagged.** Per the decision, the implemented
  revision core is canonical and the doc will be reconciled to it; for L3 the **code is
  the source of truth**. Earlier "design.md §2/§2.4" references in this file (C1/M7
  history, etc.) describe the *implemented* revision model and stand as historical
  record. I will not raise the doc/code divergence.
- **Mode change adopted: findings are deferred, not hunted.** M5, M6, M8 and L1–L7 are
  **DEFERRED** (parked below, not resolved) until the vertical slice needs them. I am
  not adding new LOW/MEDIUM polish.
- **Slice prerequisites verified present and clean.** Iteration 10 (which predates the
  directive) landed **`GraphicsPipelineBuilder`** (L2.1, dynamic rendering) + a shared
  `PipelineError.hpp`, and enabled the Vk1.3 features (`dynamicRendering`,
  `synchronization2`, `timelineSemaphore`) the slice's steps 2–5 require. I reviewed it
  for *real* bugs (not polish) and it is **clean**: complete partial-failure cleanup on
  every path (no leaks), correct dynamic-rendering `pNext` lifetime (the usual dangling
  bug — avoided), RAII move-only, reflection-driven, stage-compat checked. **Verified
  63/63 on-GPU.** So compute + graphics pipelines + descriptor allocation all exist —
  the slice (L1 → GPU bridge → managers → L4 nodes → L5 driver) is unblocked.
  - Note: **M6 is now live** in the graphics builder's vertex-input path (`to_vk_format`
    feeds `vertex_attributes`); still deferred, but it graduates from latent to
    real-on-exotic-formats. Builders also use a single set-0 layout (reflected `set`
    index ignored) — self-documented; fine until an L4 node uses a multi-set shader.
- **Going forward, my review targets the slice itself** — where the real hazards live:
  L1 `Buffer`/`Image` RAII + VMA lifetimes + `Handle::generation`; the GPU
  `ExecContext`/`GpuNode` `static_cast` seam; barrier/layout correctness and
  timeline/binary sync; frames-in-flight (M8's contract goes live here); and the
  **wiring rule** (scene subgraph must not depend on the swapchain source). I'll verify
  the "static scene re-runs raster 0×, present/blit every frame" proof (step 6) when it
  lands.
- **Process:** the directive's "commit per-iteration on a branch" is the right fix for
  the loop-lag. For the record, this review reads the **working tree** (it builds + runs
  it — 63/63 this pass); the lag is the two-loop timing race, not git state. Committing
  will still help by giving both loops a stable shared snapshot.

---

## Review context (2026-05-25, pass 10)

- **No CRITICAL or HIGH items remain.** Suite **58/58** (was 57), clean under the
  `llm-vcpkg` gate.
- Iteration 9 **fixed M7** (consumed pass-8 feedback). Verified: a new protected
  `Node::last_run_revision()` (returns `m_verified_at`) + `AccumulationNode` now resets
  on `changed_at() > last_run_revision()` — the same correct baseline as the C1 fix. My
  undemanded-reset repro, updated to the new pattern, now returns `count=1` (was 3); a
  new regression test covers it; existing-behaviour harness still all-pass (no
  regression). Good fix: the accessor is documented as the canonical baseline so future
  temporal/history nodes inherit it by construction.
- Unchanged since pass 9: `Graph.cpp` (logic), `ComputePipeline.cpp`, `Shader.cpp`,
  `Context.cpp`, `common.hpp`, `DescriptorAllocator` — so **M5/M6/M8** and the lows
  stand. (`Node.hpp` changed only to add the accessor; existing nodes unaffected.)
- No new design-scope breadth this iteration — iteration 9 spent the turn on M7.

---

## MEDIUM  *(all DEFERRED per the architect directive — parked until the slice needs them; none are resolved)*

### M8 — `DescriptorAllocator` has no threading or frames-in-flight contract  *(Deferred — goes live when the L5 driver wires it in)*

The allocator is correct single-threaded/single-frame, but two gaps block production
use in *this* engine (it's unused today, so no live bug — close before L4/L5 wire it in):

- **No threading contract.** `allocate()` mutates `m_pools` (`push_back`) and
  `m_current` with **no synchronization** (verified: zero `mutex`/`atomic`), and
  `vkAllocateDescriptorSets` requires *external* synchronization per pool. But node work
  runs on the multi-threaded `ThreadPoolScheduler`, so GPU/bridge nodes sharing one
  allocator would race (both the C++ members and the Vulkan pool). Either document
  "one allocator per recording thread / frame-slot" (the usual pattern) or guard it.
- **`reset()` ignores frames-in-flight.** Design §L2.2 sketches `reset_frame(frame_slot)`
  (§6); the implemented `reset()` resets *all* pools at once. Called between frames it
  would recycle descriptor sets an **in-flight frame's GPU is still reading** →
  use-after-free. Per-slot pools reset only when that slot's frame has retired.

Minor, same file: `create_pool()` collapses any failure to a null handle, so `allocate`
reports `eErrorOutOfPoolMemory` even when the real cause was e.g.
`eErrorOutOfDeviceMemory` (lost error fidelity).

### M6 — Reflection silently substitutes a wrong type on unhandled inputs  *(Re-verified — `Shader.cpp` unchanged)*

- `to_vk_descriptor_type` `default:` logs a warning and returns
  `vk::DescriptorType::eUniformBuffer`. This feeds `ComputePipelineBuilder`'s
  descriptor-set layout, so an unhandled binding type yields a layout that
  **mismatches the shader** → GPU validation error / wrong binding at dispatch, with
  only a log line.
- `to_vk_format` returns `eR32G32B32A32Sfloat` for any unknown scalar/vector
  (Float16/64, Int8/16/64) — a silently-wrong vertex format (bites once
  `GraphicsPipelineBuilder` lands).

Common types are handled, so normal shaders are unaffected — but it's the same
"return a plausible-but-wrong object instead of failing" pattern fixed in M4. The
builder already returns `std::expected`; surface an unhandled type as a typed error.

### M5 — Inconsistent error model; `merge_expected_t` is dead code  *(Re-verified — `common.hpp` reformatted only)*

`ComputePipeline` (`PipelineError`) and the graph (`GraphError`/`ExecError`) use typed
enums + `to_string`; **Shader** still returns bare `std::string` (the lone outlier);
Context uses `ResultVariant`. `merge_expected`/`merge_expected_t` in `common.hpp` — the
flattening tool design §9 references — still has **zero usages**, so it's untested and
its two partial specializations carry a latent ambiguity that only stays dormant
because it's never instantiated. A cross-layer decision (`progress.md` agrees):
standardise on typed errors (+ `merge_expected_t` for composition) or delete the dead
machinery. Not a bug — code works — but it's the kind of inconsistency that bites a
"clean in all layers" bar.

---

## LOW / INFO  *(all DEFERRED per the architect directive)*

- **L1 — `TransformNode::execute` does `dynamic_cast` per input per frame** (RTTI on
  the hot path). Cache the `ValueData<T>*` at wiring.
- **L2 — `Graph::visit` is recursive**; deep chains can stack-overflow. Explicit work
  stack for production.
- **L3 — Error types in the global namespace** — only `ContextErrors.hpp`
  (`InstanceCreationError`, … + `using ContextCreationError`) remains global vs the
  `veng`/lowercase-namespace rule (`ComputePipeline`/`PipelineError` already moved).
- **L4 — `debug_callback` resets the log pattern per message** (`set_pattern(fmt::format(...))`),
  not thread-safe across concurrent validation callbacks; pattern still says
  `"[IteratedFunction]"` (IFS copy-paste). Set once at init. *(Quick, satisfying fix —
  `progress.md` agrees.)*
- **L5 — `Handle::generation` is ignored by `get_node`/`get_data`** — fine for
  append-only v1; a UAF/ABA hole once eviction/removal (§6) lands.
- **L7 — Global slang session singleton is not concurrency-safe** (`get_session()`).
  Fine single-threaded; a race if compilation moves onto the job pool. Document/guard.
- **M2 (downgraded) — FAILED nodes re-execute every frame with no backoff.** Policy is
  explicit and §9-valid; only nit is a persistently-failing node re-running forever.
  Consider backoff / propagate-after-N later. Consumers correctly do not spin.
- **INFO — `execute()` is frame-synchronous** (intended; correct for CPU v1). Still
  owed for L4/L5: GPU/timeline async + frames-in-flight overlap (§2.7, §5, §6).

---

## Test coverage gaps

- **M6** — a shader with an unhandled binding type / exotic vertex format (assert the
  builder errors, not produce a mismatched layout/format).
- **M8** — a concurrent-allocation test would document/guard the threading contract
  (none today); a frames-in-flight reuse test would expose the `reset()` gap.
- **Multi-output stamping** — `changed_at` is stamped on *all* of a node's outputs when
  any changed; add a multi-output test before relying on per-output cutoff.
- A persistently-FAILED node over many frames (pins the M2 backoff question).
- Coverage is now measured (run_test.sh): 82% overall, **Graph.cpp 95%** (meets §10);
  Vulkan layers 68–80% (failure paths need device fault injection — acknowledged).

---

## Recently resolved (verified, dropped from active list)

- **M7 (MEDIUM, real bug)** — temporal reset used `== revision` (the C1 trap), missing a
  reset when the scene changed while the accumulator was undemanded → RESOLVED iter 9:
  new documented `Node::last_run_revision()`; reset is now `changed_at() >
  last_run_revision()`. **Verified:** repro on the new pattern returns `count=1` (was 3),
  +1 regression test, no regression in the existing harness, 58/58.
- **H2 (HIGH)** — `llm-vcpkg` preset + `fix_and_format.sh` / `run_test.sh` missing →
  RESOLVED iter 6. Preset inherits `dev`+`vcpkg`, warnings-as-errors + ASan/UBSan via
  global flags (so plain-`add_executable` tests are instrumented). **Verified:**
  configures clean; scripts are real; sanitized 51/51 + coverage per `progress.md`.
- **M4 (MEDIUM, real bug)** — Shader swallowed `vkCreateShaderModule` failure →
  RESOLVED iter 5 (`create_shader_module` returns `std::expected`, `create_shader`
  propagates). Verified pass 6 (source + build + 51/51).
- **M1 (MEDIUM)** — `set_app_name(title.data())` non-NUL-terminated → RESOLVED iter 5
  (`std::string` + `.c_str()`). Verified.
- **L6 (LOW)** — stage-only `Shader::operator<=>` → removed iter 5 (unused). Verified.
- **C1 / C2 (CRITICAL), H1 / H3 (HIGH), M3 (MEDIUM)** — resolved iters 3–4, verified
  passes 4–5 (cutoff predicate; try/catch; resolve stops mutating stamps; RAII
  `std::expected` pipeline builder; `ctest` 51/51). Re-confirmed green.

---

## Positives (so they are not re-flagged)

- **63/63 green (on-GPU and under ASan/UBSan); clean `-Werror` build; 95% L3-core
  coverage.** The implemented stack — L3 reactive core (+ temporal accumulation), L5 job
  pool (band dispatch, deadlock-free, TSan-clean), L2 `Compute`/`Graphics` pipeline
  builders + `DescriptorAllocator`, L0 `Context`/`Shader` (now with Vk1.3
  dynamic-rendering/sync2/timeline features) — has correct RAII + move semantics and
  `std::expected` propagation throughout. The `llm-vcpkg` gate actively catches bugs.
- `GraphicsPipelineBuilder` (iter 10) reviewed this pass: clean, no real bugs — correct
  partial-failure cleanup and dynamic-rendering `pNext` lifetime. The L2 builder layer
  is now complete enough to feed the L4 vertical slice.
- The temporal feature is now correct end-to-end: the `needs_refresh()` hook is minimal
  and regression-free, and the reset baseline (`last_run_revision()`) is documented as
  the canonical pattern so L4 history nodes inherit it.
- `DescriptorAllocator` is clean standalone code (the M8 gaps are about engine
  *integration*, not the allocator's own logic).
- The review/implementation loop has resolved every CRITICAL and HIGH item in severity
  order (C1/H1/M3 → C2/H3 → M4/M1/L6 → H2) and added regression + on-GPU tests.

---

## Remaining work is mostly NEW design scope, not findings

For the maintainer's planning — these are unbuilt parts of `design.md`, not defects:
- **General history / ping-pong nodes** (§L4) — reading *another* node's previous
  output (TAA history, prev-depth) needs double-buffered `Data` + a `temporal_inputs()`
  exclusion in `resolve`. (Iter 7 did the self-accumulation case; this is the next
  temporal step — and the right time to fix M7's reset predicate as the reusable
  node API.)
- **L2 breadth** — `GraphicsPipelineBuilder` (§L2.1) and `DescriptorWriter` (§L2.2,
  the companion to the now-landed `DescriptorAllocator`). (Note: M6 lands squarely on
  `GraphicsPipelineBuilder` once vertex formats matter; M8's frames-in-flight gap wants
  closing when the allocator meets the L5 driver.)
- **L1 resources** (typed buffer/image handles, transient/persistent pools, state
  tracker), **L4 node types**, **L5 frame driver** + present, **L6 tooling**.
