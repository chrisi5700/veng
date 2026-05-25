# veng — Implementation Progress

> Living log for the `/loop` implementation of `design.md`. Each iteration appends a
> dated section describing what landed and where to continue.

---

## Iteration 12 — 2026-05-25 — Step 2: the GPU bridge (L3→Vulkan seam)

Vertical-slice roadmap step 2 (the directive). The L3 core stays Vulkan-agnostic; this
adds the one seam where a frame plan's GPU work meets Vulkan.

### What landed

- **`veng::gpu::GpuExecContext : graph::ExecContext`** — resolves data handles and the
  frame revision against the graph (like the CPU context) **and** exposes the recording
  `vk::CommandBuffer`, the `Context` (device/allocator/queues), and the
  frame-in-flight slot.
- **`veng::gpu::GpuNode : graph::Node`** — base that performs the *single, localized*
  `static_cast<GpuExecContext&>(ctx)` in `execute()` and forwards to a
  `record(GpuExecContext&)` override. Concrete GPU nodes never touch the cast; the rest
  of L3 never sees Vulkan. (Header-only `veng_gpu` INTERFACE lib.)
- **`Graph::execute(plan, scheduler, ExecContext&)`** overload — the driver injects a
  `GpuExecContext`; the default overload still builds the CPU context. Refactored the
  default to delegate and **dropped the `shared_ptr<ExecContextImpl>`** in favor of a
  stack/reference context: `execute` is frame-synchronous (band barriers block until
  every task completes), so the context provably outlives all tasks.
- **`Buffer::mapped()`** — captures VMA's `AllocationInfo::pMappedData`, so a
  host-visible (`eMapped`) buffer exposes its persistent host pointer (needed for
  staging/uniform uploads and the bridge test's readback).

### Proof (end-to-end on-GPU)

`tests/gpu/GpuBridgeTests.cpp` (1 case / 15 assertions): a `FillBufferNode` (a real
`GpuNode`) is dispatched by `Graph::execute` with an injected `GpuExecContext`; it
records `vkCmdFillBuffer` into the handed command buffer; the test submits + fences,
then reads the host-visible target through `mapped()` and asserts the GPU wrote the
expected value. It also asserts the seam delivered the right command buffer + frame
slot. This is the first time graph-driven work has actually executed on the GPU.

- **69/69 ctest green** (was 68; +1 bridge). Clean under the `llm-vcpkg` gate
  (GpuBridgeTests + SchedulerTests pass under ASan/UBSan — the `execute` refactor is
  race-free).

### Documented limitation (matches the roadmap)

The `GpuExecContext` carries **one** command buffer, and band dispatch currently shares
it across the band's nodes — correct under the serial `InlineScheduler`, but real
multi-threaded recording needs one command buffer per (queue, frame-slot, thread).
That is exactly the **`CommandManager`** in Step 3, where **M8**'s
threading/frames-in-flight contract also goes live.

### Where to continue (next iteration) — Step 3: L2 managers

- **`SwapchainManager`** over vk-bootstrap: owns the swapchain + per-frame binary
  acquire/render-finished semaphores; exposes `extent()` (feeds the `ScreenSize`
  source), `acquire(frame_slot)` and `present(frame_slot, wait)`, and `rebuild(extent)`.
- **`CommandManager`**: one pool per (queue, frame slot, thread); `begin(...)` /
  `one_shot(...)`; thin `synchronization2` barrier helpers. Resolves the single-command-
  buffer limitation above and M8's threading contract.
- **`SyncManager`**: one timeline semaphore per queue (the queue's revision clock) +
  the binary swapchain-handoff semaphores.

Then Step 4 (slice nodes: ScreenSize → raster→ persistent scene target → swapchain
source → present/blit), Step 5 (driver loop), Step 6 (triangle + caching proof).

### Deferred (per directive): M5, M6, M8 (until Step 3 CommandManager), L1–L7.

---

## Iteration 11 — 2026-05-25 — Architect directive + Step 1: L1 GPU resources (Buffer/Image)

### Architect directive (from feedback.md, supersedes prior priorities)

A maintainer/architect reviewed the working tree and changed the mode:

- **The implemented revision-model L3 core is canonical.** `design.md §2/§L3` text was
  rewritten to a dirty-flag model the code does not match; the **decision is to keep
  the code and reconcile the doc to it** — do NOT rewrite the core. (§L4/§L5 remain the
  spec for unbuilt work.) Acknowledged; not touching the core's model.
- **The engine cannot present a pixel — that is now the only priority.** Stop
  polishing; **defer all open LOW/MEDIUM findings (M5, M6, M8, L1–L7)** and further L2
  breadth unless a slice step needs it. Build the **vertical slice** in larger steps:
  **(1) L1 resources → (2) GPU bridge (`GpuExecContext`/`GpuNode`) → (3) L2 managers
  (Swapchain/Command/Sync) → (4) L4 slice nodes → (5) L5 driver loop → (6) proof
  (triangle window; static scene re-runs present/blit but raster node 0×)**.
- **Wiring rule:** the scene subgraph must NOT depend on the swapchain source; only the
  swapchain source is dirtied each frame (on acquire), so only present/blit re-runs.
- **Process:** commit the working tree on a branch each iteration so reviews grade
  reality (the uncommitted state is the cause of the stale-review lag). **Done this
  iteration** (see below).

### Step 1 — L1 GPU resources (design.md §L1)

- **`veng::Buffer`** (`veng_resources` lib) — RAII move-only buffer over VMA;
  `create(allocator, size, usage, memory=eAuto, flags) -> std::expected<Buffer,
  vk::Result>`; frees `{VkBuffer, VmaAllocation}` on destruction.
- **`veng::Image`** — RAII move-only 2D image + matching view; `create(allocator,
  device, extent, format, usage, aspect) -> std::expected<Image, vk::Result>`; frees
  `{VkImage, VmaAllocation, VkImageView}` (view first, then VMA image).
- **`Handle::generation` now honored** — `Graph::get_node`/`get_data` validate a
  per-slot generation, so a stale handle returns `nullptr` instead of aliasing a
  future occupant (the UAF/ABA guard the design wants before eviction). Append-only v1
  keeps generations at 0; the mechanism is real and tested.

### Tested + verified

- `tests/resources/ResourceTests.cpp` (4 cases / 20 assertions) on-GPU: Buffer/Image
  allocate, report accessors, and **move transfers ownership** (moved-from emptied).
- Generation guard test in GraphTests: a handle with a bumped generation resolves to
  `nullptr`.
- **68/68 ctest green** (was 63; +4 resources +1 generation).
- ResourceTests under the **`llm-vcpkg` gate** (ASan/UBSan): pass with **no non-dbus
  leak frames** — the RAII frees are clean (only the known NVIDIA-driver libdbus
  allocations from Context creation remain, suppressed via `detect_leaks=0`).

### This iteration's working tree is now committed on a branch

Per the directive, all accumulated loop work (iterations 1–11) plus this step is
committed on a branch so the next review pass grades the real tree rather than a stale
snapshot. Not pushed (no remote action without explicit request).

### Where to continue (next iteration) — Step 2: the GPU bridge

1. **`GpuExecContext : graph::ExecContext`** carrying the recording command buffer +
   resource/sync handles + frame slot, and a **`GpuNode`** base that does the single
   localized `static_cast<GpuExecContext&>(ctx)` (design.md §L4) — the L3→Vulkan seam.
2. Then **Step 3 L2 managers** (`SwapchainManager` over vk-bootstrap exposing
   `extent()`/acquire/present; `CommandManager`; `SyncManager` one-timeline-per-queue),
   **Step 4 slice nodes**, **Step 5 driver**, **Step 6 proof**.
   `Buffer`/`Image` (this step), `ComputePipeline`/`GraphicsPipeline`, and
   `DescriptorAllocator` are the building blocks now in place for the slice.

### Deferred (per directive) — do not action until a slice step needs them

M5 (error model), M6 (reflection wrong-default), M8 (DescriptorAllocator threading /
frames-in-flight contract), L1–L7 polish. M8 specifically wants closing when the
allocator meets the L5 driver (Step 5).

---

## Iteration 10 — 2026-05-25 — L2.1 GraphicsPipelineBuilder (dynamic rendering)

### Context

`feedback.md` pass 9 confirmed 57/57 and that the `llm-vcpkg` gate is earning its keep
(it caught a `[[nodiscard]]` slip last iteration). It re-flagged **M7** (already fixed
iter 9 — pass 9 reviewed iter-8 state; confirmed intact: `last_run_revision` present,
58 tests) and added **M8** (DescriptorAllocator threading/frames-in-flight *contract*,
explicitly "no live bug, unused today, close before L4/L5"). The reviewer again named
**`GraphicsPipelineBuilder`** the major remaining L2 piece, so this iteration built it.

### What landed — `GraphicsPipelineBuilder` + `GraphicsPipeline` (design.md §L2.1)

The graphics half of the L2 pipeline builders, mirroring `ComputePipeline`:

- **`GraphicsPipeline`** — RAII, move-only owner of `{pipeline, layout,
  descriptor_set_layout}` (same ordered-`destroy()` / null-after-move pattern).
- **`GraphicsPipelineBuilder(vertex, fragment)`** — fluent
  (`color_formats`/`depth_format`/`topology`/`rasterization`) →
  `std::expected<GraphicsPipeline, PipelineError>`. It is reflection-driven:
  - **vertex input** from the vertex shader's `VertexDetails` (via the existing
    `to_binding_description`/`to_attribute_description` helpers),
  - **descriptor-set layout** merged across both stages by binding (OR-ing stage
    flags), and **push-constant ranges** from reflection,
  - **stage compatibility** validated via `ShaderDetails::matches` (vertex outputs →
    fragment inputs) before touching the device,
  - **Vulkan 1.3 dynamic rendering**: built from attachment *formats*
    (`vk::PipelineRenderingCreateInfo` in `pNext`), no `VkRenderPass` (design §L2.1).
  - Sensible raster/depth/blend defaults; viewport+scissor are dynamic state.
- **Shared `PipelineError`** — factored into `pipelines/PipelineError.hpp` (was inline
  in `ComputePipeline.hpp`) so both builders share it; added `STAGE_INCOMPATIBLE` and
  `MISSING_COLOR_FORMATS`. `to_string` is now an inline `constexpr` in the header.

### Enabling change — Vulkan 1.3 features in `Context` (design §L0)

Dynamic-rendering pipeline creation (`renderPass == VK_NULL_HANDLE`) **requires** the
`dynamicRendering` feature at device creation (VUID-…-renderPass-06576), which the
Context did not enable. Added `VkPhysicalDeviceVulkan13Features{dynamicRendering,
synchronization2}` and `VkPhysicalDeviceVulkan12Features{timelineSemaphore}` — all
three mandated by design §L0 and load-bearing for L2/L5. Verified every existing
Vulkan test still creates its device.

### Tested + verified

- `tests/pipelines/GraphicsPipelineTests.cpp` (5 cases / 27 assertions), on-GPU:
  builds a real dynamic-rendering pipeline from the matching `simple.vert`/`simple.frag`
  pair; rejects **missing color formats**, **swapped stages** (WRONG_STAGE), and
  **incompatible interfaces** (STAGE_INCOMPATIBLE — passthrough.vert/frag need a
  geometry stage between them); **move** transfers ownership.
- **63/63 ctest green** (was 58; +5). The `PipelineError` refactor left ComputePipeline
  green (16 assertions).
- Clean under the **`llm-vcpkg` gate** (warnings-as-errors + ASan/UBSan); only the
  known NVIDIA-driver libdbus LSan reports (suppressed via `detect_leaks=0`).

### Known limitation introduced (documented, not a defect)

Both pipeline builders build a **single descriptor set layout (set 0)** and ignore the
reflected `set` index — fine for the current shaders, but multi-set shaders would need
one `vk::DescriptorSetLayout` per set. Note this before L4 nodes use multi-set shaders.

### Still open / next

- **M8** (DescriptorAllocator threading + frames-in-flight contract + `create_pool`
  error fidelity) — unused today; close before the L5 driver wires it in. Likely
  resolution: document "one allocator per recording thread / frame-slot; `reset()`
  only after that slot's frame retires" (the standard pattern; a global mutex would
  serialise recording), plus return the real `vk::Result` from `create_pool`.
- **M6** (reflection wrong-default) — now directly relevant: `to_vk_format` feeds the
  graphics builder's vertex input, so an exotic vertex format would silently mis-bind.
- **M5** (error-model convergence), **L1–L7** polish (L4 `debug_callback` quick fix).
- **Design scope:** `DescriptorWriter` (§L2.2, populate the sets the allocator hands
  out), **general history/ping-pong nodes** (§L4, double-buffered `Data`), **L1
  resources** (buffer/image RAII + handles over VMA), and the **L4 vertical slice** —
  now within reach (compute + graphics pipelines + descriptor allocation all exist).

### Where to continue (next iteration)

1. **`DescriptorWriter`** (§L2.2) — fluent `vk::WriteDescriptorSet` builder to populate
   allocated sets; the last piece needed to actually *bind* resources to a pipeline.
2. **L1 resource wrappers** (typed `Buffer`/`Image` over VMA + generation handles) —
   the prerequisite for an L4 vertical slice and for a meaningful DescriptorWriter test.
3. **M8** doc/contract fix (quick) and **M6** (fold the typed-error into reflection).

---

## Iteration 9 — 2026-05-25 — Fixed M7 (temporal reset repeated the C1 `== revision` trap)

### Context

`feedback.md` pass 8 caught up to iteration 7 and **verified the `needs_refresh` hook
is correct and regression-free** — but it found a genuine new bug, **M7**, with a
repro. (Pass 8 still lagged one iteration on iteration 8's DescriptorAllocator —
confirmed intact: header present, 57 tests.) M7 was the clear, actionable finding, so
this iteration fixed it.

### M7 (MEDIUM, real correctness bug) — reset detection missed undemanded changes — **FIXED**

The accumulation node's reset test was `scene->changed_at() == ctx.revision()` — the
**exact predicate that caused the original C1 bug**. `== revision` only sees a change
made *this* frame, so if the scene changes on a frame where the accumulator is
**undemanded**, the change is missed on the next demanded frame and the accumulator
keeps adding samples onto stale-scene output (a real path tracer would ghost). The
reviewer's repro (demand→[undemanded frame w/ scene change]→re-demand) returned count 3
instead of resetting to 1.

Fix — make the correct baseline a first-class, documented part of the node API so no
temporal node reaches for the `== revision` trap again:
- **`Node::last_run_revision()`** — a protected accessor returning `m_verified_at` (the
  revision the node last produced/was-confirmed-at; the core already maintains it, and
  it is the same baseline the C1 cutoff fix uses). Documented explicitly: temporal
  nodes detect "input changed since I last ran" via `data.changed_at() >
  last_run_revision()`, never `== current revision`.
- **`AccumulationNode` reset** is now `changed_at() > last_run_revision()`.
- **Regression test** added — "resets even when its input changed while undemanded" —
  and **proven meaningful**: I temporarily reintroduced the `==` predicate and
  confirmed the test fails (count 3), then restored the fix and it passes (count 1).
  This closes the demand-switching coverage gap the review noted (the prior 3 temporal
  cases only exercised the always-demanded path).

### Verification

- **58/58 ctest green** (was 57; +1 temporal regression). TemporalTests now 4 cases /
  56 assertions.
- TemporalTests clean under the **`llm-vcpkg` gate** (warnings-as-errors + ASan/UBSan).
- Library change is tiny and safe: one documented protected accessor on `Node` (no
  behaviour change for existing nodes); the rest is in the test fixture + a new test.

### Note on the recurring C1/M7 pattern

The `== revision` vs `> last_run_revision()` trap has now bitten twice (C1 in the core
cutoff, M7 in temporal reset). The fix in both cases is identical — compare an input's
`changed_at` against the consumer's last-run revision. The new `last_run_revision()`
accessor + its doc comment is the durable guard so the reusable L4 temporal/history
node (future work) inherits the correct pattern by construction.

### Still open / next (unchanged priorities)

- **M6** (reflection wrong-default on exotic types) — best folded into
  `GraphicsPipelineBuilder`. **M5** (error-model convergence; `merge_expected_t` dead
  code). **L1–L7** polish (L4 `debug_callback` is a quick fix).
- **Design scope:** `GraphicsPipelineBuilder` (§L2.1) + `DescriptorWriter` (§L2.2);
  **general history / ping-pong nodes** (§L4 — double-buffered `Data` +
  `temporal_inputs()` exclusion in resolve; now is the time to promote the accumulation
  pattern into a reusable library node that bakes in the correct `last_run_revision`
  reset); **L1 resources**; **L4 vertical slice**; **L5 driver**.

### Where to continue (next iteration)

1. **`GraphicsPipelineBuilder`** (§L2.1) following the ComputePipeline pattern, folding
   in **M6** (typed error for unhandled reflected formats) since vertex formats matter
   there. Testable on-GPU with `simple.vert`/`simple.frag`.
2. **General history nodes** — double-buffered `Data` + `temporal_inputs()`, promoting
   today's accumulation fixture into a reusable node with the M7-correct reset.
3. **`DescriptorWriter`** (§L2.2) to populate the sets `DescriptorAllocator` hands out.

---

## Iteration 8 — 2026-05-25 — L2.2 DescriptorAllocator (growable descriptor pool)

### Context

`feedback.md` pass 7 reported **no CRITICAL/HIGH items** and again lagged one iteration
(it lists "L3 temporal/history nodes" as still unbuilt, but iteration 7 shipped them —
confirmed intact: `needs_refresh` present, 54 tests). The reviewer states plainly that
the remaining work is **new design scope, not findings**, and names §L2.2
`DescriptorAllocator` as next. The open *findings* are M6 (reflection wrong-default,
exotic types only) and M5 (error-model cleanup) — both deferred this iteration with
reasons below.

So this iteration advanced design scope with the bounded, testable L2.2 piece that
ComputePipeline explicitly left as a `// TODO`: **descriptor allocation**.

### What landed — `veng::DescriptorAllocator` (design.md §L2.2)

A growable descriptor-pool allocator in a new `veng_descriptors` library:

- `allocate(layout) -> std::expected<vk::DescriptorSet, vk::Result>` — allocates from
  the current pool and, on `eErrorOutOfPoolMemory`/`eErrorFragmentedPool`, transparently
  spawns a fresh pool and retries (the design's "grows by adding pools on exhaustion").
  Non-recoverable results surface as a typed error.
- `reset()` — recycles every set by resetting all pools while keeping them, so
  steady-state framing never churns pool creation.
- **RAII, move-only**: destroys every pool on teardown; move nulls the source (no
  double-free). Lazy: no pool is created until the first `allocate`.
- Sensible default pool-size ratios across the common descriptor types; `sets_per_pool`
  and ratios are constructor-injectable.

Returns a typed `std::expected` (nudging M5's convergence on typed errors) rather than
the design sketch's bare `vk::DescriptorSet`, matching the project's error convention.

### Tested + verified

- `tests/descriptors/DescriptorAllocatorTests.cpp` (3 cases / 29 assertions), on the
  headless GPU, self-contained (builds its own one-binding layout): allocation
  succeeds; **pool growth** (tiny 2-set pools, 5 allocations ⇒ ≥3 pools); **reset**
  recycles without dropping pools; **move** transfers ownership (moved-from owns
  nothing).
- **57/57 ctest green** (was 54; +3).
- Clean under the **`llm-vcpkg` gate**: builds warnings-as-errors (fixed a
  `[[nodiscard]]` `resetDescriptorPool` result that `-Werror` would have rejected —
  exactly the class of bug the gate exists to catch), and runs clean under ASan/UBSan.
  The only LSan reports are the NVIDIA-driver libdbus allocations from `Context`
  creation (verified: zero frames in allocator/`createDescriptorPool` code; the gated
  flow uses `detect_leaks=0`).

### Deferred (with reasons)

- **M6 (reflection wrong-default)** — fixing it cleanly means threading typed errors
  out of the deep reflection extraction (`to_vk_descriptor_type`/`to_vk_format` →
  `extract_descriptors`/Details ctors → `create_shader`'s `expected`), and its exact
  path is only reachable with exotic types (Float16/64, Int8/16/64, unusual descriptor
  kinds) that are hard to trigger and test reliably. Higher effort + uncertain test
  than its severity (exotic-only) warrants this pass; still worth doing, ideally
  alongside `GraphicsPipelineBuilder` where vertex formats start to matter.
- **M5** (Shader bare-string errors; `merge_expected_t` dead code) — cross-layer
  decision; `DescriptorAllocator` returning a typed error nudges the right direction.
- **L1–L7** polish — unchanged; **L4** (`debug_callback` per-message `set_pattern` +
  the `"[IteratedFunction]"` string) remains a quick, satisfying fix.

### Where to continue (next iteration)

1. **`GraphicsPipelineBuilder`** (§L2.1) — the other half of L2 pipelines, following
   the `ComputePipelineBuilder` pattern (stages from reflection, vertex input from
   `VertexDetails`, dynamic-rendering color/depth formats, sensible raster/depth/blend
   defaults) → `std::expected<GraphicsPipeline, PipelineError>`, RAII. Testable on-GPU
   with the existing `simple.vert`/`simple.frag`.
2. **`DescriptorWriter`** (§L2.2) — the natural companion to `DescriptorAllocator`:
   build `vk::WriteDescriptorSet`s fluently from handles to populate allocated sets.
3. **M6** — best folded in while building `GraphicsPipelineBuilder`.
4. **L1 resources** (typed buffer/image handles + VMA wrappers + pools) and the **L4
   vertical slice** (ScreenSize → raster node → persistent target → present).

---

## Iteration 7 — 2026-05-25 — L3 temporal / accumulation (the design's showcase feature)

### Context

`feedback.md` pass 6 reported **no CRITICAL items** and re-flagged **H2** as top — but
H2 was resolved in iteration 6 (pass 6 snapshotted just before it; confirmed intact:
5 `llm-vcpkg` refs in `CMakePresets.json`, both scripts present). The reviewer
explicitly signalled the findings are nearly exhausted and that what remains is
"**unbuilt design scope** (L4 node types … temporal/history nodes) — new work rather
than review findings." So, after four feedback-driven iterations, this one **advances
the design**: the headline reactive feature, **progressive accumulation** (design.md
§1, §L4 "Temporal / history", §8 "temporal feedback is not a cycle").

### What landed — `Node::needs_refresh()` + the temporal self-edge

The core addition is one small, general hook plus its wiring:

- **`Node::needs_refresh()`** (virtual, default `false`): return `true` to be
  re-evaluated on every *demanded* frame even when inputs are unchanged — a
  progressive accumulator that hasn't converged. Wired into both halves of the frame:
  - `resolve`/`visit`: a refreshing node is marked dirty (so it is planned), and
  - `run_node`: a refreshing node is never change-cut-off (so it actually runs).
  Default-`false` means ordinary pure nodes are completely unaffected — a static scene
  still costs nothing (verified: all prior tests unchanged).

- **Temporal self-edge pattern** (no new graph machinery needed): a node reads its own
  previous output via the `ExecContext` but deliberately does **not** list that output
  in `inputs()`, so `resolve` sees no dependency and no cycle — exactly the design's
  "temporal feedback via history nodes, not real cycles" (§8). Because a producer
  writes its output after reading it (read-modify-write within one `execute`), the
  self-accumulation case needs no double-buffering.

### Demonstrated + tested

`tests/rendergraph/TemporalTests.cpp` (3 cases / 43 assertions) drives an
`AccumulationNode` fixture — the showcase reduced to the mechanism it stresses (one
sample added per demanded frame; resets to sample 1 when its scene input changes;
`needs_refresh` until `max_samples`). It proves the whole §1 thesis on the CPU:
- **Accumulates while idle** — sample count climbs 1→2→3→4 with no input change.
- **Converges, then costs nothing** — at `max_samples` it stops refreshing, drops out
  of the plan (empty frame), and the value holds.
- **Resets on invalidation** — mutating the scene drops it back to sample 1, then it
  resumes converging.
- **Stays demand-driven** — an *undemanded* accumulator never refreshes (still INVALID,
  output untouched).

### Verification

- Full suite: **54/54 ctest green** (was 51; +3 temporal).
- New code is clean under the **`llm-vcpkg` gate** (warnings-as-errors build + ASan/
  UBSan): `TemporalTests` and `GraphTests` both pass instrumented.
- Library change confined to `Node.hpp` (the hook) + `Graph.cpp` (two wiring lines);
  the `AccumulationNode` lives in the test as a fixture (the reusable/ GPU accumulation
  node is L4 work — kept out of the public API until then to avoid premature surface).

### Deferred this iteration (with reasons)

- **M6 (NEW, reflection substitutes wrong types)** — `to_vk_descriptor_type` /
  `to_vk_format` return a plausible-but-wrong default on unhandled inputs (same class
  as the fixed M4). Real edge-robustness bug in L0; deferred to keep this iteration on
  the headline feature. Good next target — common types are handled so normal shaders
  are unaffected, but it should surface a typed error.
- **M5** (Shader bare-string errors; `merge_expected_t` dead code), **L1–L7** polish —
  unchanged from iter-6 notes.
- **General history / ping-pong nodes** (reading *another* node's previous output, e.g.
  TAA history, prev-depth) — needs double-buffered `Data` + a `temporal_inputs()`
  exclusion in `resolve`. The self-accumulation case covers the core mechanism; general
  history is the natural next temporal step.

### Where to continue (next iteration)

1. **M6** — make the reflection mappings fail (typed error) on unhandled
   descriptor/format types instead of substituting; add the shader test the review
   asks for. Small, clean, closes the last "wrong default" hazard.
2. **L2 breadth** — `GraphicsPipelineBuilder` (§L2.1) following the ComputePipeline
   pattern, and `DescriptorAllocator`/`Writer` (§L2.2). Needed before any raster L4 node.
3. **General history nodes** — double-buffered `Data` + `temporal_inputs()` for
   ping-pong (TAA/motion-vector style), building on this iteration's `needs_refresh`.
4. **L4 vertical slice** — `ScreenSize` source → GPU raster node → persistent scene
   target → present (the "static scene presents without re-rendering" end-to-end proof).

---

## Iteration 6 — 2026-05-25 — H2: added the `llm-vcpkg` safety gate + helper scripts

### Context

`feedback.md` pass 5 confirmed **no CRITICAL items remain** (C2/H3 verified on-GPU),
and named **H2** (missing `llm-vcpkg` preset + `fix_and_format.sh` / `run_test.sh`) the
top open item. (Pass 5 still listed M4/M1 as open, but it snapshotted the tree just
before iteration 5 landed those — re-verified present here: `create_shader_module`
returns `std::expected`, `Context::create` uses `app_name.c_str()`, `operator<=>`
gone.) So this iteration did H2.

### H2 (HIGH) — `llm-vcpkg` preset + helper scripts — **RESOLVED**

- **`llm-vcpkg` preset** (configure + build + test) in `CMakePresets.json`: inherits
  `dev-vcpkg`, sets `TREAT_WARNINGS_AS_ERRORS=ON`, and enables **ASan + UBSan**. Key
  detail: the sanitizers are applied via global `CMAKE_CXX_FLAGS` /
  `CMAKE_EXE_LINKER_FLAGS` rather than the `ENABLE_SANITIZER_*` cache vars — those only
  reach `target_add_*` targets, but the **test executables use plain `add_executable`**,
  so the cache-var route would mismatch (instrumented libs linked into an
  un-instrumented exe → `undefined __asan_*`). Global flags cover everything,
  including tests.
- **`fix_and_format.sh`** — clang-format in place over the tree, then advisory
  clang-tidy (scoped to `src/*.cpp`, focused checks) and cppcheck. Verified: runs to
  completion; format is a no-op on the already-formatted tree; tidy/cppcheck surface
  only advisory nits (swappable params, a clone branch in `Shader.cpp`; cppcheck's
  `unknownMacro` on the Vulkan dispatch macro is expected).
- **`run_test.sh`** — a dedicated gcov-instrumented build (`--coverage
  -fprofile-update=atomic`, so the threaded SchedulerTests don't corrupt counters),
  runs ctest, and emits an lcov report (filtering vcpkg/system/tests), with lcov-2.x
  `--ignore-errors` for the usual gcov mismatches. Documents the sanitized-gate
  command in its header.

### Verification

- **Build:** `cmake --build --preset llm-vcpkg` links **clean under
  warnings-as-errors** — confirmed zero warnings across *all* veng targets (libs,
  playground, bench) under `-Wall -Wextra -Wpedantic` first, so `-Werror` is safe.
- **Tests under ASan/UBSan:** `ASAN_OPTIONS=detect_leaks=0 ctest --preset llm-vcpkg`
  → **51/51 passed** (161 s; sanitizers slow it ~5×). Vulkan-free GraphTests/Scheduler
  are clean with leak detection *on*; the Vulkan tests only "leak" via **libdbus**
  inside the NVIDIA driver/loader during device enumeration (confirmed by reading the
  LSan stacks — no veng frames), hence `detect_leaks=0` in the gated flow.
- `run_test.sh` coverage flow: **works end-to-end** — 51/51 pass in the instrumented
  build and lcov reports **82.1% lines / 84.5% functions** overall. Critically, the
  L3 core meets the design's §10 target: **`Graph.cpp` 95.0%**, `Node.hpp` 95.8%,
  `Graph.hpp` 100%. The Vulkan layers are lower (Context.cpp 68%, ComputePipeline.cpp
  70%, Shader.cpp 80%) — their failure paths need device fault injection to reach.

### Why this matters

A warnings-as-errors + sanitizer gate is exactly what would have caught C2's original
missing `return` (UB under `-Werror=return-type`) before it shipped. The machinery
(`ENABLE_SANITIZER_*`, `TREAT_WARNINGS_AS_ERRORS`) already existed; this wires it into
a single named preset + scripts the workflow (and `CLAUDE.md`) assume.

### Still open / next

- **M5** (Shader still returns bare `std::string` errors — the lone outlier now that
  ComputePipeline has `PipelineError`; `merge_expected_t` remains dead code). A
  cross-layer error-model decision; good dedicated pass.
- **L1/L2/L3/L4/L5/L7** cleanups (per feedback); none urgent for v1. **L4**
  (`debug_callback` resets the log pattern per message and still says
  `"[IteratedFunction]"`) is a quick, satisfying fix.
- **Design features (the real roadmap):** **L3 temporal/history nodes** (§L4/§8 —
  accumulation/path-tracer showcase: a non-dependency temporal edge + double-buffered
  Data) and **L2 breadth** (`GraphicsPipelineBuilder` §L2.1, `DescriptorAllocator`/
  `Writer` §L2.2), then an **L4 vertical slice** + **L5 driver**.

Recommended next iteration: **L3 temporal/history nodes** — it is the highest-value
remaining *core* feature, fully testable without new infrastructure, and unlocks the
design's headline accumulation demo.

---

## Iteration 5 — 2026-05-25 — Hardened the L0 Shader/Context layer (M4, M1, L6)

### Context

`feedback.md` reached pass 4. It **verified C1/H1/M3 fixed** (good), but it is still
**one iteration behind**: it re-flags **C2** and **H3** as open against the old
`ComputePipeline` blob `b26af9e`, whereas iteration 4 already rewrote `ComputePipeline`
(RAII + `std::expected` builder) and unblocked `ctest` (root `enable_testing()` +
factory-based shader test). Confirmed both still intact this iteration: `ctest`
discovers and passes **51/51** from the root build dir. (Work remains uncommitted, so
the reviewer keeps grading an older snapshot.)

So this iteration took the genuinely-open items the review emphasised — all in the
**L0 Vulkan foundation** (`Shader`/`Context`), which is now load-bearing because the
new `ComputePipelineBuilder` consumes `Shader`s.

### M4 (MEDIUM, real correctness bug) — Shader swallowed module-creation failure — **FIXED**

`create_shader_module` logged a failed `vkCreateShaderModule` but returned the null
handle, and `create_shader` wrapped it in a **successful** `std::expected<Shader>` —
a broken Shader would then flow into pipeline creation and crash the GPU with the
cause buried in one log line. Fix: `create_shader_module` now returns
`std::expected<vk::ShaderModule, std::string>` and `create_shader` propagates the
error (`return std::unexpected{module.error()}`) before constructing the Shader. The
existing failure-path tests (`nonexistent shader`, `invalid entry point`) already
guard the "create_shader never returns a broken Shader on failure" contract; this
closes the one step that previously violated it. (The device-level
module-creation-failure path itself needs fault injection to unit-test — not feasible
through the concrete `vk::Device` — so it is covered by the contract tests + code
review rather than a dedicated case.)

### M1 (MEDIUM) — non-NUL-terminated `string_view` to a C API — **FIXED**

`Context::create` passed `title.data()` to `vkb::InstanceBuilder::set_app_name`
(a `const char*`); `string_view::data()` isn't guaranteed NUL-terminated. Now copies
into a local `std::string` and passes `.c_str()`.

### L6 (LOW) — `Shader::operator<=>` ordered by stage only — **FIXED (dropped)**

It ordered two distinct shaders of the same stage as equivalent and, being a
user-defined `<=>`, generated no `operator==`. It had **zero uses** anywhere, and
`Shader` is a move-only resource handle with no meaningful ordering — so it was
removed rather than given a contrived identity.

### Verification

- Full build clean; `ctest` from the root build dir: **51/51 passed** (unchanged
  count — these are correctness/cleanup fixes, and the shader happy-path tests
  confirm M4 didn't regress valid loads).
- Changes confined to `src/shader/Shader.cpp`, `src/context/Context.cpp`,
  `include/veng/shader/Shader.hpp`.

### Still open (deferred, with reasons)

- **H2 (`llm-vcpkg` preset + `fix_and_format.sh`/`run_test.sh` absent)** — the review
  keeps flagging this; it's the obvious next infra task (a warnings-as-errors gate
  would have caught C2 originally). Deferred only to keep this iteration focused on
  the correctness bugs; should be done next.
- **M5 (inconsistent error model; `merge_expected_t` dead code)** — needs a
  cross-layer decision (standardise on `ResultVariant`/`merge_expected_t`, or delete
  the unused machinery). Bigger than a single fix; worth a dedicated pass.
- **M2 residual** — a *persistently* failing node re-runs every frame (no backoff).
  Policy is §9-valid; backoff/propagate-after-N is a later option.
- **L1** (per-frame `dynamic_cast` in TransformNode), **L2** (recursive `visit` →
  stack-overflow risk on deep chains), **L3** (global-namespace error types),
  **L4** (`debug_callback` resets the log pattern per message and still says
  `"[IteratedFunction]"` — set once at init), **L5** (`Handle::generation` ignored),
  **L7** (global slang session not concurrency-safe) — all noted; none urgent for v1.

### Where to continue (next iteration)

1. **H2 infra** — add the `llm-vcpkg` preset (inherit `dev-vcpkg` + warnings-as-errors
   + ASan/UBSan; clang-tidy/cppcheck only if installed) and the two helper scripts,
   or correct `CLAUDE.md`. Quick, high-leverage, and unblocks the mandated gate.
2. **L3 temporal/history nodes** (design §L4/§8) — the top remaining *core* feature
   (accumulation/path-tracer showcase): a non-dependency temporal edge that reads the
   previous revision + double-buffered `Data`.
3. **L2 breadth** — `GraphicsPipelineBuilder` (§L2.1) and `DescriptorAllocator`/
   `Writer` (§L2.2), following the ComputePipeline builder pattern.
4. **L4 vertical slice** — `ScreenSize` source → GPU raster node → persistent scene
   target → present, bridging L1/L2 into L3 via a GPU `ExecContext`.

### Note for the maintainer

The review pass grades committed git state, but every loop iteration's work (C1, H1,
M3, C2, H3, and now M4/M1/L6) lives in the **uncommitted working tree** — which is
why the critique keeps re-listing already-fixed items. Committing the tree (on a
branch) would let the review reflect reality. Not done here (no request to commit).

---

## Iteration 4 — 2026-05-25 — Unblocked ctest (H3) + rebuilt ComputePipeline (C2); GPU is available

### Context correction

`feedback.md` advanced to "pass 3" but was **reviewed against the old committed blob
`fa7580b`** (it says "the tree is byte-identical to pass 2"). So its re-flags of **C1,
H1, M3 are stale** — those were fixed in iteration 3 and verified again here (the C1
repro returns 20, not the stale 10). The reviewer evidently reads committed git
state; this loop's work lives in the **working tree (uncommitted)**, which is why the
review keeps re-seeing pre-iteration-3 code. Did not commit (no user request to).

**Also: the long-standing open question is answered by the code itself** —
`veng::Context::create(std::string_view) -> std::expected<Context, ContextCreationError>`
already exists. And a **real GPU (RTX 4060) is present**, so Vulkan layers are
testable here (ContextTests/ShaderLoadingTests pass on-device).

### H3 (HIGH) — `ctest` discovery — **RESOLVED (two distinct bugs)**

1. `tests/shader/ShaderValidationTests.cpp` used the removed `veng::Context("str")`
   ctor. Replaced its 12 uses with a `make_context()` helper over the factory
   (mirrors ShaderLoadingTests). The target builds again.
2. Even with all targets building, `ctest -N` still reported **0** — the *root*
   `CMakeLists.txt` never called `enable_testing()`, so the root `CTestTestfile.cmake`
   didn't recurse into `tests/` (discovery files existed only under `tests/`). Added
   `enable_testing()` before the `add_subdirectory` calls.

`ctest` now discovers and passes **51/51** tests from the root build dir.

### C2 (CRITICAL) — `ComputePipeline` non-functional — **RESOLVED**

Rewrote the L2 compute pipeline to the design's §L2.1 shape:
- Moved into `namespace veng`; added `PipelineError` + `to_string`.
- `ComputePipeline` is now an **RAII, move-only owner** of `{pipeline,
  pipeline_layout, descriptor_set_layout}` with a `destroy()` that frees in order and
  nulls handles (no leaks, no double-free on move). Dropped the stray
  descriptor-pool / command-pool / fence / queue members — those are separate
  concerns (DescriptorAllocator §L2.2, Command/Sync managers §L2.4).
- New `ComputePipelineBuilder(const Shader&).build(context, cache={}) ->
  std::expected<ComputePipeline, PipelineError>`: derives the descriptor-set layout
  and push-constant ranges **from reflection** (no hand-written create-info), checks
  every Vulkan result, cleans up partial state on failure, validates the compute
  stage. Removes the missing-`return` UB and the asserts the review flagged.
- Added a compute test shader `shaders/tests/loading/compute/simple_comp.slang`
  (storage buffer + push constant) and `tests/pipelines/ComputePipelineTests.cpp`
  (builds on-device, rejects a non-compute shader, move transfers ownership) — 3
  cases / 16 assertions, all green.

### Verification

- Full build clean; `ctest` from the root build dir: **51/51 passed** (was 0
  discoverable). New: 3 ComputePipeline cases. Existing GraphTests (15) /
  SchedulerTests (6) / shader / context suites all still pass.

### Still open (deferred, with reasons)

- **M4 (NEW, Shader)** — `create_shader_module` returns a null module on failure and
  `create_shader` wraps it in a *successful* `expected`, so a broken Shader flows
  into pipeline creation. Real bug; in the 1045-line `Shader.cpp`. Next-iteration
  target (now that the pipeline builder consumes Shaders, this matters more).
- **M1 (Context `string_view::data()` not NUL-terminated)** — one-line fix in
  `Context.cpp`; left grouped with the Context work.
- **M2 (FAILED busy-retry/propagation)** — policy made explicit in code last
  iteration; richer policy deferred.
- **M5 (inconsistent error model; `merge_expected_t` is dead code)**, **L1/L2/L6/L7**
  — cleanups; L6 (`Shader::operator<=>` orders by stage only, yields no `==`) is a
  quick correctness fix worth doing soon.
- **H2 (`llm-vcpkg` preset + scripts absent)** — still unaddressed; the warnings-as-
  errors gate it implies would have caught C2 originally. Good next infra task.

### Where to continue (next iteration)

1. **M4** — make `Shader::create_shader_module` return `std::expected` and have
   `create_shader` propagate it (no silent null-module Shaders). Add the failure-path
   test the review asks for.
2. **H2 infra** — add the `llm-vcpkg` preset (warnings-as-errors + ASan/UBSan; add
   clang-tidy/cppcheck only if installed) + `fix_and_format.sh` / `run_test.sh`.
3. **L3 temporal/history nodes** (design §L4/§8) — still the top remaining *core*
   feature (accumulation/path-tracer showcase): a non-dependency temporal edge +
   double-buffered Data.
4. **L2 breadth** — `GraphicsPipelineBuilder` (§L2.1) and `DescriptorAllocator`/
   `Writer` (§L2.2), now that the ComputePipeline builder establishes the pattern.

---

## Iteration 3 — 2026-05-25 — Consumed the review: fixed C1 (cutoff) + H1 (exceptions)

### Summary

First iteration with a populated `feedback.md` (review pass 2). Focused on the two
items that were both **actionable in the L3/L5 core I own** and severity CRITICAL/
HIGH; the rest are Vulkan-layer items blocked on the `Context` API decision, or
infra/meta (see "Deferred" below).

### C1 (CRITICAL) — within-frame cutoff left demand-switched nodes permanently stale — **FIXED**

My iteration-1 cutoff predicate was `input.changed_at == current_revision`, but
`resolve`'s dirtiness test is `input.changed_at > verified_at`. The mismatch meant a
node dirtied on a frame where it was **not demanded** got planned by resolve but
skipped by the cutoff — and because resolve had already advanced its `verified_at`,
it read clean forever. Reproduced exactly (demand A → mutate source → demand only B →
re-demand A returned the stale value).

Fix (matches the reviewer's direction):
- `resolve`/`visit` **no longer advances `verified_at`** — planning must not mark a
  node up-to-date before it runs.
- `verified_at` is now advanced only in `run_node` (execute), on both the
  recompute path and the cutoff-skip path (a cut-off node *is* confirmed current),
  never on FAILED.
- The cutoff baseline is now `input.changed_at > node.verified_at` for every input —
  identical to resolve's dirtiness test, so the two can no longer disagree. This also
  resolves the substance of **M3** (resolve no longer mutates validity stamps).

Verified by the standalone repro (now passes) and two new regression tests covering
the previously-untested **demand-switching** path (the coverage gap the review
called out): "dirtied while undemanded → recomputes when next demanded" and "dirty
across several undemanded frames → catches up." The existing cutoff/caching tests and
the playground caching demo are unchanged in behaviour.

### H1 (HIGH) — a throwing node hung the frame and terminated a worker — **FIXED**

`run_node` now wraps `node.execute()` in `try/catch(...)`; any exception becomes a
`FAILED` completion (CAS out of PROCESSING + `notify_all`). Previously a throw left
the node stuck in PROCESSING (band barrier waits forever) and escaped the worker
thread (`std::terminate`). New SchedulerTests case asserts a throwing node yields a
FAILED node and a returning (non-hanging) frame under the real thread pool.

### Tests / verification

- GraphTests: **15 cases / 94 assertions** (was 13/68) — added the two C1 regressions.
- SchedulerTests: **6 cases / 180 assertions** (was 5/178) — added the H1 throw test.
- Re-ran both suites under **ThreadSanitizer** (3×) after the change — clean. The C1
  fix moved `verified_at` writes into the worker threads; each node's stamp is still
  written only by its own task and read by next frame's resolve after the band
  barrier, so there is no new race.
- `feedback.md` positives (band dispatch, `m_heights`, `ThreadPoolScheduler`) were
  re-confirmed correct by the reviewer and are untouched.

### Deferred this iteration (with reasons)

- **C2 / H3 (ComputePipeline non-functional; ShaderValidationTests uses removed
  `Context(const char*)`)** — both Vulkan-layer, gated on the author's intended
  `Context` factory shape (still the open question). Untouched.
- **H2 (no `llm-vcpkg` preset / `fix_and_format.sh` / `run_test.sh`)** — pure infra;
  worth adding next (a warnings-as-errors gate would have caught C2's missing
  `return`), but skipped here to keep the iteration correctness-focused and avoid a
  clang-tidy/cppcheck-availability rabbit hole.
- **M1 (Context `string_view::data()` not NUL-terminated)** — real one-line UB fix
  but in the WIP `Context.cpp`; leaving it with the other Context-layer items.
- **M2 (FAILED busy-retry + no propagation)** — made the policy explicit in code
  (retry-on-demand, hold-last-good, no propagation) per §9; a richer policy
  (backoff / propagate) is a deliberate later choice.
- **L1 (per-frame `dynamic_cast` in TransformNode)** — LOW; cache `ValueData<T>*` at
  wiring to drop RTTI from the hot path. Noted, not urgent.

### Where to continue (next iteration)

1. **H2 infra**: add the `llm-vcpkg` preset (inherit `dev-vcpkg` + warnings-as-errors
   + ASan/UBSan; add clang-tidy/cppcheck only if installed) and the two helper
   scripts, or correct `CLAUDE.md`. This gives the safety gate the project assumes.
2. **L3 temporal/history nodes** (design §L4/§8) — still the highest-value remaining
   *core* feature (accumulation/path-tracer showcase). Needs a non-dependency
   "temporal edge" that reads the previous revision, plus double-buffered `Data`.
3. **L5 frame driver/clock** with a pluggable present sink (GPU-free, testable).
4. **Vulkan layers (C2/H3/M1)** once the `Context` API shape is decided.

---

## Iteration 2 — 2026-05-25 — L5 Job System + height-batched execute

### Summary

Implemented the **L5 job system** — a real multi-threaded `Scheduler` — and, in the
process, fixed a latent deadlock in `Graph::execute` that the iteration-1 inline
scheduler had hidden. `feedback.md` was **still empty**, so the work followed
`design.md` and my own iteration-1 "where to continue" note (the threaded scheduler
was item 1).

### The deadlock that the inline scheduler hid

Iteration-1 `execute` had each consumer task **block** (atomic wait) on its
producer task. Under a fixed-size pool, if blocked consumers occupy every worker
while their producers sit queued, the pool deadlocks (classic "pool task blocks on
another pool task"). The inline scheduler never exposed this because it ran tasks
synchronously in height order.

**Fix — height-batched dispatch (design.md §L5):** `execute` now dispatches the
plan one topological-height band at a time, with a barrier between bands. Nodes
within a band are mutually independent (deps have strictly lower height, already
finished), so **no task ever blocks on another task** → deadlock-free at any pool
size. The per-node dependency wait is gone; the band barrier replaces it, and a
frame is now a synchronous unit (what the L5 driver loop needs). Within-frame
change-cutoff is preserved (a band task checks whether any input's `changed_at`
equals this revision).

### What landed (all builds + tests green, ThreadSanitizer-clean)

- `include/veng/rendergraph/ThreadPoolScheduler.hpp` + `src/.../ThreadPoolScheduler.cpp`
  — fixed worker pool draining a shared task queue, implementing `graph::Scheduler`.
  Single queue for v1 (work-stealing is a noted future throughput optimization).
- `FramePlan` now carries per-node `m_heights` (parallel to `m_order`) so `execute`
  can form height bands; `resolve` populates it.
- `Graph::execute` rewritten to band dispatch + barrier; extracted `run_node`
  (cutoff → execute → CAS completion → output stamping).
- `tests/rendergraph/SchedulerTests.cpp` — 5 cases / 178 assertions: diamond matches
  inline result (shared dep runs once), wide band aggregates correctly, a 32-deep
  chain on a 2-thread pool does **not** deadlock, 70 repeated frames keep
  caching/cutoff exact, single-thread pool drains every band.
- CMake: `find_package(Threads)`, link `Threads::Threads` into `veng_render_graph`,
  added the `ThreadPoolScheduler.cpp` source and the `SchedulerTests` target.

### Verify / how to run

- `cmake --build --preset dev-vcpkg --target GraphTests SchedulerTests`
- `./build/dev-vcpkg/tests/GraphTests` → 68 assertions / 13 cases.
- `./build/dev-vcpkg/tests/SchedulerTests` → 178 assertions / 5 cases.
- **ThreadSanitizer:** configured an isolated `build/tsan` with
  `-DCMAKE_CXX_FLAGS="-fsanitize=thread -g" -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"`
  (the test exes use plain `add_executable`, so sanitizers must be applied globally,
  not via the `ENABLE_SANITIZER_*` cache vars which only affect `target_add_*`).
  Both suites ran clean across repeated runs — no data races. (`build/tsan` removed
  afterward; it is gitignored anyway.)

### Where to continue (next iteration)

L3 + the L5 job pool are done and verified. Suggested next, in order:

1. **L5 frame driver / clock:** a small driver that loops resolve→execute→(present
   hook) and owns the `Scheduler`, with frames-in-flight bookkeeping. Still
   GPU-free if the "present" step is a pluggable sink — testable with a fake sink.
2. **L3 temporal/history nodes (design.md §L4 "Temporal / history", §8):** the
   showcase accumulation/path-tracer needs feedback edges that read the *previous*
   revision. Current cycle detection would reject these — they need a distinct
   "temporal edge" kind that does not count as a dependency in `visit`, plus
   double-buffered `Data`. This is the highest-value remaining *core* feature.
3. **Multi-output stamping precision:** `run_node` stamps *all* of a node's outputs
   when any output changed (conservative). Fine for single-output transforms;
   refine if/when multi-output nodes appear.
4. **Unblock the Vulkan layers (L0–L2)** once the `Context` "rusty" API shape is
   known — see the open question below and iteration 1's notes.

### Note on the iteration-1 "known breakage" list

Still accurate: `tests/shader/ShaderValidationTests.cpp` (removed `Context(const
char*)` ctor) blocks `ctest` discovery for the whole project; `ComputePipeline::create`
is incomplete; there is no `llm-vcpkg` preset / `fix_and_format.sh` / `run_test.sh`.
None addressed this iteration (out of scope; needs the author's `Context` direction).

---

## Iteration 1 — 2026-05-25 — L3 Reactive Core

### Summary

Implemented the **L3 Reactive Core** (`veng::graph`) end to end — the design's
"heart" (§L3) and the layer it calls out as Vulkan-free and the priority for test
coverage (§10). This replaces the prototype's racy push-dirty model
(`playground/main.cpp`) with the revision-based pull model from §2, and fixes the
three documented hazards (completion race, raw-pointer edges, thread-per-node).

`feedback.md` was **empty** this iteration, so there was no critique to fold in;
the work followed `design.md` directly and erred toward a small, fully-tested core.

### What landed (all builds + tests green)

New / rewritten files:

- `include/veng/rendergraph/RenderGraphCommon.hpp` — `Revision`, `ExecutionState`
  (VALID/PROCESSING/INVALID/FAILED), generation-tagged `Handle<Tag>` →
  `NodeHandle`/`DataHandle`, and `GraphError`/`ExecError` with `to_string`.
- `include/veng/rendergraph/data/Data.hpp` — `Data` base (producer + `changed_at`
  stamp, frame-boundary `commit_pending`) and typed `ValueData<T>` (source `set`,
  producer `produce`, equality-gated cutoff with an always-changed fallback for
  non-comparable `T`).
- `include/veng/rendergraph/nodes/Node.hpp` — `Node` base (atomic `ExecutionState`
  + `verified_at`), the `ExecContext` and `Scheduler` DI interfaces, an
  `InlineScheduler` (deterministic test runner / single-threaded default), and the
  generic `TransformNode<R(Args...)>` workhorse.
- `include/veng/rendergraph/Graph.hpp` + `src/rendergraph/Graph.cpp` — the `Graph`:
  ownership of nodes/data, typed `add_source` / `add_transform` authoring helpers,
  `mutate_source`, the sync **resolve** pass (apply mutations → bump revision →
  walk demanded cone → cycle detection + topological-height ordering → `FramePlan`),
  the async **execute** pass (CAS completion, per-output `changed_at` stamping,
  dependency wait via atomic `wait`/`notify`), and `to_dot()`.
- `tests/rendergraph/GraphTests.cpp` — 13 Catch2 cases / 68 assertions: fresh
  execution, cross-frame caching, source invalidation, source + downstream cutoff,
  diamond (single shared eval + height order), cycle rejection, INVALID_SINK,
  wrong-type → FAILED, lazy (undemanded) eval, revision stamping, non-comparable
  fallback, `to_dot`.
- `playground/main.cpp` — replaced the dead prototype with a working IFS demo on the
  new core (proves caching: re-present runs 0 nodes; rename re-runs the diamond;
  equal-value set is gated to 0 nodes).

CMake: `veng_render_graph` is now Vulkan-agnostic (compiles only `Graph.cpp`, owns
its include dir, no link to `veng_pipelines`); removed the obsolete
`src/rendergraph/{nodes/Node.cpp,data/Data.cpp}`; added the `GraphTests` target;
linked `playground` against `veng_render_graph`.

### Key design decision made while implementing

**Change-cutoff is applied *within* the frame, not just across frames.** `resolve`
plans conservatively (the whole dirty cone), but `execute` dynamically skips a
*previously-valid* node whose inputs did not actually change this revision (detected
via `input.changed_at == current_revision`). A node that has never produced a valid
output always runs. This is what makes §2.4's "sub-pixel jitter doesn't ripple"
behavior real (see the abs() cutoff test) rather than a one-frame-late effect.

### Verify / how to run

- Build the core + tests: `cmake --build --preset dev-vcpkg --target GraphTests`
- Run tests: `./build/dev-vcpkg/tests/GraphTests` → *All tests passed (68 assertions
  in 13 test cases)*.
- Demo: `./build/dev-vcpkg/playground/playground`.

### Known pre-existing breakage (NOT caused by this iteration, NOT yet addressed)

These were already broken in the working tree (the "Life is hard" + "rusty Context"
WIP) before this iteration; they are in files untouched by L3 work:

1. **`tests/shader/ShaderValidationTests.cpp:18`** uses the removed
   `veng::Context("string")` constructor → fails to build. Because
   `catch_discover_tests` runs discovery as a post-build step, this one failing
   target blocks `ctest` discovery for the *whole* project (`ctest -N` shows 0
   tests). Each test binary still runs fine when invoked directly.
2. **`src/pipelines/ComputePipeline.cpp`** `create()` is incomplete: no `return`
   (control reaches end of non-void), unused `pipeline`, and the `// TODO`
   descriptor pool/set block (design §L2.1). Compiles with warnings under `dev`
   (warnings-as-errors is OFF there) but is non-functional.
3. **`CMakePresets.json` has no `llm-vcpkg` preset** even though `CLAUDE.md`
   instructs using it. The configured presets are `dev-vcpkg` / `release-vcpkg`.
   `./fix_and_format.sh` and `./run_test.sh` (referenced in `CLAUDE.md`) also do not
   exist; `clang-format` (v22) works directly with the repo `.clang-format`.

### Where to continue (next iteration)

Roadmap (design §11). L3 is done; suggested order:

1. **L5 driver seam against L3 (still Vulkan-free):** a real `Scheduler` —
   a work-stealing job pool implementing `graph::Scheduler` — plus a frame
   driver/clock loop. Can be unit-tested without a GPU and exercises the async path
   (the inline scheduler currently never blocks on the dep-wait).
2. **L3 polish:** serialize topology (§L6) for regression tests; consider
   per-output (not per-node) `changed_at` stamping for multi-output nodes (current
   code conservatively stamps all outputs of a changed node); add history/ping-pong
   node support (temporal feedback is explicitly *not* a cycle — needs a dedicated
   edge kind so resolve doesn't reject it).
3. **Unblock the Vulkan layers (L0–L2):** decide the new `veng::Context` API shape
   (the "rusty" refactor implies an `expected`-returning factory), then fix
   `ShaderValidationTests` and finish `ComputePipeline::create` (descriptor
   allocator/writer + return value, design §L2.1/§L2.2). This unblocks `ctest`.
4. **L4 vertical slice:** `ScreenSize` source → one GPU raster node → persistent
   scene target → swapchain/present node, bridging L1/L2 into L3's `Node`/`Data`
   (the `ExecContext` interface is the seam — an L4 GPU `ExecContext` can layer
   command recording over the core one). Prove "static scene presents without
   re-rendering" end to end.

### Open questions for the author

- Is the `Context` "rusty" refactor meant to expose `static expected<Context,…>
  create(...)`? Knowing the intended shape unblocks fixing the shader test cleanly.
- Should `feedback.md` be populated with the critique referenced in the loop prompt?
  It was empty, so no critique was applied this iteration.
