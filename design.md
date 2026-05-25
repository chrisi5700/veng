# veng — Reactive Frame Graph: Design Specification

> Status: design draft. The **core** interface sketches (§2, §4 L3) are intended to
> be implementable roughly as written; Vulkan call sequences in the lower layers
> remain illustrative. Conventions follow `CLAUDE.md` (C++23/26, `snake_case`
> functions/variables, `CamelCase` types, `m_` private members, `UPPER_CASE`
> constants, `std::expected` for fallible APIs).

---

## 1. Overview & Goals

`veng` is a **reactive frame graph**: a persistent dependency graph in which both
GPU pipelines *and* their inputs — including CPU-side resources — are nodes. When
an input changes it is marked dirty; on the next frame we re-execute only the part
of the graph that is both **dirty** and **demanded** by what we are presenting. If
nothing relevant changed, we re-present the previously rendered image and burn
almost no GPU.

This is **demand-bounded, invalidation-driven incremental computation** applied to
rendering. It is *not* a Frostbite-style transient frame graph (rebuilt every
frame for aliasing + automatic barriers). For the Vulkan half, Themaister's
*Granite* render graph is the best reference; for the incremental half, the
push/pull reactive lineage (MobX/SolidJS signals) and Jane Street's Incremental
(height-ordered scheduling). See §12.

### Goals

- **Render only what changed.** A static scene with a static camera costs a blit
  and a present, nothing more.
- **Precise invalidation.** Savings scale with dependency granularity, so the
  graph topology must let a single moving object dirty only its own work.
- **A Vulkan-agnostic, testable core.** The graph algorithm (dirty propagation,
  the forward sweep, the dataflow executor, cycle detection) has zero Vulkan
  dependencies and is unit-tested with fake nodes against an injected scheduler.
- **Type-safe value flow via templates** — `Data<T>`, no `std::any`/erasure.
- **Modern Vulkan.** Target Vulkan 1.3: dynamic rendering, `synchronization2`,
  timeline semaphores, VMA for allocation, Slang for shaders + reflection.
- **Concise construction.** A builder/manager layer turns the verbose Vulkan
  create-info dance into reflection-driven builders — "vk-bootstrap for the
  rendering construction logic."

### Non-goals / deferred to v2

- **No UI / immediate-mode overlay nodes.** Pure 3D. (An always-dirty UI leaf next
  to the sink would force the final composite every frame.)
- **Temporal / history nodes deferred.** No TAA history, ping-pong, or
  accumulation in v1. The DAG is acyclic with no across-time edges for now.
- **Automatic barrier inference deferred.** v1 GPU nodes emit their own barriers
  via `ExecContext` helpers, querying the persistent resource-state tracker.
  Declared-access + inferred barriers is a v2 concern.
- **Transient memory aliasing deferred** (lifetime-interval binning); v1 uses
  straightforward persistent/transient allocation without aliasing.
- No multi-GPU, no bindless descriptors v1, no shader hot-reload v1.

---

## 2. Core Mental Model

### 2.1 Two traversals

Two opposite directions through the graph:

- **Invalidation is upstream → downstream.** Marking a node dirty concerns its
  *consumer* cone. We do **not** eagerly cascade (no recursive walk on mutation);
  marking is O(1) and the propagation happens once, in order, during the sweep.
- **Demand is downstream → upstream.** The **sinks** (the present node, plus any
  readbacks) define the *demanded cone* via reverse edges — the bound within which
  we bother to propagate dirtiness and execute.

So a frame's work is precisely:

> **(demanded cone) ∩ (dirty), executed in topological order.** A node runs only
> if it is both reachable from a sink *and* dirty. A dirty subtree feeding nothing
> presented costs nothing; a demanded subtree that is clean costs nothing.

### 2.2 Invalidation-driven dirtiness + forward topological sweep

Each value slot carries **two per-frame booleans** (not a revision counter):

- **`m_dirty`** — *candidate*: "this may need recomputing this frame." Seeded by
  `invalidate()` (a source mutation, a swapchain acquire, or an imperative force),
  and propagated downstream by the sweep.
- **`m_changed`** — *actual*: "the value this produced this frame differs from last
  frame." Set when the producing node executes; drives cutoff (§2.4).

The **resolve** pass is a **forward sweep in topological order** over the demanded
cone. Because topology is stable across frames (a persistent graph), the
topological order and the demanded set are cached and recomputed only on a graph
edit (§2.5).

```
resolve(sinks) -> FramePlan:
    demanded = nodes reachable from `sinks` via reverse (dependency) edges   # cached
    plan = []
    for node in topo_order if node in demanded:        # sources-first
        if any(input.m_dirty for input in node.inputs()):
            for out in node.outputs(): out.m_dirty = true   # propagate forward
            plan.push_back(node)
    return FramePlan{ .order = plan }                  # dependency-ordered
```

**Why a forward sweep, and why it is correct under async execution.** The sweep
propagates the *plan-time* flag `m_dirty`, never an execution-time result. Since
we visit producers before consumers (topological order), every `input.m_dirty` is
final when read. This is exactly where a recursive, revision-comparing verify
(Salsa-style) breaks for deferred GPU execution: it would compare a dependency's
"changed-at" stamp that has **not been updated yet** because the dependency has
not executed. Dirty flags sidestep that entirely — the plan is decided before any
work runs.

Cycle detection lives in the `topo_order` computation (a back edge → `GraphError`),
done once per topology edit, not per frame.

### 2.3 The second axis: execution state

The two dirty/changed booleans answer *"is this in the plan, and did it move?"*.
They do **not** track the asynchronous lifecycle of the (re)computation, which runs
on a job pool (CPU) or a queue (GPU). That is a separate, orthogonal axis:

```cpp
enum class ExecutionState : std::uint8_t {
    IDLE,        // not scheduled this frame
    PROCESSING,  // execute() running / GPU commands in flight
    DONE,        // execute() returned this frame
    FAILED,      // execute() errored (see §9)
};
```

For the executor, **"done" = `execute()` returned.** For a CPU node that means the
work finished; for a GPU node it means its commands were *recorded and submitted*.
Cross-node GPU *ordering* is carried by timeline semaphores at submission (§5), not
by the CPU executor — so the executor stays uniform and Vulkan-agnostic.

### 2.4 Change cutoff

`Data<T>::store` compares the new value to the old and reports whether it changed:

```cpp
template <class T>
bool Data<T>::store(T next) {
    if constexpr (std::equality_comparable<T>) {
        const bool changed = !(next == m_value);
        m_value = std::move(next);
        m_changed = changed;     // cutoff: unchanged value stops downstream
        return changed;
    } else {
        m_value = std::move(next);
        m_changed = true;        // no cheap equality → always "changed"
        return true;
    }
}
```

Cutoff is enforced **at execution time** by the executor: a planned node checks
whether *any of its inputs actually changed this frame* before running (§2.6). So
even though the conservative sweep may include a node, it is skipped if a mid-graph
producer recomputed to an identical value. This is what stops sub-pixel camera
jitter or a toggled-back value from rippling through.

- **CPU value nodes** with `equality_comparable` outputs get true value-cutoff.
- **GPU nodes** report `changed = (they executed)` — their output is image
  *contents*, not a comparable value, so "changed" means "re-recorded this frame."
  Their cutoff comes entirely from their inputs.

### 2.5 Frame-boundary updates ("only update the graph between frames")

All graph mutation is **queued and applied at the start of a frame**, then the
frame index is bumped:

- **Source value updates** apply (equality-gated: a no-op write sets neither flag).
- **Topology edits** (add/remove nodes, rewire) apply, invalidating the cached
  `topo_order` / demanded set.
- Per-frame `m_dirty` / `m_changed` from the previous frame are reset; newly
  applied invalidations seed the new frame.

Within a frame the graph is immutable. This gives a consistent snapshot to the
executor, matches Vulkan reality (you cannot mutate a buffer the GPU is mid-read
on), and coalesces bursty inputs (a drag-resize collapses to one extent change).

```cpp
using FrameIndex = std::uint64_t;   // monotonic; frame_slot = FrameIndex % FRAMES_IN_FLIGHT
```

### 2.6 Frame lifecycle: resolve (sync) then execute (async dataflow)

1. **Apply** queued mutations / topology edits; bump `FrameIndex`; reset per-frame
   flags; re-seed invalidations.
2. **Resolve** (sync, main thread): the forward sweep (§2.2) → `FramePlan`.
3. **Execute** (async dataflow): run the plan respecting dependencies (below).
4. **Present**: once the sink(s) are done, the present node has blitted the
   (possibly cached) scene image into the acquired swapchain image; present.

**The executor is a dependency-counter dataflow engine**, pure and Vulkan-agnostic:

```
execute(plan, scheduler):
    for node in plan.order:
        node.pending = count(inputs whose producer is in plan.order)
    ready = { node in plan.order : node.pending == 0 }
    dispatch each ready node to `scheduler`:
        if any(input.m_changed for input in node.inputs()):
            node.execute(ctx)          # recompute; outputs set m_changed via store()
        else:
            for out in node.outputs(): out.m_changed = false   # false positive → skip
        node.state = DONE
        for consumer in node.consumers ∩ plan.order:
            if --consumer.pending == 0: ready.push(consumer)
    frame is complete when all sink nodes are DONE
```

This is "execution driven by invalidation": only nodes seeded dirty (directly, or
via the sweep from a dirty input) are ever scheduled. A frame whose only dirty seed
is the swapchain acquire runs just the present blit; a dirty scene node is what
triggers a re-render. GPU ordering inside the run is handled by §5, not here.

> **v1 execution note.** CPU transform nodes run in parallel on the job pool
> (dependency-ordered as above). GPU nodes record into one primary command buffer
> per queue **in plan order** on a single recording thread (one submit per queue,
> timeline semaphores between queues). Multi-threaded recording with secondary
> buffers is a v2 optimization.

### 2.7 CPU vs GPU completion

- **CPU nodes** signal completion via the atomic `ExecutionState` (`PROCESSING` →
  `DONE`) with `wait`/`notify`; the executor's pending-counter decrement is driven
  off that.
- **GPU nodes** carry a **timeline semaphore** value. A GPU `Data`'s contents are
  "ready" when `vkGetSemaphoreCounterValue(sem) >= ready_value`. The timeline is a
  monotonic per-queue **completion counter**; the CPU can wait, query, and signal
  it, and it subsumes fences for frame pacing.

This is why `GPUNode` / `GPUSync` already use timeline semaphores — that choice is
correct and load-bearing (§5).

---

## 3. Architecture & Layering

```
L0  Vulkan Foundation     vk-bootstrap (Context), VMA, Slang Shader + reflection, sync primitives
L1  Resource Management   typed handles, transient + persistent pools, per-frame rings, barrier/state tracker
L2  Construction Layer    pipeline builders, descriptor allocator/writer, swapchain manager, command + sync managers
L3  Reactive Core         Data<T>/Node, dirty sweep, dataflow executor, scheduler interface, cycle detection   [Vulkan-agnostic]
L4  Node Types            sources (incl. ScreenSize), CPU transform, GPU compute/raster, bridges, swapchain/present
L5  Frame Orchestration   driver/clock, frames-in-flight, job system, present cadence, device-loss recovery
L6  Authoring & Tooling   typed builder API, GraphViz dump, serialization, validation layers
```

**Dependency directions** (critical):

- **L3 has no Vulkan dependency.** It depends only on an abstract `Scheduler`, the
  `Node`/`DataNode` bases, and `ExecContext`. This is what makes the sweep +
  executor unit-testable without a GPU.
- **L2 and L3 are independent** of each other.
- **L4 is the bridge**: concrete node types use L1/L2 to do Vulkan work and derive
  from L3's `Node`. They receive a Vulkan-aware `ExecContext` (§4 L4).
- L5 drives L3 and owns L2's swapchain/command/sync managers.

---

## 4. Layer Specifications

### L0 — Vulkan Foundation *(largely exists)*

- `veng::Context` (exists): instance, physical/logical device, queue indices
  (graphics + dedicated compute), VMA allocator. Built via vk-bootstrap.
- `veng::Shader` (exists): Slang module + reflection (`DescriptorInfo`,
  `PushConstantInfo`, `VertexAttribute`/`VertexBinding`, per-stage `*Details`),
  feeding L2's builders.
- Require Vulkan 1.3 (dynamic rendering, `synchronization2`, timeline semaphores).

### L1 — Resource Management

GPU resources are referenced by **generation-tagged handles** (stable across pool
reallocation, serializable, UAF/ABA-safe against async work):

```cpp
template <class Tag>
struct Handle {
    static constexpr std::uint32_t INVALID_INDEX = ~0u;
    std::uint32_t index      = INVALID_INDEX;
    std::uint32_t generation = 0;
    [[nodiscard]] bool valid() const { return index != INVALID_INDEX; }
};
using BufferHandle = Handle<struct BufferTag>;
using ImageHandle  = Handle<struct ImageTag>;
```

> Note: handles are for **GPU resources** (L1) — values that outlive a single node
> and are touched by async GPU work. Graph **edges** between nodes use typed
> pointers (§4 L3), which is ergonomic with templates and safe because the graph
> owns all nodes/data in address-stable storage and drains in-flight work on
> teardown.

Responsibilities:

- **Typed buffer/image wrappers** over VMA allocations (RAII, move-only).
- **Two allocation pools**: a **persistent** pool for cached subgraph outputs that
  must survive across frames (a clean node's image cannot be recycled), and a
  **transient** pool for the dirty frontier. (Aliasing within the transient pool is
  v2.)
- **Per-frame ring buffering** for resources written each frame (uniforms,
  descriptor sets, command buffers). A clean scene target is read (blitted) from a
  stable slot; it rotates into a fresh slot only when re-rendered (§6).
- **Cross-frame resource-state tracker**: per resource, current layout +
  last-writer stage/access + last-signaled timeline value. Skipped nodes don't
  re-record, so barrier decisions must read persisted state, not a single frame's
  command list.

### L2 — Construction Layer (Builders & Managers)

> "vk-bootstrap for the rendering construction logic." Keeps L4 node
> implementations free of create-info boilerplate; reflection-driven, RAII,
> rebuildable.

#### L2.1 Pipeline builders

Descriptor-set layouts and push-constant ranges are generated **from reflection**
(`DescriptorInfo`/`PushConstantInfo`), closing the `// TODO` block in
`src/pipelines/ComputePipeline.cpp`.

```cpp
class GraphicsPipelineBuilder {
public:
    GraphicsPipelineBuilder& stages(const Shader& vertex, const Shader& fragment);
    GraphicsPipelineBuilder& tessellation(const Shader& control, const Shader& eval);
    GraphicsPipelineBuilder& geometry(const Shader& geom);
    GraphicsPipelineBuilder& vertex_input(std::span<const VertexBinding>, std::span<const VertexAttribute>);
    GraphicsPipelineBuilder& rasterization(/* cull, polygon mode, ... */);
    GraphicsPipelineBuilder& depth_stencil(/* test, write, compare ... */);
    GraphicsPipelineBuilder& blend(std::span<const vk::PipelineColorBlendAttachmentState>);
    GraphicsPipelineBuilder& color_formats(std::span<const vk::Format>);  // dynamic rendering
    GraphicsPipelineBuilder& depth_format(vk::Format);
    [[nodiscard]] std::expected<GraphicsPipeline, PipelineError> build(const Context&, vk::PipelineCache = {}) const;
};

class ComputePipelineBuilder {
public:
    explicit ComputePipelineBuilder(const Shader& compute);
    [[nodiscard]] std::expected<ComputePipeline, PipelineError> build(const Context&, vk::PipelineCache = {}) const;
};
```

- Stage compatibility validated via existing `ShaderDetails::matches`.
- Shared `vk::PipelineCache` owned here and reused across rebuilds.

#### L2.2 Descriptor management

- **v1**: `DescriptorAllocator` (grows by adding pools on exhaustion) +
  `DescriptorWriter` (builds `vk::WriteDescriptorSet`s from handles).
- **v2**: bindless `UPDATE_AFTER_BIND` indexing.

```cpp
class DescriptorAllocator {
public:
    [[nodiscard]] vk::DescriptorSet allocate(vk::DescriptorSetLayout);
    void reset_frame(std::size_t frame_slot);
};
```

#### L2.3 Swapchain manager

Wraps vk-bootstrap's swapchain builder; owns the present-surface lifecycle.

```cpp
struct AcquiredImage {
    std::uint32_t index;
    ImageHandle   image;         // imported transient resource (rotating identity)
    vk::Semaphore acquire_ready; // binary; blit submission waits on it
};

class SwapchainManager {
public:
    static std::expected<SwapchainManager, SwapchainError> create(const Context&, /* surface, */ vk::Extent2D);
    [[nodiscard]] std::expected<AcquiredImage, vk::Result> acquire(std::size_t frame_slot);
    [[nodiscard]] vk::Result present(std::size_t frame_slot, vk::Semaphore wait);
    void rebuild(vk::Extent2D new_extent);     // resize / OUT_OF_DATE
    [[nodiscard]] vk::Extent2D extent() const; // feeds the ScreenSize source node
};
```

- Owns the **binary** acquire/render-finished semaphores per frame slot (swapchain
  present is binary-only in core Vulkan).
- `rebuild()` updates `extent()`; the driver pushes the new extent into the
  **`ScreenSize` source node** (L4), turning resize into a normal invalidation.

#### L2.4 Command & sync managers

```cpp
class CommandManager {
public:
    [[nodiscard]] vk::CommandBuffer begin(QueueKind, std::size_t frame_slot);  // v1: one per queue/slot
    [[nodiscard]] vk::CommandBuffer one_shot(QueueKind);                       // transient uploads
    void image_barrier(vk::CommandBuffer, ImageHandle, /* src/dst stage+access, layout */);
};

class SyncManager {
public:
    [[nodiscard]] std::uint64_t next_value(QueueKind);    // monotonic completion counter
    [[nodiscard]] vk::Semaphore timeline(QueueKind) const;
    // Binary semaphore pool for swapchain handoff.
};
```

### L3 — Reactive Core *(Vulkan-agnostic; the heart; fully unit-tested)*

Lifts the prototype in `playground/main.cpp` into the library, fixing its races and
switching to the dirty-flag/sweep model. The value type is templated; the **only**
runtime polymorphism is the `Node`/`DataNode` bases the scheduler needs.

```cpp
namespace veng::graph {

class Node;

// Topology base: everything the sweep/executor needs without knowing T.
class DataNode {
public:
    virtual ~DataNode() = default;
    [[nodiscard]] Node*                  producer()  const { return m_producer; }   // null => source
    [[nodiscard]] std::span<Node* const> consumers() const { return m_consumers; }
    [[nodiscard]] bool dirty()   const { return m_dirty; }
    [[nodiscard]] bool changed() const { return m_changed; }
    void invalidate() { m_dirty = true; }   // O(1) seed; sweep does the propagation
protected:
    Node*              m_producer = nullptr;
    std::vector<Node*> m_consumers;
    bool               m_dirty    = true;
    bool               m_changed  = false;
    friend class Graph;
};

// Typed value slot — no erasure; consumers read m_value as `const T&`.
template <class T>
class Data final : public DataNode {
public:
    [[nodiscard]] const T& value() const { return m_value; }
    bool store(T next);            // §2.4: sets m_changed, returns whether it changed
private:
    T m_value{};
};

// Per-frame execution environment (generic; GPU specialization in L4).
class ExecContext {
public:
    virtual ~ExecContext() = default;
    [[nodiscard]] virtual std::size_t  frame_slot()  const = 0;  // 0..FRAMES_IN_FLIGHT-1
    [[nodiscard]] virtual FrameIndex   frame_index() const = 0;  // monotonic
};

class Node {
public:
    virtual ~Node() = default;
    [[nodiscard]] virtual std::span<DataNode* const> inputs()  const = 0;
    [[nodiscard]] virtual std::span<DataNode* const> outputs() const = 0;
    // Pure function of resolved inputs. Writes outputs via Data<T>::store.
    [[nodiscard]] virtual std::expected<void, ExecError> execute(ExecContext&) = 0;
protected:
    std::atomic<ExecutionState> m_state = ExecutionState::IDLE;
    std::atomic<std::uint32_t>  m_pending = 0;   // executor dependency counter
    friend class Graph;
};

// Injected: real impl is the job system; tests inject a deterministic runner.
class Scheduler {
public:
    virtual ~Scheduler() = default;
    virtual void submit(std::function<void()> task) = 0;
};

struct FramePlan {
    std::vector<Node*> order;   // dirty ∩ demanded, dependency (topological) order
    // v2: per-queue batch boundaries for command-buffer grouping.
};

enum class GraphError : std::uint8_t { CYCLE_DETECTED, UNKNOWN_SINK };
enum class ExecError  : std::uint8_t { NODE_FAILED, DEVICE_LOST };

class Graph {
public:
    template <class N, class... Args> N* add(Args&&...);   // graph owns; returns stable ptr
    template <class T> Data<T>* add_source(T initial);

    template <class T> void update_source(Data<T>*, T next);  // queued to next frame boundary

    [[nodiscard]] std::expected<FramePlan, GraphError> resolve(std::span<DataNode* const> sinks);  // §2.2
    void execute(const FramePlan&, Scheduler&, ExecContext&);                                       // §2.6

    [[nodiscard]] std::string to_dot() const;   // L6 tooling
};

} // namespace veng::graph
```

Core guarantees:

- **Correct under async.** The forward sweep decides the plan from plan-time
  `m_dirty` flags only (§2.2); no execution-time result is read during planning.
- **Diamonds are safe.** Dependency counters + topological order resolve a shared
  dependency once before either consumer dispatches.
- **Cutoff propagates** at execution via the `m_changed` check (§2.4, §2.6).
- **Cycle detection** when (re)building the cached topological order.
- **No Vulkan.** `ExecContext`, `Scheduler`, and the sweep/executor are pure.

### L4 — Node Types

Concrete nodes are typed (e.g. a transform node templated on `Ret(Args...)`, the
prototype's `CPUTransformer`). GPU nodes derive from a `GpuNode` base that localizes
the context downcast:

```cpp
class GpuExecContext : public graph::ExecContext {  // constructed by the L5 executor
public:
    [[nodiscard]] virtual vk::CommandBuffer cmd()        = 0;   // this queue's CB for the frame
    [[nodiscard]] virtual ResourceManager&  resources()  = 0;
    [[nodiscard]] virtual SyncManager&      sync()        = 0;
    [[nodiscard]] virtual QueueKind         queue()      const = 0;
};

class GpuNode : public graph::Node {
protected:
    virtual std::expected<void, graph::ExecError> execute_gpu(GpuExecContext&) = 0;
    std::expected<void, graph::ExecError> execute(graph::ExecContext& c) final {
        return execute_gpu(static_cast<GpuExecContext&>(c));   // safe: GPU executor supplies GpuExecContext
    }
};
```

Node categories, annotated by **dirty frequency** (the design lens — payoff is the
gap between *rarely* and *always*):

#### Sources (reactive roots; externally updated; equality-gated)

- **`ScreenSize` / `Resolution`** *(rarely dirty — only on resize)*. Holds
  `vk::Extent2D`. **Most attachments are sized from it** (optionally via a
  render-scale factor), so it is a first-class input: render-target nodes depend on
  it. Fed by `SwapchainManager::extent()` on rebuild → a resize becomes an ordinary
  cascade (extent changes → sized attachments reallocate → scene re-renders →
  present blits). Equality-gated and coalesced.
- **Camera**, **time**, **light params** *(camera-dirty / animated)*; **per-object
  transforms** *(per-object — keep granularity fine)*; **scene set**, **assets**.

#### CPU transform *(job pool; pure functions)*

- **Generic transform node** (`CPUTransformer<Ret(Args...)>`) — the workhorse.
- **Culling**, scene-graph flatten, LOD select, depth/material sort, light
  clustering.

#### Bridge (CPU ↔ GPU)

- **Upload/staging** (CPU buffer → GPU). **Readback** (GPU → CPU, *latent by K
  frames*; never a stall) — note this is the one place a CPU consumer must wait on
  GPU completion; modeled as a deferred-result node.

#### GPU compute *(dirty only when params/inputs change)*

- Generic dispatch (`ComputePipeline`), mip generation, reductions/histogram, GPU
  culling → indirect-draw build.

#### GPU raster *(the meat)*

- G-buffer/geometry, depth prepass, **shadow maps** *(one node per light; static
  light over static geometry never re-renders)*, forward/opaque, transparency,
  **sky/atmosphere** *(sun + time only)*, post chain (SSAO/bloom/SSR/DoF/tonemap).

#### Swapchain & present (sink + external import + frame clock)

Modeled as a **per-frame source** plus a **present/blit node**, with a strict
decoupling rule:

- **Persistent scene color target** (graph-owned, cacheable, sized from
  `ScreenSize`) is what the 3D graph renders into — the cacheable reactive output.
- **Swapchain source**: `invalidate()`d on every `acquire()`. Its value is the
  rotating acquired image + binary acquire semaphore (external rotating identity —
  "not quite an image," hence not a normal cacheable `Data`).
- **Present/blit node depends on *both*** scene color *and* the swapchain source.
  Only the swapchain source re-dirties each frame, so **only the blit re-runs**:
  scene color → acquired image, transition to `PRESENT_SRC`, present.

> **Load-bearing rule:** the scene subgraph must **not** depend on the swapchain
> source. If acquire-dirtiness leaks upstream, you re-render every frame. *Present
> depends on swapchain; scene does not.* The swapchain is the clock for the present
> tail only — and "rendering is a natural consequence" of either the swapchain
> source (→ blit) or a scene-feeding source (→ re-render) being marked dirty.

> *Deferred (v2):* temporal/history nodes (TAA, ping-pong, accumulation). These
> read the previous frame's output and require a delay edge that intentionally
> breaks the DAG across time — out of scope for v1.

### L5 — Frame Orchestration & Driver

- **The driver is the clock.** Each iteration: apply queued updates → bump
  `FrameIndex` → `resolve(sinks)` → `execute(plan)` → present. The present node is
  the root sink; the acquire/present cadence paces it.
- **Frames in flight** (`FRAMES_IN_FLIGHT` = 2–3): per-frame ring slots for command
  buffers, uniforms, descriptor sets, binary swapchain semaphores. Clean scene
  reads share a stable slot; re-renders rotate into a fresh slot (§6).
- **Job system**: a work-stealing pool implementing `graph::Scheduler`. GPU nodes
  must not block a CPU thread on a semaphore — poll the timeline or suspend a
  coroutine. v1 records GPU commands single-threaded in plan order (§2.6 note).
- **Present cadence**: *conservative (default)* — present every frame, blit the
  cached scene (robust, FIFO-friendly, the blit is the only guaranteed per-frame
  GPU). *Aggressive (opt-in, v2)* — skip acquire+present when nothing is dirty;
  needs OS expose/damage wake-ups.
- **Device-loss recovery** (§9).

### L6 — Authoring & Tooling

- **Typed builder API**: factory helpers in the spirit of the prototype's
  `CPUTransformer::create(func, inputs...)` — compile-time-typed edges (the
  `Data<T>*` inputs are checked against the functor signature), graph owns the
  result.
- **`to_dot()`**: GraphViz dump colored by dirty/changed/state and dirty frequency.
  Build it *early*.
- **Serialization** of topology; validation layers in debug.

---

## 5. Synchronization Strategy

Timeline semaphores and barriers are **not alternatives** — different scopes, both
required.

- **Barriers** (`vkCmdPipelineBarrier2`) — *intra-queue, intra-command-buffer*. The
  only mechanism for (1) fine-grained stage/access scoping, (2) cache
  availability/visibility, (3) **image layout transitions**.
- **Timeline semaphores** — *inter-submit / inter-queue / CPU↔GPU*. A signal→wait
  pair carries a memory dependency, but you **still** need a barrier for layout
  transitions and a barrier *pair* for queue-family ownership transfer on
  `EXCLUSIVE` resources.

**The lever is batching granularity.** Partition the plan into per-queue batches;
linearize each into one command buffer (v1: one per queue); **barriers between
adjacent nodes inside a batch**, **timeline semaphores at batch/queue boundaries**.

Reactive-specific rules:

- **Consumers wait on the producer's *recorded* `ready_value`, never a freshly
  allocated one.** A cached/skipped producer signals nothing this frame; the
  monotonic timeline means waiting on its last-recorded value passes immediately —
  cross-frame caching falls out for free. Store `ready_value` on the GPU `Data`.
- **One timeline per queue**; **swapchain handoff is binary** (the one exception).
- v1 barriers are emitted manually by GPU nodes via `GpuExecContext`/`CommandManager`
  helpers, querying the persistent state tracker (L1); automatic inference is v2.

---

## 6. Resource Lifetime: Caching vs Frames-in-Flight

- **Caching vs recycling.** A clean node's output must stay resident (persistent
  pool) — it cannot be recycled like a transient.
- **Cross-frame state.** Barrier/layout decisions read the *persistent* per-resource
  state tracker, because a skipped node didn't re-record its transitions.
- **Frames-in-flight + caching.** A clean scene image is only *read* (blitted) while
  cached → one stable slot is safe across in-flight frames. The hazard appears only
  when it goes dirty and re-renders: render into a *fresh ring slot* so an in-flight
  blit of the previous slot is not stomped.
- **Eviction.** Under memory pressure, drop a long-cached image and `invalidate()`
  its producer; it recomputes on next demand. The dirty model supports this for
  free.

---

## 7. Key Design Decisions

| Decision | Choice | Rationale | Alternative |
|---|---|---|---|
| Validity model | **Invalidation-driven dirty flags + forward topological sweep** | Correct under deferred/async execution (propagates plan-time flags, not execution-time stamps); matches "dirty drives rendering"; O(cone) per frame | Pull-with-revisions (Salsa) — assumes synchronous inline recompute; the revision-verify reads stale "changed-at" when execution is deferred |
| Value flow | **Templated `Data<T>` (typed access); polymorphism only on `Node`/`DataNode` bases** | No `std::any`/erasure; compile-time-checked edges; mirrors the prototype | Type-erased `Data` base — simpler container but unsafe value access |
| Cutoff | **`m_changed` flag set by `store()`; checked at execute** | Within-frame cutoff atop a conservative sweep; sub-pixel/no-op changes stop propagating | None / revision backdating |
| Graph edges | **Typed `Data<T>*` / `Node*`; graph owns address-stable storage, drains on teardown** | Ergonomic with templates; stability + drain cover the UAF concern | Generation handles everywhere — safer but erases types; kept for GPU resources (L1) and tooling |
| Executor | **Dependency-counter dataflow; "done" = `execute()` returned** | Uniform for CPU + GPU; GPU ordering carried by semaphores, keeping L3 Vulkan-free | Streaming propagation — natural cutoff but poor GPU batching |
| `ExecContext` | **Generic L3 base; `GpuExecContext` adds Vulkan; cast localized in `GpuNode`** | Keeps L3 Vulkan-agnostic; one safe downcast site | Vulkan in the base (pollutes L3) / templating Node on context (breaks the container) |
| Input timing | **Frame-boundary snapshots** | Immutable per-frame graph; matches GPU reality; free debounce | Mid-flight mutation — racy |
| Swapchain | **Per-frame source + present/blit; scene target persistent & decoupled** | Only the blit re-runs; scene stays cached | Render into swapchain images with per-image validity — flicker, no saving when static |
| Sync | **Timeline between batches/queues; barriers within; manual emission v1** | Correct scopes; minimal submissions | Auto barrier inference — deferred to v2 |

---

## 8. Edge Cases & Hazards (checklist)

- [x] **Async planning staleness** — *resolved by design:* the forward sweep uses
      plan-time `m_dirty`, never execution-time results (§2.2).
- [ ] **Completion race**: a re-invalidation during `execute()`. *Mitigation:*
      frame-boundary snapshots (§2.5) make mid-flight invalidation impossible; the
      `PROCESSING → DONE` transition is the anchor.
- [ ] **Per-swapchain-image staleness / flicker**: persistent scene target +
      per-frame blit (L4).
- [ ] **Caching vs recycling**: persistent/transient pool split (§6).
- [ ] **Cross-frame barrier state**: persistent state tracker (L1).
- [ ] **Cycles**: rejected when building the cached topo order (§2.2). (No temporal
      across-time edges in v1.)
- [ ] **Granularity trap**: per-object/region nodes + equality-gated sources.
- [ ] **Timeline deadlock**: wait on recorded `ready_value` (§5).
- [ ] **Resize cascade**: `ScreenSize` source → normal invalidation; coalesce.
- [ ] **Latent readback**: never stall; surface results K frames later.
- [ ] **Purity**: nodes are pure functions of (inputs, resources); model all state
      as data nodes.

---

## 9. Error Handling & Device Loss

- `Node::execute` returns `std::expected<void, ExecError>`; failure → `FAILED`,
  policy is propagate-to-dependents or hold-last-good per node type.
- `VK_ERROR_DEVICE_LOST`: drain in-flight work, tear down device resources, rebuild
  `Context` + GPU resources, `invalidate()` all sources, re-resolve. Handle
  indirection (L1) makes "rebuild backing resource, keep handle" feasible.
- Fallible construction returns `std::expected` via `common.hpp`'s
  `merge_expected_t` / `ResultVariant`.

---

## 10. Testing Strategy

- **The L3 core is Vulkan-free and is the coverage priority** (95% target). Test
  with fake `Node`/`Data<T>` and a **deterministic single-threaded `Scheduler`**.
- Cases: the forward sweep over diamonds and deep cones; cutoff (a planned node
  whose input recomputed to an equal value is skipped); cycle rejection; dependency
  counters / ordering; lazy demand (a dirty-but-undemanded subtree is absent from
  the plan); frame-boundary snapshot semantics.
- L2 builders tested against reflection fixtures (layout generation needs no
  device; pipeline `build` uses a headless device). Integration on a headless
  device per `tests/context` + `tests/shader`.

---

## 11. Implementation Roadmap

1. **L3 reactive core first** — `Data<T>`/`DataNode`/`Node`, the forward sweep,
   the dependency-counter executor, `Scheduler`, cycle detection, `to_dot()`. Lift
   `CPUTransformer` from `playground/`. Heavy unit tests with a fake scheduler.
   *(The novel, riskiest part; building it first de-risks everything.)*
2. **L1 + L2** — finish `ComputePipeline` (descriptors + return value),
   `DescriptorAllocator`/`Writer`, `GraphicsPipeline` + builder, buffer/image
   wrappers + handles, then `SwapchainManager`/`CommandManager`/`SyncManager`.
3. **L4 vertical slice** — `ScreenSize` source → one GPU raster node → persistent
   scene target → swapchain/present node. Prove "static scene presents without
   re-rendering" end to end.
4. **L5 driver + job system + frames-in-flight** — real `Scheduler`, conservative
   present cadence, device-loss recovery.
5. **Breadth in L4** — shadow maps, compute, post chain.
6. **v2** — temporal/history nodes, automatic barrier inference, transient
   aliasing, bindless, aggressive present, multi-threaded recording.

---

## 12. Prior Art & References

- **Granite render graph** — Hans-Kristian Arntzen (Themaister): the reference for
  Vulkan barrier tracking + async compute automation.
- **Frostbite FrameGraph** — Yuriy O'Donnell, GDC 2017 (the *transient* baseline we
  deviate from).
- **Jane Street Incremental** (OCaml) — diamond-correctness and height-ordered
  scheduling.
- **SolidJS / MobX signals** — the push/pull reactive lineage (we use the
  push-dirty variant).
- **Vulkan spec, Synchronization chapter**; timeline semaphore (core 1.2),
  `synchronization2` (core 1.3); **VMA**; **vk-bootstrap**; **Slang** reflection.
