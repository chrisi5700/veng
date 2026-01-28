# VENG

What am I doing here? I'm trying to build a sort of frame graph that additionally has reactive dataflow with memoization.

# Reactive Render Graph Design Document

## Overview

A demand-driven, incremental computation system for Vulkan rendering pipelines that only regenerates data when invalidated. Built on a bipartite DAG structure alternating between compute nodes and data edges.

**Key Properties:**
- Pull-based evaluation (demand-driven)
- Automatic invalidation propagation
- Cache-aware execution (no redundant work)
- Safe ownership semantics via `shared_ptr` chains

## Core Concepts

### Bipartite DAG Structure

```
Node → Data → Node → Data → Node
 ○  ────▢────  ○  ────▢────  ○
```

- **Nodes** (○): Computation units (compute shaders, graphics pipelines, allocators)
- **Data** (▢): Resources between nodes (buffers, images)
- Strict alternation creates clear data flow

### Ownership Model

```cpp
struct Data {
    std::shared_ptr<Node> parent;      // Who produces this data
    std::vector<Node*> next;           // Who consumes this data
    bool valid;
    // + buffer handle, version, etc.
};

struct Node {
    std::vector<std::shared_ptr<Data>> in_bound;   // Inputs (owned)
    std::vector<Data*> out_bound;                  // Outputs (borrowed)
    Fence done_fence;
};
```

**Invariant:** *"If you can call a method on a Node/Data, all its raw pointers are valid"*

**Why it's safe:**
- Data owns its producer (parent)
- Nodes own their inputs (in_bound)
- Creates dependency chain: leaf outputs keep entire upstream alive
- Raw pointers (`next`, `out_bound`) can never dangle because:
    - Objects holding them are kept alive by the ownership chain
    - Objects pointed to cannot be destroyed while reachable

## API Design

### Node Interface

```cpp
struct Node {
    std::string name;
    
    // Connection API
    void set_input(std::string descriptor_name, std::shared_ptr<Data> data);
    std::vector<std::shared_ptr<Data>> get_output(Allocator& alloc);
    
    // Execution
    void generate();        // Demand-driven generation
    void invalidate();      // Cascade invalidation
    
protected:
    virtual void executeImpl() = 0;  // Vulkan work here
    
private:
    std::vector<Data*> out_bound;
    std::vector<std::shared_ptr<Data>> in_bound;
    Fence done_fence;
};
```

### Data Interface

```cpp
struct Data {
    std::string name;
    
    // State
    bool valid;
    uint64_t version;
    
    // Resource
    std::shared_ptr<BufferAllocation> buffer;  // Ref-counted
    
    // Lifecycle
    void request();         // Pull generation
    void invalidate();      // Cascade invalidation
    Fence& get_fence();     // Synchronization
    
private:
    std::vector<Node*> next;
    std::shared_ptr<Node> parent;
};
```

### Graph Builder

```cpp
class RenderGraph {
public:
    // Node creation
    template<typename NodeType, typename... Args>
    std::shared_ptr<NodeType> add_node(std::string name, Args&&... args);
    
    // Connection
    std::shared_ptr<Data> connect(
        std::shared_ptr<Node> producer,
        std::shared_ptr<Node> consumer,
        std::string data_name,
        BufferHandle buffer = {}
    );
    
    // Multiple consumers
    std::shared_ptr<Data> connect(
        std::shared_ptr<Node> producer,
        std::vector<std::shared_ptr<Node>> consumers,
        std::string data_name,
        BufferHandle buffer = {}
    );
    
    // Debug
    void validate();
    void print_graph();
    
private:
    std::vector<std::shared_ptr<Node>> all_nodes;
    std::vector<std::shared_ptr<Data>> all_data;
};
```

## Usage Example

### Basic IFS Renderer

```cpp
RenderGraph graph;

// Create nodes
auto allocator = graph.add_node<AllocatorNode>("Allocator");
auto compute = graph.add_node<ComputeNode>("Compute", compute_pipeline);
auto graphics = graph.add_node<GraphicsNode>("Graphics", graphics_pipeline);
auto present = graph.add_node<PresentNode>("Present", swapchain);

// Connect pipeline
auto raw_data = graph.connect(allocator, compute, "Raw Data", buffer_42);
auto particle_data = graph.connect(compute, graphics, "Particle Data", buffer_42); // Same buffer!
auto image = graph.connect(graphics, present, "Image");

// Render loop
while (running) {
    present->generate();  // Only recomputes invalid data
}

// User interaction
on_slider_change([&](int particle_count) {
    raw_data->invalidate();  // Cascade to all dependents
});

on_window_resize([&]() {
    image->invalidate();  // Only graphics+present regenerate
});
```

### Output Behavior

**Initial generation:**
```
Allocator → Compute → Graphics → Present
All stages execute (cache cold)
```

**Window resize (image invalidated):**
```
Graphics → Present
Particle data stays cached! ✅
```

**Slider change (raw_data invalidated):**
```
Allocator → Compute → Graphics → Present
Full cascade (particles need regeneration)
```

## Execution Flow

### Pull-Based Generation

```cpp
void Node::generate() {
    // 1. Check if already valid
    if (done_fence.is_signaled() && all_outputs_valid()) {
        return;  // Cache hit!
    }
    
    // 2. Request inputs (recursive)
    for (auto& data : in_bound) {
        if (!data->valid) {
            data->request();  // → parent->generate()
        }
    }
    
    // 3. Wait for inputs
    for (auto& data : in_bound) {
        data->get_fence().wait();
    }
    
    // 4. Execute
    executeImpl();  // Record/submit Vulkan commands
    
    // 5. Mark outputs valid
    for (auto* data : out_bound) {
        data->valid = true;
    }
    done_fence.set_done();
}
```

### Invalidation Propagation

```cpp
void Data::invalidate() {
    if (!valid) return;  // Already invalid
    
    valid = false;
    version++;
    
    // Cascade to consumers
    for (auto* node : next) {
        node->invalidate();
    }
}

void Node::invalidate() {
    // Cascade to outputs
    for (auto* data : out_bound) {
        data->invalidate();
    }
    done_fence.reset();
}
```

## Vulkan Integration

### VulkanNode Example

```cpp
struct ComputeNode : Node {
    VkPipeline pipeline;
    VkDescriptorSet descriptor_set;
    VkCommandBuffer cmd;
    VkQueue queue;
    
    void executeImpl() override {
        // Update descriptors from in_bound data
        for (size_t i = 0; i < in_bound.size(); ++i) {
            VkDescriptorBufferInfo info{
                .buffer = in_bound[i]->buffer->vk_buffer,
                .offset = 0,
                .range = VK_WHOLE_SIZE
            };
            // Write descriptor...
        }
        
        // Record commands
        vkBeginCommandBuffer(cmd, ...);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, ..., 1, &descriptor_set, ...);
        vkCmdDispatch(cmd, ...);
        vkEndCommandBuffer(cmd);
        
        // Submit
        VkSubmitInfo submit{
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd
        };
        vkQueueSubmit(queue, 1, &submit, done_fence.vk_fence);
    }
};
```

### Fence Wrapper

```cpp
struct Fence {
    VkDevice device;
    VkFence vk_fence;
    std::atomic<bool> signaled{false};
    
    void wait() {
        if (signaled.load()) return;
        vkWaitForFences(device, 1, &vk_fence, VK_TRUE, UINT64_MAX);
        signaled.store(true);
    }
    
    void set_done() {
        signaled.store(true);
    }
    
    void reset() {
        vkResetFences(device, 1, &vk_fence);
        signaled.store(false);
    }
};
```

### Buffer Aliasing

Multiple Data objects can share the same buffer:

```cpp
// Compute reads and writes same buffer (in-place)
auto raw_data = std::make_shared<Data>("Raw", buffer_42, version=0);
auto particle_data = std::make_shared<Data>("Particles", buffer_42, version=1);

compute->set_input("input", raw_data);
compute->out_bound.push_back(particle_data.get());
```

Logical DAG maintained despite physical buffer reuse.

## Implementation Considerations

### Thread Safety

```cpp
struct Node {
    std::mutex gen_mutex;
    std::atomic<bool> generating{false};
    
    void generate() {
        // Prevent duplicate execution
        std::unique_lock lock(gen_mutex);
        if (generating.load()) {
            lock.unlock();
            done_fence.wait();
            return;
        }
        generating.store(true);
        lock.unlock();
        
        // ... generation logic ...
        
        generating.store(false);
    }
};
```

### Partial Invalidation

Invalidate specific outputs without affecting siblings:

```cpp
void Node::invalidate_output(size_t index) {
    out_bound[index]->invalidate();
    // Other outputs stay valid
}
```

### Callbacks/Hooks

```cpp
struct Data {
    std::function<void()> on_invalidate;
    std::function<void()> on_validated;
    
    void invalidate() {
        if (on_invalidate) on_invalidate();
        // ... rest of invalidation
    }
};

// Usage: UI feedback
particle_data->on_invalidate = [&] { 
    ui->show_status("Recomputing particles..."); 
};
```

### Debug Validation

```cpp
void RenderGraph::validate() {
    for (auto& data : all_data) {
        // Check parent exists
        assert(data->parent != nullptr);
        
        // Check bidirectional links
        for (auto* consumer : data->next) {
            auto it = std::find_if(
                consumer->in_bound.begin(),
                consumer->in_bound.end(),
                [&](auto& d) { return d.get() == data.get(); }
            );
            assert(it != consumer->in_bound.end());
        }
    }
}
```

## Advantages

1. **Automatic cache management** - No manual tracking of "dirty" state
2. **No redundant work** - Only recomputes when truly needed
3. **Clear data flow** - Bipartite structure makes dependencies explicit
4. **Safe lifetimes** - Ownership chain prevents dangling pointers
5. **Flexible invalidation** - Selective or cascading as needed
6. **Extensible** - Easy to add hooks, metrics, debugging

## Comparison to Traditional Frame Graphs

| Feature | Traditional Frame Graph | This System |
|---------|------------------------|-------------|
| Regeneration | Every frame | On-demand (cached) |
| Persistent resources | Imported manually | First-class citizens |
| Invalidation | Manual tracking | Automatic propagation |
| Data flow | Implicit (resource names) | Explicit (graph edges) |
| Execution order | Topological sort once | Pull-based, dynamic |

## Future Work

- **Parallel execution** - Independent branches record commands concurrently
- **Incremental barriers** - Only insert barriers for actually-executed stages
- **Profiling integration** - Track regeneration patterns
- **Serialization** - Save/load graph structure
- **Visual debugging** - GraphViz export of current graph state
- **Smart invalidation** - Partial invalidation for large buffers
- **Resource pooling** - Automatic buffer reuse between frames

## References

- Frostbite Frame Graph (GDC 2017)
- Halide autoscheduler
- Incremental computation (adapton, salsa)
- Self-adjusting computation theory

---

 