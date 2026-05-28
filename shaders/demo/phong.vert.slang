// Vertex shader for the multi-mesh / shared-materials Phong example.
//
// Resources:
//   * transforms_sphere / transforms_cube / transforms_torus  : per-instance Transform
//     arrays, one per mesh kind. The draw call that binds the sphere mesh reads from
//     transforms_sphere, etc. A per-draw `mesh_kind` push constant picks which one.
//   * (materials, light, view — fragment-side, see phong.frag.slang)
//
// Why three buffers instead of one big sorted array? In this engine descriptor sets are
// node-level (shared across the node's draws); to give each draw a distinct slice without
// each draw also needing its own pipeline, we publish one SSBO per mesh kind and let the
// per-draw push constant pick. Each draw's `instanceCount` comes from its matching SSBO's
// published `BufferRef::count` (`set_instances_from(transforms_<kind>_ref)`), so empty
// arrays draw zero instances — the buildup phase that has only spheres so far is one line
// in the sim thread, no special-casing here.
//
// Each Transform carries a `material_index` into the shared materials[] array; the frag
// re-reads it (with nointerpolation, since integers can't be interpolated).
struct VSIn
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal   : NORMAL;
    [[vk::location(2)]] float3 color    : COLOR0; // unused (material colors override)
};

struct VSOut
{
    float4                                       position       : SV_Position;
    [[vk::location(0)]] nointerpolation uint     material_index : COLOR0;
    [[vk::location(1)]]                 float3   world_position : POSITION0;
    [[vk::location(2)]]                 float3   world_normal   : NORMAL0;
};

struct PushData
{
    float4x4 view_proj; // offset 0  (64 bytes) — shared across draws (camera-driven)
    uint     mesh_kind; // offset 64 (4 bytes)  — per-draw: 0=sphere, 1=cube, 2=torus
};
[[vk::push_constant]] PushData pc;

struct Transform
{
    float4x4 model;
    float4x4 normal_mat;
    uint     material_index;
    // Three scalar uints — NOT uint3. In std430, uint3 (uvec3) has alignment 16, which
    // would push the struct's stride from 144 to 160 and silently misalign every element
    // past index 0. Scalar uints have alignment 4, so material_index + three scalar pads
    // pack tightly into the trailing 16 bytes and the stride matches the C++ side's 144.
    uint     _pad0;
    uint     _pad1;
    uint     _pad2;
};
[[vk::binding(0, 0)]] StructuredBuffer<Transform> transforms_sphere;
[[vk::binding(1, 0)]] StructuredBuffer<Transform> transforms_cube;
[[vk::binding(2, 0)]] StructuredBuffer<Transform> transforms_torus;

[shader("vertex")]
VSOut main(VSIn input, uint instance_id : SV_InstanceID)
{
    Transform t;
    if (pc.mesh_kind == 0)      t = transforms_sphere[instance_id];
    else if (pc.mesh_kind == 1) t = transforms_cube[instance_id];
    else                        t = transforms_torus[instance_id];

    VSOut  output;
    float4 world_pos      = mul(float4(input.position, 1.0), t.model);
    output.world_position = world_pos.xyz;
    output.position       = mul(world_pos, pc.view_proj);
    output.world_normal   = mul(float4(input.normal, 0.0), t.normal_mat).xyz;
    output.material_index = t.material_index;
    return output;
}
