// Vertex shader for the instanced + lit rigid-body viewer: each instance reads its own
// model matrix from a `StructuredBuffer<Instance>` indexed by SV_InstanceID. The view-
// projection arrives as a push constant (driven by the orbit camera). World-space position
// and normal are forwarded so the fragment shader can do Lambert against a directional
// light. The mesh vertex layout adds a normal between the existing position + color (so the
// reflected binding is `(float3 position, float3 normal, float3 color)`, tightly packed —
// the cube mesh on the CPU side must match that layout).
//
// The instance's model matrix is applied directly to the position. For normals we use the
// same upper 3x3 — valid as long as the per-instance transform has uniform scale (the
// rigid-body case). A non-uniform-scale path would multiply by the inverse-transpose; not
// needed here.
struct VSIn
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal   : NORMAL;
    [[vk::location(2)]] float3 color    : COLOR0;
};

struct VSOut
{
    float4                       position : SV_Position;
    [[vk::location(0)]] float3   world_position : POSITION0;
    [[vk::location(1)]] float3   world_normal   : NORMAL0;
    [[vk::location(2)]] float3   color          : COLOR0;
};

struct Instance
{
    float4x4 model;
};
[[vk::binding(1, 0)]] StructuredBuffer<Instance> instances;

struct PushData
{
    float4x4 view_proj;
};
[[vk::push_constant]] PushData pc;

[shader("vertex")]
VSOut main(VSIn input, uint instanceID : SV_InstanceID)
{
    VSOut output;
    Instance i = instances[instanceID];

    // Slang lays the push-constant + storage-buffer matrices row-major; glm provides them
    // column-major, so apply from the right (mul(v, M)) to get M * v in math terms.
    float4 world_pos = mul(float4(input.position, 1.0), i.model);
    output.world_position = world_pos.xyz;
    output.position       = mul(world_pos, pc.view_proj);
    output.world_normal   = normalize(mul(float4(input.normal, 0.0), i.model).xyz);
    output.color          = input.color;
    return output;
}
