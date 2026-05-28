// Vertex shader for the non-instanced lit example: per-draw model + shared view_proj come
// in as two mat4s in one push-constant block (128 bytes — the spec's guaranteed minimum).
// This is the simplest possible per-vertex lit path, no SSBO and no instanceID; one draw
// per object. Pairs with `lit_directional.frag` (which only reads normal + color, so we
// can skip the world-position passthrough that the instanced shader emits).
//
// The vertex layout matches the rest of the example library: (position, normal, color).
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

struct PushData
{
    float4x4 model;     // offset 0  (64 bytes)
    float4x4 view_proj; // offset 64 (64 bytes)
};
[[vk::push_constant]] PushData pc;

[shader("vertex")]
VSOut main(VSIn input)
{
    VSOut output;
    // Slang lays the push-constant float4x4 row-major; glm provides column-major, so apply
    // from the right (mul(v, M)) to get M * v in math terms — same convention as the rest
    // of the demos.
    float4 world_pos      = mul(float4(input.position, 1.0), pc.model);
    output.world_position = world_pos.xyz;
    output.position       = mul(world_pos, pc.view_proj);
    output.world_normal   = normalize(mul(float4(input.normal, 0.0), pc.model).xyz);
    output.color          = input.color;
    return output;
}
