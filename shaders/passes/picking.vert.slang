// Picking pass vertex shader for veng::passes::PickingPass. Transforms buffer-backed geometry
// by a per-object MVP push constant and forwards that object's integer id (flat) to the
// fragment shader, which encodes it into the color attachment for CPU readback.
//
// The id is carried as a `nointerpolation` varying (integers can't be interpolated) — the same
// trick the phong example uses for material_index — so the only push constants are vertex-stage
// (mvp + id) and the pipeline needs no mixed-stage push range.
//
// Vertex input mirrors the engine's standard mesh layout {position, normal, color} so the
// reflected vertex stride matches the mesh buffer; only `position` is used here.
struct VSIn
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal   : NORMAL;  // unused; declared to keep the stride right
    [[vk::location(2)]] float3 color    : COLOR0;  // unused
};

struct VSOut
{
    float4              position : SV_Position;
    [[vk::location(0)]] nointerpolation uint id : COLOR0; // flat per-object id -> fragment
};

struct PushData
{
    float4x4 mvp; // offset 0  (64 bytes)
    uint     id;  // offset 64 (4 bytes) — object id (>= 1; 0 is reserved for "no hit")
};
[[vk::push_constant]] PushData pc;

[shader("vertex")]
VSOut main(VSIn input)
{
    VSOut output;
    // glm column-major matrix arrives transposed (Slang lays the float4x4 row-major), so apply
    // it from the right to get M * v — the engine's mesh.vert convention.
    output.position = mul(float4(input.position, 1.0), pc.mvp);
    output.id       = pc.id;
    return output;
}
