// Test vertex shader for the instanced path: a single shared triangle (in a vertex buffer)
// is drawn `instanceCount` times; each instance reads its own offset from a `StructuredBuffer
// <Body>` indexed by SV_InstanceID. With per-vertex color = white the rendered pixel is the
// per-instance offset visualized as the triangle's position, so a readback can identify
// instance N. Pairs with tinted.frag (the existing white-times-tint test fragment shader).
struct VSIn
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 color : COLOR0;
};

struct VSOut
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 color : COLOR0;
};

struct Body
{
    float4 offset; // xy translation in NDC, zw unused
};
[[vk::binding(1, 0)]] StructuredBuffer<Body> bodies;

[shader("vertex")]
VSOut main(VSIn input, uint instanceID : SV_InstanceID)
{
    VSOut output;
    Body  body = bodies[instanceID];
    output.position = float4(input.position.xy + body.offset.xy, input.position.z, 1.0);
    output.color = input.color;
    return output;
}
