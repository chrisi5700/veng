// Vertex shader for real, buffer-backed geometry: position + per-vertex color come from a
// vertex buffer (a MeshNode's VkBuffer), transformed by the MVP push constant. This is the
// "actual verts" counterpart to cube.vert (which fabricates the cube from SV_VertexID); the
// reactive cube demo uses it with a 24-vertex / 36-index cube mesh. Pairs with cube.frag.
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

struct PushData
{
    float4x4 mvp;
};
[[vk::push_constant]] PushData pc;

[shader("vertex")]
VSOut main(VSIn input)
{
    VSOut output;
    // Same convention as cube.vert: the column-major glm matrix arrives transposed (Slang
    // lays the push-constant float4x4 row-major), so apply it from the right to get M * v.
    output.position = mul(float4(input.position, 1.0), pc.mvp);
    output.color = input.color;
    return output;
}
