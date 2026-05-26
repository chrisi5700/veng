// Test vertex shader for buffer-backed geometry: position + color straight from a vertex
// buffer, no transform. Lets a MeshNode test prove that real vertex data (positions AND
// per-vertex color) flowed through the bound VkBuffer into the draw. Pairs with
// mesh_triangle.frag.
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

[shader("vertex")]
VSOut main(VSIn input)
{
    VSOut output;
    output.position = float4(input.position, 1.0);
    output.color = input.color;
    return output;
}
