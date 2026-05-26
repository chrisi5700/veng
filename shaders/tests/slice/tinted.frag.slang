// Test fragment shader: the vertex color modulated by a uniform tint (descriptor set 0,
// binding 0). With white vertices the output is exactly the tint, so a readback proves the
// uniform buffer reached the shader through the descriptor set. Pairs with mesh_triangle.vert.
struct PSIn
{
    [[vk::location(0)]] float3 color : COLOR0;
};

struct Tint
{
    float4 tint;
};
[[vk::binding(0, 0)]] ConstantBuffer<Tint> tint;

[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    return float4(input.color * tint.tint.rgb, 1.0);
}
