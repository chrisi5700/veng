// Fragment shader for the mesh cube: the interpolated per-vertex color modulated by a
// global tint that arrives in a uniform buffer (descriptor set 0, binding 0). The tint is
// uploaded by a UniformNode and bound by GraphicsNode::add_uniform — this is the demo's
// `add_uniform` end-to-end. Pairs with mesh.vert.
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
