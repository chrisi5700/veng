// Silhouette fragment shader for veng::passes::OutlinePass: fill every covered pixel with
// solid white. That white mask is what the blur turns into a glow, and what the ring pass
// subtracts from the blurred copy to leave only the halo spilling OUTSIDE the shape. The
// interpolated color is ignored — it exists only to match the vertex shader's output
// interface (reflection-checked vert-out -> frag-in compatibility).
struct PSIn
{
    [[vk::location(0)]] float3 color : COLOR0;
};

[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    return float4(1.0, 1.0, 1.0, 1.0);
}
