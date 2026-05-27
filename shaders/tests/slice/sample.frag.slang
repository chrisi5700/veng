// Trivial sampled-image fragment shader for the sampled-image test: return the input texture
// sampled at the fullscreen UV. Proves a GraphicsNode can bind a prior pass's output as a
// texture (sampler + image descriptor) and read it back. Paired with demo/fullscreen.vert.
struct PSIn
{
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

[[vk::binding(0, 0)]] Texture2D source;
[[vk::binding(1, 0)]] SamplerState samp;

[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    return source.Sample(samp, input.uv);
}
