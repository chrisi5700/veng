// Composite pass (fullscreen): the displayed image = the rendered scene plus the outline
// ring added on top. Additive, so when the outline is disabled the `ring` input is rebound to
// a black texture and this emits the scene unchanged — the "passes scene through" path, with
// no branch in the shader. Output is left in TRANSFER_SRC for the present blit.
struct PSIn
{
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

[[vk::binding(0, 0)]] Texture2D scene;
[[vk::binding(1, 0)]] Texture2D ring;
[[vk::binding(2, 0)]] SamplerState samp;

[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    const float3 base = scene.Sample(samp, input.uv).rgb;
    const float3 glow = ring.Sample(samp, input.uv).rgb;
    return float4(base + glow, 1.0);
}
