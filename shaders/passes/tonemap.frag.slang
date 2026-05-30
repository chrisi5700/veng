// Tonemap / colour-management resolve. Samples the engine's linear-light HDR scene target
// (R16G16B16A16_SFLOAT, values may exceed 1.0), applies an exposure scale and the ACES filmic
// curve to map HDR linear -> display-referred linear [0,1], and writes that. The pass's colour
// attachment is the swapchain's _SRGB format, so the hardware encodes linear -> sRGB on store —
// the shader therefore outputs LINEAR and must NOT apply the sRGB curve itself (doing so would
// double-encode). This is the single point where lighting leaves linear space, the foundation
// for physically-correct lighting and (linear) transparency blending upstream.
struct PSIn
{
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

[[vk::binding(0, 0)]] Texture2D    hdr;
[[vk::binding(1, 0)]] SamplerState samp;

struct Push
{
    float exposure; // linear multiplier applied before the curve (1.0 = neutral)
};
[[vk::push_constant]] Push pc;

// Narkowicz 2015 ACES filmic approximation (operates on linear HDR, returns linear [0,1]).
float3 aces(float3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    const float3 hdr_linear = hdr.Sample(samp, input.uv).rgb * pc.exposure;
    return float4(aces(hdr_linear), 1.0);
}
