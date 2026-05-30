// Horizontal pass of a separable Gaussian blur over the silhouette mask. RADIUS/SIGMA fix the
// kernel *quality* (tap count + falloff) at compile time; the `width` push constant scales the
// tap SPACING at runtime, which is what makes the outline's size configurable:
//   width = 1.0  -> the default ~RADIUS-texel reach
//   width > 1.0  -> the same taps spread wider: a larger, softer glow (very large values start
//                   to undersample, since the tap count is fixed)
//   width < 1.0  -> a tighter glow
// `texel` (= 1/extent) keeps the offsets in UV space so the size is resolution-independent.
// Output feeds the vertical (ring) pass, which must use the SAME width for a symmetric blur.
struct PSIn
{
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

[[vk::binding(0, 0)]] Texture2D silhouette;
[[vk::binding(1, 0)]] SamplerState samp;

struct Push
{
    float2 texel; // offset 0  — 1 / extent (UV-space texel size)
    float  width; // offset 8  — glow spread multiplier
};
[[vk::push_constant]] Push pc;

static const int   RADIUS = 80;   // taps each side (texels) at width = 1
static const float SIGMA  = 32.0; // gaussian falloff (texels)

[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    float sum  = 0.0;
    float wsum = 0.0;
    for (int i = -RADIUS; i <= RADIUS; ++i)
    {
        const float w = exp(-float(i) * float(i) / (2.0 * SIGMA * SIGMA));
        sum += silhouette.Sample(samp, input.uv + float2(pc.texel.x * float(i) * pc.width, 0.0)).r * w;
        wsum += w;
    }
    const float v = sum / wsum;
    return float4(v, v, v, 1.0);
}
