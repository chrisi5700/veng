// Horizontal pass of a separable Gaussian blur over the silhouette mask. A wide, densely
// sampled kernel (radius RADIUS texels, sigma SIGMA) so the glow is large and smooth — sized
// to be clearly visible on a high-DPI / 4K display, not a hairline rim. `texel` (= 1/extent)
// arrives as a push constant so the offsets are in UV space at any resolution. Output feeds
// the vertical (ring) pass.
struct PSIn
{
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

[[vk::binding(0, 0)]] Texture2D silhouette;
[[vk::binding(1, 0)]] SamplerState samp;

struct Push
{
    float2 texel;
};
[[vk::push_constant]] Push pc;

static const int RADIUS = 80;    // taps each side (in texels) — ~10x the old ~8px reach
static const float SIGMA = 32.0; // gaussian falloff (texels)

[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    float sum = 0.0;
    float wsum = 0.0;
    for (int i = -RADIUS; i <= RADIUS; ++i)
    {
        const float w = exp(-float(i) * float(i) / (2.0 * SIGMA * SIGMA));
        sum += silhouette.Sample(samp, input.uv + float2(pc.texel.x * float(i), 0.0)).r * w;
        wsum += w;
    }
    const float v = sum / wsum;
    return float4(v, v, v, 1.0);
}
