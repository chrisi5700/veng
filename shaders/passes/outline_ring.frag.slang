// Vertical pass of the separable Gaussian (completing the 2D blur) AND the ring extraction:
// subtracting the sharp silhouette from the blurred copy leaves only the halo that spills
// OUTSIDE the shape — the outline. The kernel matches outline_blur so the blur is symmetric,
// and `width` (the same spread multiplier the horizontal pass got) keeps both axes in step.
// `color` tints the halo. Widening the blur dims the halo (its energy spreads thin), so it is
// amplified (BOOST) and written PREMULTIPLIED, letting the compositor simply add it over the
// scene.
struct PSIn
{
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

[[vk::binding(0, 0)]] Texture2D blurredH;   // horizontally-blurred silhouette
[[vk::binding(1, 0)]] Texture2D silhouette; // the sharp mask, to knock out the interior
[[vk::binding(2, 0)]] SamplerState samp;

struct Push
{
    float4 color; // offset 0  — glow tint (rgb used, a ignored); first so its 16B slot has no pad
    float2 texel; // offset 16 — 1 / extent
    float  width; // offset 24 — glow spread multiplier (MUST match outline_blur)
};
[[vk::push_constant]] Push pc;

static const int   RADIUS = 80;   // must match outline_blur
static const float SIGMA  = 32.0; // must match outline_blur
static const float BOOST  = 4.0;  // widening dims the halo; amplify it back to a bright glow

[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    float blurred = 0.0;
    float wsum    = 0.0;
    for (int i = -RADIUS; i <= RADIUS; ++i)
    {
        const float w = exp(-float(i) * float(i) / (2.0 * SIGMA * SIGMA));
        blurred += blurredH.Sample(samp, input.uv + float2(0.0, pc.texel.y * float(i) * pc.width)).r * w;
        wsum += w;
    }
    blurred /= wsum;
    const float sharp = silhouette.Sample(samp, input.uv).r;
    const float ring  = saturate((blurred - sharp) * BOOST); // halo outside the shape, brightened
    return float4(pc.color.rgb * ring, ring);
}
