// Vertical pass of the separable Gaussian (completing the 2D blur of the silhouette) AND the
// ring extraction: subtracting the sharp silhouette from the blurred one leaves only the halo
// that spills *outside* the cube — the outline. The kernel matches blur_h (wide + dense) so the
// glow is large on a 4K display. A wide blur spreads the mask's energy thin, so the difference
// is amplified (BOOST) to keep the halo bright, then tinted and written premultiplied so the
// composite can simply add it over the scene.
struct PSIn
{
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

[[vk::binding(0, 0)]] Texture2D blurredH; // horizontally-blurred silhouette
[[vk::binding(1, 0)]] Texture2D silhouette; // the sharp mask, to knock out the interior
[[vk::binding(2, 0)]] SamplerState samp;

struct Push
{
    float2 texel;
};
[[vk::push_constant]] Push pc;

static const int RADIUS = 80;    // must match blur_h
static const float SIGMA = 32.0; // must match blur_h
static const float BOOST = 4.0;  // widening dims the halo; amplify it back to a bright glow
static const float3 OUTLINE_COLOR = float3(0.15, 0.85, 1.0); // cyan glow

[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    float blurred = 0.0;
    float wsum = 0.0;
    for (int i = -RADIUS; i <= RADIUS; ++i)
    {
        const float w = exp(-float(i) * float(i) / (2.0 * SIGMA * SIGMA));
        blurred += blurredH.Sample(samp, input.uv + float2(0.0, pc.texel.y * float(i))).r * w;
        wsum += w;
    }
    blurred /= wsum;
    const float sharp = silhouette.Sample(samp, input.uv).r;
    const float ring = saturate((blurred - sharp) * BOOST); // halo outside the cube, brightened
    return float4(OUTLINE_COLOR * ring, ring);
}
