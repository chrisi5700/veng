// Fills the whole target with a push-constant color (paired with demo/fullscreen.vert). Lets
// the sampled-image test produce a known, uniform source texture to sample and read back.
struct PSIn
{
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

struct Push
{
    float4 color;
};
[[vk::push_constant]] Push pc;

[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    return pc.color;
}
