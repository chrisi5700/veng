// Fragment shader expecting float4 at location 0 (mismatch!)
struct FragmentInput
{
    [[vk::location(0)]] float4 normal : NORMAL;
};

[shader("fragment")]
float4 main(FragmentInput input) : SV_Target
{
    return input.normal;
}
