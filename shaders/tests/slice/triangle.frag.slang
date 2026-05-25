// Solid red, so a readback can tell triangle pixels from the clear color.
[shader("fragment")]
float4 main() : SV_Target
{
    return float4(1.0, 0.0, 0.0, 1.0);
}
