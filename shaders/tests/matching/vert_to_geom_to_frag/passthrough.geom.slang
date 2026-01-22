// Geometry shader matching vertex output
struct GeometryInput
{
    [[vk::location(0)]] float4 color : COLOR;
};

struct GeometryOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float4 color : COLOR;
    [[vk::location(1)]] float3 barycentrics : TEXCOORD;
};

[shader("geometry")]
[maxvertexcount(3)]
void main(
    triangle GeometryInput input[3],
    inout TriangleStream<GeometryOutput> outputStream)
{
    GeometryOutput output;

    for (int i = 0; i < 3; i++) {
        output.position = float4(0, 0, 0, 1); // Placeholder
        output.color = input[i].color;
        output.barycentrics = float3(i == 0 ? 1 : 0, i == 1 ? 1 : 0, i == 2 ? 1 : 0);
        outputStream.Append(output);
    }
}
