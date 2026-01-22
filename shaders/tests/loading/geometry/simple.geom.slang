// Simple geometry shader for testing input/output extraction
struct GeometryInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float4 color : COLOR;
};

struct GeometryOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float4 color : COLOR;
};

[shader("geometry")]
[maxvertexcount(3)]
void main(
    triangle GeometryInput input[3],
    inout TriangleStream<GeometryOutput> outputStream)
{
    GeometryOutput output;
    for (int i = 0; i < 3; i++) {
        output.position = float4(input[i].position, 1.0);
        output.color = input[i].color;
        outputStream.Append(output);
    }
}
