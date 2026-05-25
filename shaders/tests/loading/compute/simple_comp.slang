// Simple compute shader: scales each element of a storage buffer by a
// push-constant factor. Exercises both reflection paths the pipeline builder
// consumes — one descriptor (storage buffer) and a push-constant block.
struct Params
{
    uint  count;
    float scale;
};

[[vk::push_constant]]
Params params;

[[vk::binding(0, 0)]]
RWStructuredBuffer<float> data;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i < params.count)
    {
        data[i] = data[i] * params.scale;
    }
}
