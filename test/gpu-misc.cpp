#include "runtime.hpp"
#include "common.h"
#include <vector>

using Vector =  gpu::Vector<double, CudaHostAllocator<double>, CudaDeviceAsyncAllocator<double>>;

int test_std_vector_async(CudaRuntime& cudart)
{
    std::vector<Vector> vecs;
    Vector vec = Vector(10, 1.0f);
    // we create empty vectors without streams 
    //vecs.resize(10);
    Vector vec_tot(10, 0.0f);
    for (int i = 0; i < 50; i++)
    {
        vecs.push_back(vec);
        
    }
    for (int i = 0; i < 50; i++)
    {
        lahva::gpu::AddVectors(cudart, 1.0, vecs[i], vec_tot);
    }
    vec_tot.copy2host(cudart);
    cudart.synchronize();
    if (vec_tot.sum() != (50*10)) return 1;

    //vecs.push_back(vec);
    // then we just deallocate them and see if that works without a segfault or cuda error
    return 0;
};

int test_GPUTimer_stdConst(CudaRuntime& cudart)
{
    GPUTimer timer;
    Vector vec = Vector(10, 0.0f);
    Vector vec2 = Vector(10, 0.0f);
    timer.push("Test");
    for (int i = 0; i < 10; i++)
    {
       lahva::gpu::AddVectors(cudart, 1.0, vec, vec2);
    }
    timer.pop();
    timer.print_entries();
    return 0;
};

int test_GPUTimer(CudaRuntime& cudart)
{
    GPUTimer timer(&cudart);
    Vector vec = Vector(10, 0.0f);
    Vector vec2 = Vector(10, 0.0f);
    timer.push("Test");
    for (int i = 0; i < 10; i++)
    {
       lahva::gpu::AddVectors(cudart, 1.0, vec, vec2);
    }
    timer.pop();
    timer.print_entries();
    return 0;
};

int test_GPUTimer_stream(CudaRuntime& cudart)
{
    GPUTimer timer(&cudart);
    Vector vec = Vector(10, 0.0f);
    Vector vec2 = Vector(10, 0.0f);
    timer.push("Test", cudart.getStream());
    for (int i = 0; i < 10; i++)
    {
       lahva::gpu::AddVectors(cudart, 1.0, vec, vec2);
    }
    timer.pop(cudart.getStream());
    timer.print_entries();
    return 0;
};

int test_GPUTimer_stream_no_pop(CudaRuntime& cudart)
{
    GPUTimer timer;
    Vector vec = Vector(10, 0.0f);
    Vector vec2 = Vector(10, 0.0f);
    timer.push("Test", cudart.getStream());
    for (int i = 0; i < 10; i++)
    {
       lahva::gpu::AddVectors(cudart, 1.0, vec, vec2);
    }
    //timer.pop(cudart.getStream());
    timer.print_entries();
    return 0;
};

int main()
{
    int stat = 0;
    CudaRuntime cudart = CudaRuntime();
    stat += test_std_vector_async(cudart);
    stat += test_GPUTimer_stdConst(cudart);
    stat += test_GPUTimer(cudart);
    cudart.createStream();
    stat += test_std_vector_async(cudart);
    stat += test_GPUTimer(cudart);
    stat += test_GPUTimer_stream(cudart);
    stat += test_GPUTimer_stream_no_pop(cudart);


    return stat;
};