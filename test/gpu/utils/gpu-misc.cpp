#include "runtime.hpp"
#include "test_common.h"
#include <vector>
#include "timer.hpp"

template <typename T>
using Vector = gpu::Vector<T, CudaHostAllocator<T>, CudaDeviceAsyncAllocator<T>>;

template <typename T>
int test_std_vector_async(CudaRuntime& cudart)
{
    std::vector<Vector<T>> vecs;
    Vector<T> vec = Vector<T>(10, (T)1.0);
    // we create empty vectors without streams
    //vecs.resize(10);
    Vector<T> vec_tot(10, (T)0.0);
    for (int i = 0; i < 50; i++)
    {
        vecs.push_back(vec);

    }
    for (int i = 0; i < 50; i++)
    {
        lahva::gpu::AddVectors(cudart, (T)1.0, vecs[i], vec_tot);
    }
    vec_tot.copy2host(cudart);
    cudart.synchronize();
    if (vec_tot.sum() != (50*10)) return 1;

    //vecs.push_back(vec);
    // then we just deallocate them and see if that works without a segfault or cuda error
    return 0;
};

template <typename T>
int test_std_vector_push_GPU_values(CudaRuntime& cudart)
{
    std::vector<Vector<T>> vecs;
    Vector<T> vec = Vector<T>(10, (T)0.0);
    // we create empty vectors without streams
    //vecs.resize(10);
    Vector<T> vec_tot(10, (T)1.0);
    for (int i = 0; i < 50; i++)
    {

        lahva::gpu::AddVectors(cudart, (T)1.0, vec_tot, vec);
        cudart.synchronize();
        vecs.push_back(vec);
        cudart.synchronize();
        lahva::gpu::ScaleVector(cudart, (T)0.0, vec);
    }
    lahva::gpu::ScaleVector(cudart, (T)0.0, vec_tot);

    for (int i = 0; i < 50; i++)
    {
        lahva::gpu::AddVectors(cudart, (T)1.0, vecs[i], vec_tot);
    }
    vec_tot.copy2host(cudart);
    cudart.synchronize();
    if (vec_tot.sum() != (50*10)) return 1;

    //vecs.push_back(vec);
    // then we just deallocate them and see if that works without a segfault or cuda error
    return 0;
};

template <typename T>
int test_GPUTimer_stdConst(CudaRuntime& cudart)
{
    GPUTimer timer;
    Vector<T> vec = Vector<T>(10, (T)0.0);
    Vector<T> vec2 = Vector<T>(10, (T)0.0);
    timer.push("Test");
    for (int i = 0; i < 10; i++)
    {
       lahva::gpu::AddVectors(cudart, (T)1.0, vec, vec2);
    }
    timer.pop();
    timer.print_entries();
    return 0;
};

template <typename T>
int test_GPUTimer(CudaRuntime& cudart)
{
    GPUTimer timer(&cudart);
    Vector<T> vec = Vector<T>(10, (T)0.0);
    Vector<T> vec2 = Vector<T>(10, (T)0.0);
    timer.push("Test");
    for (int i = 0; i < 10; i++)
    {
       lahva::gpu::AddVectors(cudart, (T)1.0, vec, vec2);
    }
    timer.pop();
    timer.print_entries();
    return 0;
};

template <typename T>
int test_GPUTimer_stream(CudaRuntime& cudart)
{
    GPUTimer timer(&cudart);
    Vector<T> vec = Vector<T>(10, (T)0.0);
    Vector<T> vec2 = Vector<T>(10, (T)0.0);
    timer.push("Test", cudart.getStream());
    for (int i = 0; i < 10; i++)
    {
       lahva::gpu::AddVectors(cudart, (T)1.0, vec, vec2);
    }
    timer.pop(cudart.getStream());
    timer.print_entries();
    return 0;
};

template <typename T>
int test_GPUTimer_stream_no_pop(CudaRuntime& cudart)
{
    GPUTimer timer;
    Vector<T> vec = Vector<T>(10, (T)0.0);
    Vector<T> vec2 = Vector<T>(10, (T)0.0);
    timer.push("Test", cudart.getStream());
    for (int i = 0; i < 10; i++)
    {
       lahva::gpu::AddVectors(cudart, (T)1.0, vec, vec2);
    }
    //timer.pop(cudart.getStream());
    timer.print_entries();
    return 0;
};

int main()
{
    int stat = 0;
    CudaRuntime cudart = CudaRuntime();
    std::cout << "std_vector_async (double)" << std::endl;
    stat += test_std_vector_async<double>(cudart);
    std::cout << "std_vector_async (float)" << std::endl;
    stat += test_std_vector_async<float>(cudart);
    std::cout << "GPUTimer_stdConst (double)" << std::endl;
    stat += test_GPUTimer_stdConst<double>(cudart);
    std::cout << "GPUTimer_stdConst (float)" << std::endl;
    stat += test_GPUTimer_stdConst<float>(cudart);
    std::cout << "GPUTimer (double)" << std::endl;
    stat += test_GPUTimer<double>(cudart);
    std::cout << "GPUTimer (float)" << std::endl;
    stat += test_GPUTimer<float>(cudart);
    std::cout << "cuda create stream" << std::endl;
    cudart.createStream();
    std::cout << "std_vector_async (double)" << std::endl;
    stat += test_std_vector_async<double>(cudart);
    std::cout << "std_vector_async (float)" << std::endl;
    stat += test_std_vector_async<float>(cudart);
    std::cout << "GPUTimer (double)" << std::endl;
    stat += test_GPUTimer<double>(cudart);
    std::cout << "GPUTimer (float)" << std::endl;
    stat += test_GPUTimer<float>(cudart);
    std::cout << "GPUTimer_stream (double)" << std::endl;
    stat += test_GPUTimer_stream<double>(cudart);
    std::cout << "GPUTimer_stream (float)" << std::endl;
    stat += test_GPUTimer_stream<float>(cudart);
    std::cout << "GPUTimer_stream_no_pop (double)" << std::endl;
    stat += test_GPUTimer_stream_no_pop<double>(cudart);
    std::cout << "GPUTimer_stream_no_pop (float)" << std::endl;
    stat += test_GPUTimer_stream_no_pop<float>(cudart);
    std::cout << "std_vector_push_GPU_values (double)" << std::endl;
    stat += test_std_vector_push_GPU_values<double>(cudart);
    std::cout << "std_vector_push_GPU_values (float)" << std::endl;
    stat += test_std_vector_push_GPU_values<float>(cudart);

    return stat;
};