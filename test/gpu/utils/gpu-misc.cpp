#include "test_common.h"
#include "array_utils.hpp"
#include "timer.hpp"

using lahva::CudaRuntime;
using lahva::GPUTimer;
template <typename T>
using Vector = lahva::gpu::Vector<T, lahva::CudaHostAllocator<T>, lahva::CudaDeviceAsyncAllocator<T>>;

// ============================================================================
// Async Vector and GPU Timer Tests
// ============================================================================

template <typename T>
int test_std_vector_async(CudaRuntime& cudart)
{
    std::vector<Vector<T>> vecs;
    Vector<T> vec = Vector<T>(10, (T)1.0);
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

    T expected = (T)(50*10);
    if (!check(vec_tot.sum(), expected, check_msg(get_type_name<T>(), "async vector sum"))) return TEST_FAIL;
    return TEST_PASS;
};

template <typename T>
int test_std_vector_push_GPU_values(CudaRuntime& cudart)
{
    std::vector<Vector<T>> vecs;
    Vector<T> vec = Vector<T>(10, (T)0.0);

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

    T expected = (T)(50*10);
    if (!check(vec_tot.sum(), expected, check_msg(get_type_name<T>(), "GPU push values sum"))) return TEST_FAIL;
    return TEST_PASS;
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
    cudart.synchronize();

    auto entries = timer.print_entries();
    if (entries.empty()) {
        std::cerr << check_msg(get_type_name<T>(), "");
        return TEST_FAIL;
    }
    return TEST_PASS;
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
    cudart.synchronize();

    auto entries = timer.print_entries();
    if (entries.empty()) {
        std::cerr << check_msg(get_type_name<T>(), "");
        return TEST_FAIL;
    }
    return TEST_PASS;
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
    cudart.synchronize();

    auto entries = timer.print_entries();
    if (entries.empty()) {
        std::cerr << check_msg(get_type_name<T>(), "");
        return TEST_FAIL;
    }
    return TEST_PASS;
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
    timer.pop(cudart.getStream());
    cudart.synchronize();

    auto entries = timer.print_entries();
    if (entries.empty()) {
        std::cerr << check_msg(get_type_name<T>(), "");
        return TEST_FAIL;
    }
    return TEST_PASS;
};

// ============================================================================
// Main
// ============================================================================

int main()
{
    int total_failures = 0;
    CudaRuntime cudart = CudaRuntime();

    // Async vector and GPU timer tests without stream
    total_failures += test_std_vector_async<double>(cudart);
    total_failures += test_std_vector_async<float>(cudart);
    total_failures += test_GPUTimer_stdConst<double>(cudart);
    total_failures += test_GPUTimer_stdConst<float>(cudart);
    total_failures += test_GPUTimer<double>(cudart);
    total_failures += test_GPUTimer<float>(cudart);

    // Async vector and GPU timer tests with stream
    cudart.createStream();

    total_failures += test_std_vector_async<double>(cudart);
    total_failures += test_std_vector_async<float>(cudart);
    total_failures += test_GPUTimer<double>(cudart);
    total_failures += test_GPUTimer<float>(cudart);
    total_failures += test_GPUTimer_stream<double>(cudart);
    total_failures += test_GPUTimer_stream<float>(cudart);
    total_failures += test_GPUTimer_stream_no_pop<double>(cudart);
    total_failures += test_GPUTimer_stream_no_pop<float>(cudart);
    total_failures += test_std_vector_push_GPU_values<double>(cudart);
    total_failures += test_std_vector_push_GPU_values<float>(cudart);

    if (total_failures > 0) {
        std::cerr << "gpu/utils/gpu-misc tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All gpu/utils/gpu-misc tests passed!" << std::endl;
    return TEST_PASS;
};