#include "test_common.h"
#include "array_utils.hpp"

using namespace lahva::gpu;
using lahva::Shape;
using lahva::CudaRuntime;

template<typename T>
using MyMatrix = Matrix<T, lahva::CudaHostAllocator<T>, lahva::CudaDeviceAsyncAllocator<T>>;

template <typename T>
int test_Frobenius_norm(const CudaRuntime& cudart)
{
    MyMatrix<T> A(Shape(50,50));
    fill_with_rd_values(A);
    T res_gpu = 1.0;
    res_gpu = FrobeniusNorm(cudart, A);

    T res_cpu = 0.0;
    res_cpu = FrobeniusNorm(A);

    if (!check(res_gpu, res_cpu, check_msg(get_type_name<T>(), "")))
    {
        std::cout << "Test failed: FrobeniusNorm" << std::endl;
        std::cout << "Frobenius norm on the GPU: " << res_gpu << std::endl;
        std::cout << "Frobenius norm on the CPU: " << res_cpu << std::endl;
        std::cout << "Diff: " << std::abs(res_gpu - res_cpu) << std::endl;
        return 1;
    }
    
    return 0;
};

template <typename T>
int test_Frobenius_norm2(const CudaRuntime& cudart)
{
    MyMatrix<T> A(Shape(100,100));
    fill_with_rd_values(A);

    MyMatrix<T> B(Shape(100,100));
    fill_with_rd_values(B);

    T res_gpu = 1.0;
    res_gpu = FrobeniusNorm(cudart, A, B);

    T res_cpu = 0.0;
    res_cpu = FrobeniusNorm(A, B);

    if (!check(res_gpu, res_cpu, check_msg(get_type_name<T>(), "")))
    {
        std::cout << "Test failed: FrobeniusNorm2" << std::endl;
        std::cout << "Frobenius norm2 on the GPU: " << res_gpu << std::endl;
        std::cout << "Frobenius norm2 on the CPU: " << res_cpu << std::endl;
        std::cout << "Diff: " << std::abs(res_gpu - res_cpu) << std::endl;
        return 1;
    }
    
    return 0;
};

template <typename T>
int test_Frobenius_norm2_diff(const CudaRuntime& cudart)
{
    MyMatrix<T> A(Shape(100,100));
    fill_with_rd_values(A);

    MyMatrix<T> B(Shape(100,100));
    fill_with_rd_values(B);

    T res_gpu = 1.0;
    res_gpu = FrobeniusNorm(cudart, A, B);

    T res_gpu_diff = 0.0;
    AddVectors(cudart, (T)-1.0, A, B);
    res_gpu_diff = FrobeniusNorm(cudart, B);

    if (!check(res_gpu, res_gpu_diff, check_msg(get_type_name<T>(), "")))
    {
        std::cout << "Test failed: FrobeniusNorm2, take the differenc first" << std::endl;
        std::cout << "GPU: " << res_gpu << " GPU first difference then Frob: " << res_gpu_diff << std::endl;
        std::cout << "Diff: " << std::abs(res_gpu - res_gpu_diff) << std::endl;
        return 1;
    }
    
    return 0;
};

template <typename T>
int test_HadamardProduct(const CudaRuntime& cudart)
{

    int n=1000;
    std::cout << "Matrix size (n): " << n << std::endl;

    MyMatrix<T> A(Shape(n,n),0.0);
    MyMatrix<T> B(Shape(n,n),0.0);
    MyMatrix<T> C_gpu(Shape(n,n),0.0);
    
    fill_with_rd_values(A);
    fill_with_rd_values(B);

    A.updateGPUvalues(cudart);
    B.updateGPUvalues(cudart);
    cudart.synchronize();
    
    // GPU Hadamard product
    HadamardProduct(cudart, A, B, C_gpu);
    C_gpu.copy2host(cudart);
    cudart.synchronize();
    
    // CPU Hadamard product
    using CPUMatrix = lahva::cpu::Matrix<T, lahva::StdAllocator<T>>;
    CPUMatrix A_cpu(Shape(n,n),0.0);
    CPUMatrix B_cpu(Shape(n,n),0.0);
    CPUMatrix C_cpu(Shape(n,n),0.0);

    lahva::cpu::CopyVectors(A, A_cpu);
    lahva::cpu::CopyVectors(B, B_cpu);
    lahva::cpu::HadamardProduct(A_cpu, B_cpu, C_cpu);
    
    // Comparison
    if (!(check(C_gpu.data(), C_cpu.data(), C_cpu.size(), check_msg(get_type_name<T>(), ""))))
    {
        std::cout << "Test failed: HadamardProduct" << std::endl;
        return 1;
    }

    return 0;
};

// ============================================================================
// Main
// ============================================================================

int main(){
    int total_failures = 0;
    CudaRuntime cudart;
    total_failures += test_Frobenius_norm<double>(cudart);
    total_failures += test_Frobenius_norm<float>(cudart);
    total_failures += test_Frobenius_norm2<double>(cudart);
    total_failures += test_Frobenius_norm2<float>(cudart);
    total_failures += test_Frobenius_norm2_diff<double>(cudart);
    total_failures += test_Frobenius_norm2_diff<float>(cudart);
    total_failures += test_HadamardProduct<double>(cudart);
    total_failures += test_HadamardProduct<float>(cudart);

    if (total_failures > 0) {
        std::cerr << "gpu/blas/additional-level1 tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All gpu/blas/additional-level1 tests passed!" << std::endl;
    return TEST_PASS;
};