#include "../common.h"
#include "../utils.hpp"
#include <random>
#include <limits>

using namespace lahva::gpu;

template<typename T>
using MyMatrix = Matrix<T, CudaHostAllocator<T>, CudaDeviceAsyncAllocator<T>>;

template <typename T>
void fill_with_rd_values(Matrix_<T>& m)
{
    std::random_device rd;  // Obtain a random number from hardware
    std::minstd_rand eng(rd());

    std::normal_distribution<> distr(0.0, 1.0e+3);    
    for (size_t i = 0; i < m.shape().first; i++)
    {
        for (size_t j = 0; j < m.shape().second; j++)
        {
            m(i, j) = distr(eng);
        }
    }
}

template <class T>
bool IsEqual(T rhs, T lhs)
{
    T diff = std::abs(lhs - rhs);

    T epsilon = std::numeric_limits<T>::epsilon( ) * std::max(std::abs(rhs), std::abs(lhs));
    std::cout << "Epsilon: " << epsilon << std::endl;
    return diff <= epsilon ;
}

template <typename T>
int test_Frobenius_norm(const CudaRuntime& cudart)
{
    MyMatrix<T> A(Shape(50,50));
    fill_with_rd_values(A);
    T res_gpu = 1.0;
    res_gpu = FrobeniusNorm(cudart, A);

    T res_cpu = 0.0;
    res_cpu = FrobeniusNorm(A);
    
    T eps = std::abs( std::nextafter(res_gpu, +INFINITY) -res_gpu);
    std::cout <<eps * 10<< std::endl;

    if (!check(res_gpu, res_cpu, eps*100, "FrobeniusNorm"))
    {
        std::cout << "Test failed: FrobeniusNorm" << std::endl;
        std::cout << "Frobenius norm on the GPU: " << res_gpu << std::endl;
        std::cout << "Frobenius norm on the CPU: " << res_cpu << std::endl;
        std::cout << "Diff: " << std::abs(res_gpu - res_cpu) << std::endl;
        std::cout << "Threshold: " << eps*100 << std::endl;
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
    T eps = std::abs( std::nextafter(res_cpu, +INFINITY) -res_cpu);
    
    if (!check(res_gpu, res_cpu, eps*100, "FrobeniusNorm2"))
    {
        std::cout << "Test failed: FrobeniusNorm2" << std::endl;
        std::cout << "Frobenius norm2 on the GPU: " << res_gpu << std::endl;
        std::cout << "Frobenius norm2 on the CPU: " << res_cpu << std::endl;
        std::cout << "Diff: " << std::abs(res_gpu - res_cpu) << std::endl;
        std::cout << "Threshold: " << eps*100 << std::endl;
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
    
    T eps = std::abs( std::nextafter(res_gpu, +INFINITY) -res_gpu);
    
    
    if (!check(res_gpu, res_gpu_diff, eps*10, "FrobeniusNorm2 Differnce then FrobeniusNorm"))
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
    using CPUMatrix = cpu::Matrix<T, StdAllocator<T>>;
    CPUMatrix A_cpu(Shape(n,n),0.0);
    CPUMatrix B_cpu(Shape(n,n),0.0);
    CPUMatrix C_cpu(Shape(n,n),0.0);

    cpu::CopyVectors(A, A_cpu);
    cpu::CopyVectors(B, B_cpu);
    cpu::HadamardProduct(A_cpu, B_cpu, C_cpu);
    
    // Comparison
    T eps = std::abs( std::nextafter(C_cpu.data()[0], +INFINITY) -C_cpu.data()[0]);
    if (!(check(C_gpu.data(), C_cpu.data(), eps, C_cpu.size(), "HadamardProduct")))
    {
        std::cout << "Test failed: HadamardProduct" << std::endl;
        return 1;
    }

    return 0;
};


int main(){
    int exit = 0;
    CudaRuntime cudart;
    std::cout << "Starting test_Frobenius_norm (double)" << std::endl;
    exit += test_Frobenius_norm<double>(cudart);

    std::cout << "Starting test_Frobenius_norm (float)" << std::endl;
    exit += test_Frobenius_norm<float>(cudart);

    std::cout << "Starting test_Frobenius_norm2 (double)" << std::endl;
    exit += test_Frobenius_norm2<double>(cudart);

    std::cout << "Starting test_Frobenius_norm2 (float)" << std::endl;
    exit += test_Frobenius_norm2<float>(cudart);

    std::cout << "Starting test_Frobenius_norm2_diff (double)" << std::endl;
    exit += test_Frobenius_norm2_diff<double>(cudart);

    std::cout << "Starting test_Frobenius_norm2_diff (float)" << std::endl;
    exit += test_Frobenius_norm2_diff<float>(cudart);

    std::cout << "Starting test_HadamardProduct (double)" << std::endl;
    exit += test_HadamardProduct<double>(cudart);

    std::cout << "Starting test_HadamardProduct (float)" << std::endl;
    exit += test_HadamardProduct<float>(cudart);

    return exit;
};