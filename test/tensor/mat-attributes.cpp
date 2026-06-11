#include "../common.h"
#include "../utils.hpp"
#include <random>

using namespace lahva::gpu;

template<typename T>
using CPUMatrix = lahva::cpu::Matrix<T>;
template<typename T>
using GPUMatrix = Matrix<T, CudaHostAllocator<T>, CudaDeviceAsyncAllocator<T>>;

template<typename T>
using CPUVector = lahva::cpu::Vector<T>;
template<typename T>
using GPUVector = Vector<T, CudaHostAllocator<T>, CudaDeviceAsyncAllocator<T>>;

template <typename Tensor>
void fill_with_rd_values(Tensor& m)
{
    std::random_device rd;  // Obtain a random number from hardware
    std::minstd_rand eng(rd());

    std::normal_distribution<> distr(0.0, 1.0e+3);    
    for (size_t i = 0; i < m.size(); i++)
        m.data()[i] = distr(eng);
}

template <typename T>
int test_symmetrize_cpu_gpu(CudaRuntime& cudart)
{
    const int n = 10;
    
    // Generate matrices
    CPUMatrix<T> cpu_mat(Shape(n, n), 0.0);
    fill_with_rd_values(cpu_mat);

    GPUMatrix<T> gpu_mat(Shape(n, n), 0.0);
    cpu::CopyVectors(cpu_mat, gpu_mat);
    gpu_mat.copy2device(cudart);
    
    // Symmetrize both
    cpu_mat.symmetrize();
    gpu_mat.symmetrize(cudart);
    
    // Copy result back to host
    gpu_mat.copy2host(cudart);
    cudart.synchronize();

    // Comparison
    T eps = std::abs(std::nextafter(cpu_mat.data()[0], +INFINITY) - cpu_mat.data()[0]);
    if (!check(gpu_mat.data(), cpu_mat.data(), eps, cpu_mat.size(), "Symmetrize"))
    {
        std::cout << "Test failed: Symmetrize" << std::endl;
        return 1;
    }
    
    return 0;
}

template <typename T>
int test_get_diagonal_cpu_gpu(CudaRuntime& cudart)
{
    
    const int n = 10;
    
    // Generate matrices
    CPUMatrix<T> cpu_mat(Shape(n, n), 0.0);
    fill_with_rd_values(cpu_mat);

    GPUMatrix<T> gpu_mat(Shape(n, n), 0.0);
    cpu::CopyVectors(cpu_mat, gpu_mat);
    gpu_mat.copy2device(cudart);
    
    // Take diagonals
    CPUVector<T> cpu_diag = cpu_mat.get_diagonal();
    GPUVector<T> gpu_diag = gpu_mat.get_diagonal(cudart);
    
    // Copy result back to host
    gpu_diag.copy2host(cudart);
    cudart.synchronize();

    // Comparison
    T eps = std::abs(std::nextafter(cpu_diag.data()[0], +INFINITY) - cpu_diag.data()[0]);
    if (!check(gpu_diag.data(), cpu_diag.data(), eps, cpu_diag.size(), "Get Diagonal"))
    {
        std::cout << "Test failed: Get Diagonal" << std::endl;
        return 1;
    }
    
    return 0;

}

template <typename T>
int test_set_diagonal_cpu_gpu(CudaRuntime& cudart)
{
    const int n = 10;
    
    // Generate matrices
    CPUMatrix<T> cpu_mat(Shape(n,n), 0.0);
    GPUMatrix<T> gpu_mat(Shape(n,n), 0.0);
    
    // Set diagonals
    CPUVector<T> cpu_diag = CPUVector<T>(n);
    fill_with_rd_values(cpu_diag);
    cpu_mat.set_diagonal(cpu_diag);

    GPUVector<T> gpu_diag = GPUVector<T>(n);
    cpu::CopyVectors(cpu_diag, gpu_diag);
    gpu_mat.set_diagonal(cudart, gpu_diag);
    
    // Copy result back to host
    gpu_mat.copy2host(cudart);
    cudart.synchronize();

    // Comparison
    T eps = std::abs(std::nextafter(cpu_mat.data()[0], +INFINITY) - cpu_mat.data()[0]);
    if (!check(gpu_mat.data(), cpu_mat.data(), eps, cpu_mat.size(), "Set Diagonal"))
    {
        std::cout << "Test failed: Set Diagonal" << std::endl;
        return 1;
    }
    
    return 0;
}

int main() {

    int stat = 0;
    CudaRuntime cudart = CudaRuntime();
    std::cout << "symmetrize_cpu_gpu (double)" << std::endl;
    stat += test_symmetrize_cpu_gpu<double>(cudart);
     std::cout << "symmetrize_cpu_gpu (float)" << std::endl;
    stat += test_symmetrize_cpu_gpu<float>(cudart);
    std::cout << "get_diagonal_cpu_gpu (double)" << std::endl;
    stat += test_get_diagonal_cpu_gpu<double>(cudart);
    std::cout << "get_diagonal_cpu_gpu (float)" << std::endl;
    stat += test_get_diagonal_cpu_gpu<float>(cudart);
    std::cout << "set_diagonal_cpu_gpu (double)" << std::endl;
    stat += test_set_diagonal_cpu_gpu<double>(cudart);
    std::cout << "set_diagonal_cpu_gpu (float)" << std::endl;
    stat += test_set_diagonal_cpu_gpu<float>(cudart);
    return stat;
}