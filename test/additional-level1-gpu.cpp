#include "common.h"
#include "utils.hpp"
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
    MyMatrix<T> A(Shape(100,100));
    fill_with_rd_values(A);
    T res_gpu = 1.0;
    res_gpu = FrobeniusNorm(cudart, A);

    T res_cpu = 0.0;
    res_cpu = FrobeniusNorm(A);
    
    T eps = std::abs( std::nextafter(res_cpu, +INFINITY) -res_cpu);

    if (!check(res_gpu, res_cpu, eps*10, "FrobeniusNorm"))
    {
        std::cout << "Test failed: FrobeniusNorm" << std::endl;
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
    std::cout <<eps * 10<< std::endl;
    
    if (!check(res_gpu, res_cpu, eps*10, "FrobeniusNorm2"))
    {
        std::cout << "Test failed: FrobeniusNorm2" << std::endl;
        std::cout << "GPU: " << res_gpu << " CPU: " << res_cpu << std::endl;
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

int main(){
    int exit = 0;
    CudaRuntime cudart;
    exit += test_Frobenius_norm<double>(cudart);
    exit += test_Frobenius_norm<float>(cudart);
    exit += test_Frobenius_norm2<double>(cudart);
    exit += test_Frobenius_norm2<float>(cudart);
    exit += test_Frobenius_norm2_diff<double>(cudart);
    exit += test_Frobenius_norm2_diff<float>(cudart);

    return exit;
};