#include "test_common.h"
#include <random>
#include <omp.h>
#include "timer.hpp"

using namespace lahva::gpu;
using lahva::Shape;
using lahva::CudaRuntime;
using lahva::Timer;


template <typename T>
void fill_diagonally_dominant(Matrix<T>& m)
{
    std::random_device rd;  // Obtain a random number from hardware
    std::minstd_rand eng(rd()); // Seed the generator

    std::normal_distribution<> distr(0.0, 0.1);
    #pragma omp parallel for shared(m)    
    for (size_t i = 0; i < m.shape().first; i++)
    {
        for (size_t j = 0; j < m.shape().second; j++)
        {
            if (i == j)
                m(i, j) = 1.0 + std::abs(distr(eng));
            else
                m(i, j) = std::abs(distr(eng));
        }
    }
}

template <typename T>
int CompareGEMMS(Shape& shape)
{
    CudaRuntime cudart(false);
    MPRuntime mp_rt;
    mp_rt.nsplits_FP32 = 4;
    mp_rt.fast_mode = true;
    mp_rt.batch_mode = true;

    MixedPrecisionMatrix<T> A(shape);
    MixedPrecisionMatrix<T> B(shape);
    Matrix<T> C(shape);
   
    fill_diagonally_dominant(A);
    fill_diagonally_dominant(B);
    
    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);

    MatrixMatrixProduct(cudart, A, B, C, 1.0, 0.0);
    cudart.synchronize();
    C.copy2host(cudart);
    cudart.synchronize();

    MixedPrecisionMatrix<T> C2(shape);
    C2.copy2device(cudart);
    MatrixMatrixProduct(cudart, mp_rt, A, B, C2, (T)1.0, (T)0.0);
    C2.copy2host(cudart);
    cudart.synchronize();

    // Not using the check function here because the regular double comparison is too strict for mixed-precision
    if (FrobeniusNorm(cudart, C, C2) > 1e-5) 
    {
        std::cout << "Test failed: Precision: " << get_type_name<T>() << " Shape: (" << shape.first << ", " << shape.second << ")" << "Frobenius Norm: " << FrobeniusNorm(cudart, C, C2) << std::endl;
        return TEST_FAIL;
    }

    return TEST_PASS;

}

template <typename T>
int mp_gemms()
{
    for (int i = 10; i <16; i++)
    {
        int n = int((i*256));
        Shape shape(n, n);
        if (CompareGEMMS<double>(shape)) return TEST_FAIL;
    }

    return TEST_PASS;
}

int main()
{   
    int total_failures = 0;
    total_failures += mp_gemms<double>();
    total_failures += mp_gemms<float>();

    if (total_failures > 0) {
        std::cerr << "gpu/utils/test-mp tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All gpu/utils/test-mp tests passed!" << std::endl;
    return TEST_PASS;
};
