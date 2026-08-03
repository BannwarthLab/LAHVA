#include <omp.h>
#include <random>
#include "test_common.h"
#include "utils.hpp"
#include "timer.hpp"

using namespace lahva::gpu;
template <typename T>
void fill_with_rd_values(Matrix<T>& m)
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
                m(i, j) = (T)(1.0 + std::abs(distr(eng)));
            else
                m(i, j) = (T)std::abs(distr(eng));
        }
    }
}

template <typename T>
double CompareGEMMS(Shape& shape)
{
    CudaRuntime cudart(false);
    MPRuntime mp_rt;
    mp_rt.fast_mode = true;
    mp_rt.batch_mode = true;
   
    MixedPrecisionMatrix<T> A(shape);
    MixedPrecisionMatrix<T> B(shape);
    Matrix<T> C(shape);
    
    fill_with_rd_values(A);
    fill_with_rd_values(B);
   
    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);
   
    MatrixMatrixProduct(cudart, A, B, C, (T)1.0, (T)0.0);
    cudart.synchronize();
    
    C.copy2host(cudart);
    cudart.synchronize();
    std::cout << "C(0,0)\t" << C(0,0) << std::endl;
    std::cout << "C(1,0)\t" << C(1,0) << std::endl;

    MixedPrecisionMatrix<T> C2(shape);
    C2.copy2device(cudart);

    MatrixMatrixProduct(cudart, mp_rt, A, B, C2, (T)1.0, (T)0.0);
    cudart.synchronize();

    std::cout << "Ozaki: Forb. Norm " << FrobeniusNorm(cudart, C, C2) << std::endl;
    return FrobeniusNorm(cudart, C, C2);
}


int main()
{
    std::cout.precision(12);

    Shape shape(2560, 2560);
    std::cout << "Shape: 2560x2560 (double)" << std::endl;
    double err = CompareGEMMS<double>(shape);
    if (err > 1e-5)
    {
        std::cout << "Test failed with error: " << err << std::endl;
        return 1;
    }

    std::cout << "Shape: 2560x2560 (float)" << std::endl;
    double err_float = CompareGEMMS<float>(shape);
    if (err_float > 0.1)
    {
        std::cout << "Test failed with error: " << err_float << std::endl;
        return 1;
    }

    return 0;
};
