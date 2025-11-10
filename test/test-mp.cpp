#include "common.h"
#include <random>
#include "utils.hpp"

#include <omp.h>

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
                m(i, j) = 1.0 + std::abs(distr(eng));
            else
                m(i, j) = std::abs(distr(eng));
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
    //cudart.setblockSize(1024);
    CPUTimer timer;
    timer.push("Total");
    timer.push("Setup");
    MixedPrecisionMatrix<T> A(shape);
    MixedPrecisionMatrix<T> B(shape);
    Matrix<T> C(shape);
    timer.pop();
    timer.push("FillData");
    fill_with_rd_values(A);
    fill_with_rd_values(B);
    timer.pop();
    timer.push("Copy2Device");
    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);
    timer.pop();
    
    timer.push("GEMM");
    MatrixMatrixProduct(cudart, A, B, C, 1.0, 0.0);
    cudart.synchronize();
    timer.pop();

    C.copy2host(cudart);
    cudart.synchronize();
    std::cout << "C(0,0)\t" << C(0,0) << std::endl;
    std::cout << "C(1,0)\t" << C(1,0) << std::endl;

    MixedPrecisionMatrix<T> C2(shape);
    C2.copy2device(cudart);

    timer.push("Markidis");
    MatrixMatrixProduct(cudart, mp_rt, A, B, C2, (T)1.0, (T)0.0);
    cudart.synchronize();
    timer.pop();
   
    cudart.synchronize();
    std::cout << "Markidis: Forb. Norm " << FrobeniusNorm(cudart, C, C2) << std::endl;
    timer.pop();
    std::cout << timer.print_entries() << std::endl;
    return FrobeniusNorm(cudart, C, C2);

}


int main()
{   
    for (int i = 10; i <16; i++)
    {
        std::cout.precision(12);
        int n = int((i*256));
        std::cout << "Shape: " << n << "x" << n << std::endl;
        Shape shape(n, n);
        //CompareGEMMS<float>(shape);
        double err = CompareGEMMS<double>(shape);
        if (err > 1e-5)
        {
            std::cout << "Test failed with error: " << err << std::endl;
            return 1;
        }
    }
    return 0;
};
