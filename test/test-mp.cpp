#include "common.h"
#include <random>
#include "utils.hpp"



using namespace lahva::gpu;
template <typename T>
void fill_with_rd_values(Matrix<T>& m)
{
    std::random_device rd;  // Obtain a random number from hardware
    std::minstd_rand eng(7);

    std::normal_distribution<> distr(0.0, 1.0e+0);    
    for (size_t i = 0; i < m.shape().first; i++)
    {
        for (size_t j = 0; j < m.shape().second; j++)
        {
            m(i, j) = distr(eng);
        }
    }
}

template <typename T>
void CompareGEMMS(Shape& shape)
{
    CudaRuntime cudart(false);
    //cudart.setblockSize(1024);
    CPUTimer timer;

    MixedPrecisionMatrix<T> A(shape);
    MixedPrecisionMatrix<T> B(shape);
    Matrix<T> C(shape);

    fill_with_rd_values(A);
    fill_with_rd_values(B);
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
    Matrix<T> buffer(shape, cudart, C2.get_gpuallocator());
    MatrixMatrixProduct(cudart, A, B, C2, buffer, (T)1.0, (T)0.0);
    cudart.synchronize();
    timer.pop();

    C2.copy2host(cudart);
    cudart.synchronize();
    //std::cout << "C2(0,0)\t" << C2[84118] << std::endl;
    //std::cout << "C2(1,0)\t" << C[84118] << std::endl;

    
    cudart.synchronize();
    std::cout << "Markidis: Forb. Norm " << FrobeniusNorm(C, C2) << " " << FrobeniusNorm(cudart, C, C2) << std::endl;
    AddVectors(cudart, -1.0, C, C2);
    C2.copy2host(cudart);
    cudart.cublasSetStream_();
    
    std::cout << timer.print_entries() << std::endl;
}


int main()
{   
    for (int i = 1; i <9; i++)
    {
        std::cout.precision(12);
        int n = int((i*256));
        std::cout << "Shape: " << n << "x" << n << std::endl;
        Shape shape(n, n);
        //CompareGEMMS<float>(shape);
        CompareGEMMS<double>(shape);
    }
};
