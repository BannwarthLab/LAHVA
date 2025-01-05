#include "common.h"
#include <random>

using namespace lahva::gpu;
template <typename T>
void fill_with_rd_values(Matrix<T>& m)
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

void CompareGEMMS(Shape& shape)
{
    CudaRuntime cudart(false);
    cudart.setblockSize(1024);
    CPUTimer timer;
    
    Matrix<float> A(shape);
    Matrix<float> B(shape);
    Matrix<float> C(shape);
    

    fill_with_rd_values(A);
    fill_with_rd_values(B);
    timer.push("Copy2Device");
    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);
    timer.pop();
    
    timer.push("SGEMM");
    MatrixMatrixProduct(cudart, A, B, C, 1.0, 0.0);
    cudart.synchronize();
    timer.pop();

    C.copy2host(cudart);
    cudart.synchronize();
    std::cout << "C(0,0)\t" << C(0,0) << std::endl;

    Matrix<float> C1(shape);
    C1.copy2device(cudart);

    timer.push("TF32");
    MatrixMatrixProductTF32(cudart, A, B, C1, 1.0, 0.0);
    cudart.synchronize();
    timer.pop();

    C1.copy2host(cudart);
    cudart.synchronize();
    std::cout << "C1(0,0)\t" << C1(0,0) << std::endl;
    AddVectors(cudart, -1.0, C, C1);
    C1.copy2host(cudart);
    cudart.synchronize();
    
    std::cout << "TF32: Forb. Norm " << FrobeniusNorm( C1) << std::endl; 
    std::cout << "TF32: Forb. Sum " << C1.sum() << " " << C1(0,0) << std::endl;

    Matrix<float> C2(shape);
    C2.copy2device(cudart);

    timer.push("Markidis");
    MatrixMatrixProductMP(cudart, A, B, C2, 1.0, 0.0);
    cudart.synchronize();
    timer.pop();

    C2.copy2host(cudart);
    cudart.synchronize();
    std::cout << "C2(0,0)\t" << C2(0,0) << std::endl;
    AddVectors(cudart, -1.0, C, C2);
    C2.copy2host(cudart);
    cudart.synchronize();
    std::cout << "Markidis: Forb. Norm " << FrobeniusNorm( C2) << " " << FrobeniusNorm(cudart, C2) << std::endl; 
    std::cout << "Markidis: Forb. Sum " << C2.sum()<< " " << C2(0,0) << std::endl; 

    timer.print_entries();
}


int main()
{   
    for (int i = 1; i < 10; i++)
    {
        std::cout.precision(12);
        int n = int((i*10)/8)*8;
        std::cout << "Shape: " << n << "x" << n << std::endl;
        Shape shape(n, n);
        CompareGEMMS(shape);
    }
};