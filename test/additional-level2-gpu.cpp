#include "common.h"
#include "utils.hpp"

using namespace lahva::gpu;

template<typename T>
int test_add_matrices(CudaRuntime& cudart)
{
    Matrix<T> A(Shape(2,3), 1.0);
    Matrix<T> B(Shape(2,3), 2.0);
    Matrix<T> C(A.shape(), 0.0);

    AddMatrices(cudart, A, B, C, (T)1.0, (T)1.0);

    C.copy2host(cudart);
    cudart.synchronize();
    Matrix<T> res(A.shape(), 3.0);
    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices")))
    {
        std::cout << "Test failed: AddMatrices" << std::endl;
        return 1;
    }

    AddMatrices( cudart,"N", "N", (T)1.0, A, B, (T)1.0, C);
    C.copy2host(cudart);
    cudart.synchronize();

    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices")))
    {
        std::cout << "Test failed: AddMatrices" << std::endl;
        return 1;
    }

    return 0;
};

template<typename T>
int test_add_matrices_transposed(CudaRuntime& cudart)
{
    Matrix<T> A(Shape(2,3), 1.0);
    Matrix<T> B(Shape(3,2), 2.0);
    Matrix<T> C(A.shape(), 0.0);

    AddMatrices( cudart,A, B, C, (T)1.0, (T)1.0, "N", "T");
    C.copy2host(cudart);
    cudart.synchronize();
    Matrix<T> res(A.shape(), 3.0);

    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices")))
    {
        std::cout << "Test failed: AddMatrices" << std::endl;
        return 1;
    }

    AddMatrices( cudart,"N", "T", (T) 1.0, A, B, (T) 1.0, C);
    C.copy2host(cudart);
    cudart.synchronize();
    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices")))
    {
        std::cout << "Test failed: AddMatrices" << std::endl;
        return 1;
    }    

    B = Matrix<T>(Shape(2,3), 2.0);
    A = Matrix<T>(Shape(3,2), (T) 1.0);

    AddMatrices( cudart,A, B, C, (T) 1.0, (T) 1.0, "T", "N");
    C.copy2host(cudart);
    cudart.synchronize();
    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices")))
    {
        std::cout << "Test failed: AddMatrices" << std::endl;
        return 1;
    }

    AddMatrices( cudart,"T", "N", (T) 1.0, A, B, (T) 1.0, C);
    C.copy2host(cudart);
    cudart.synchronize();
    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices")))
    {
        std::cout << "Test failed: AddMatrices" << std::endl;
        return 1;
    }

    A = Matrix<T>(Shape(2,3), 1.0);
    res = Matrix<T>(Shape(3,2), 3.0);
    C = Matrix<T>(Shape(3,2), 0.0);

    AddMatrices( cudart,A, B, C, (T) 1.0, (T) 1.0, "T", "T");
    C.copy2host(cudart);
    cudart.synchronize();
    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices")))
    {
        std::cout << "Test failed: AddMatrices" << std::endl;
        return 1;
    }

    AddMatrices( cudart,"T", "T", (T) 1.0, A, B, (T) 1.0, C);
    C.copy2host(cudart);
    cudart.synchronize();
    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices")))
    {
        std::cout << "Test failed: AddMatrices" << std::endl;
        return 1;
    }

    return 0;
};


int main(){
    int exit = 0;
    CudaRuntime cudart(false);
    exit += test_add_matrices<double>(cudart);
    exit += test_add_matrices_transposed<double>(cudart);
    exit += test_add_matrices<float>(cudart);
    exit += test_add_matrices_transposed<float>(cudart);

    return exit;
};