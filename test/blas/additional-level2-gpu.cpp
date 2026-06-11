#include "../common.h"
#include "../utils.hpp"

using namespace lahva::gpu;


template<typename T>
std::initializer_list<T> getam(){
    std::initializer_list<T> am({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    return am;
};

template<typename T>
std::initializer_list<T> getbm(){
    std::initializer_list<T> bm({7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    return bm;
};

template<typename T>
std::initializer_list<T> getres()
{
    std::initializer_list<T> rm({8.0, 10.0, 12.0, 14.0, 16.0, 18.0});
    return rm;
};


template<typename T>
int test_add_matrices(CudaRuntime& cudart)
{
    Matrix<T> A(Shape(2,3), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Matrix<T> B(Shape(2,3), {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    Matrix<T> C(A.shape(), 0.0);

    AddMatrices(cudart, A, B, C, (T) 1.0, (T) 1.0);
    C.copy2host(cudart);
    cudart.synchronize();
    Matrix<T> res(A.shape(), {8.0, 10.0, 12.0, 14.0, 16.0, 18.0});
    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices")))
    {
        std::cout << "Test failed: AddMatrices" << std::endl;
        return 1;
    }

    AddMatrices(cudart, "N", "N", (T)1.0, A, B, (T)1.0, C);
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
    Matrix<T> A(Shape(2,3), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Matrix<T> B(Shape(3,2), {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    Matrix<T> C(A.shape(), 0.0);
    A.print();
    B.print();
    AddMatrices(cudart, A, B, C, (T)1.0, (T)1.0, "N", "T");
    C.copy2host(cudart);
    cudart.synchronize();
    Matrix<T> res(A.shape(), {8.0, 12.0, 11.0, 15.0, 14.0, 18.0});

    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices: B transposed")))
    {
        std::cout << "Test1 failed: AddMatrices" << std::endl;
        return 1;
    }

    AddMatrices(cudart, "N", "T", (T) 1.0, A, B, (T) 1.0, C);
    C.copy2host(cudart);
    cudart.synchronize();
    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrcies B transposed")))
    {
        std::cout << "Test2 failed: AddMatrices" << std::endl;
        return 1;
    }    

    B = Matrix<T>(Shape(2,3), {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    A = Matrix<T>(Shape(3,2), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    AddMatrices(cudart, A, B, C, (T) 1.0, (T) 1.0, "T", "N");
    C.copy2host(cudart);
    cudart.synchronize();
    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices A transposed")))
    {
        std::cout << "Test3 failed: AddMatrices" << std::endl;
        return 1;
    }

    AddMatrices(cudart, "T", "N", (T) 1.0, A, B, (T) 1.0, C);
    C.copy2host(cudart);
    cudart.synchronize();
    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices  A transposed")))
    {
        std::cout << "Test4 failed: AddMatrices" << std::endl;
        return 1;
    }

    A = Matrix<T>(Shape(2,3), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    res = Matrix<T>(Shape(3,2), {8.0, 12.0, 16.0, 10.0, 14.0, 18.0});
    C = Matrix<T>(Shape(3,2), 0.0);

    AddMatrices(cudart, A, B, C, (T) 1.0, (T) 1.0, "T", "T");
    C.copy2host(cudart);
    cudart.synchronize();
    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices A and B transposed")))
    {
        std::cout << "Test5 failed: AddMatrices" << std::endl;
        return 1;
    }

    AddMatrices(cudart, "T", "T", (T) 1.0, A, B, (T) 1.0, C);
    C.copy2host(cudart);
    cudart.synchronize(); 
    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices A and B transposed")))
    {
        std::cout << "Test6 failed: AddMatrices" << std::endl;
        return 1;
    }

    return 0;
};


int main(){
    int exit = 0;
    CudaRuntime cudart;
    exit += test_add_matrices<double>(cudart);
    exit += test_add_matrices_transposed<double>(cudart);
    exit += test_add_matrices<float>(cudart);
    exit += test_add_matrices_transposed<float>(cudart);

    return exit;
};