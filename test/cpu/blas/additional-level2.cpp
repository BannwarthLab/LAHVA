#include "test_common.h"

using namespace lahva::cpu;
using lahva::Shape;

// ============================================================================
// Add matrices Tests
// ============================================================================

template<typename T>
int test_add_matrices()
{ 
    Matrix<T> A(Shape(2,3), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Matrix<T> B(Shape(2,3), {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    Matrix<T> C(A.shape(), 0.0);

    AddMatrices(A, B, C, (T) 1.0, (T) 1.0);

    Matrix<T> res(A.shape(),{8.0, 10.0, 12.0, 14.0, 16.0, 18.0});
    if (!(check(C.data(), res.data(), res.size(), check_msg(get_type_name<T>(), "check 1"))))
    {
        return TEST_FAIL;
    }

    AddMatrices("N", "N", (T)1.0, A, (T)1.0, B, C);

    if (!(check(C.data(), res.data(), res.size(), check_msg(get_type_name<T>(), "check 2"))))
    {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template<typename T>
int test_add_matrices_transposed()
{
    Matrix<T> A(Shape(2,3), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Matrix<T> B(Shape(3,2), {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    Matrix<T> C(A.shape(), 0.0);
    A.print();
    B.print();
    AddMatrices(A, B, C, (T)1.0, (T)1.0, "N", "T");

    Matrix<T> res(A.shape(), {8.0, 12.0, 11.0, 15.0, 14.0, 18.0});

    if (!(check(C.data(), res.data(), res.size(), check_msg(get_type_name<T>(), "check 1"))))
    {
        return TEST_FAIL;
    }

    AddMatrices("N", "T", (T) 1.0, A, (T) 1.0, B, C);

    if (!(check(C.data(), res.data(), res.size(), check_msg(get_type_name<T>(), "check 2"))))
    {
        return TEST_FAIL;
    }

    B = Matrix<T>(Shape(2,3), {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    A = Matrix<T>(Shape(3,2), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    AddMatrices(A, B, C, (T) 1.0, (T) 1.0, "T", "N");

    if (!(check(C.data(), res.data(), res.size(), check_msg(get_type_name<T>(), "check 3"))))
    {
        return TEST_FAIL;
    }

    AddMatrices("T", "N", (T) 1.0, A, (T) 1.0, B, C);

    if (!(check(C.data(), res.data(), res.size(), check_msg(get_type_name<T>(), "check 4"))))
    {
        return TEST_FAIL;
    }

    A = Matrix<T>(Shape(2,3), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    res = Matrix<T>(Shape(3,2), {8.0, 12.0, 16.0, 10.0, 14.0, 18.0});
    C = Matrix<T>(Shape(3,2), 0.0);

    AddMatrices(A, B, C, (T) 1.0, (T) 1.0, "T", "T");

    if (!(check(C.data(), res.data(), res.size(), check_msg(get_type_name<T>(), "check 5"))))
    {
        return TEST_FAIL;
    }

    AddMatrices("T", "T", (T) 1.0, A, (T) 1.0, B, C);

    if (!(check(C.data(), res.data(), res.size(), check_msg(get_type_name<T>(), "check 6"))))
    {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

// ============================================================================
// Main
// ============================================================================

int main(){
    int total_failures = 0;
    total_failures += test_add_matrices<double>();
    total_failures += test_add_matrices_transposed<double>();
    total_failures += test_add_matrices<float>();
    total_failures += test_add_matrices_transposed<float>();

    if (total_failures > 0) {
        std::cerr << "cpu/blas/additional-level2 tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All cpu/blas/additional-level2 tests passed!" << std::endl;
    return TEST_PASS;
}