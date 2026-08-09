#include "test_common.h"
#include "array_utils.hpp"

#define M 10
#define N 5 

using namespace lahva::cpu;
using lahva::Shape;

float vf[9] = {1.0, 4.0, 5.0, 0.0, 2.0, 6.0, 0.0, 0.0, 3.0};
double vd[9] = {1.0, 4.0, 5.0, 0.0, 2.0, 6.0, 0.0, 0.0, 3.0};
float *pf = vf;
double *pd = vd;

template<typename T> T* get_test_data();
template<> double* get_test_data<double>() { return pd; }
template<> float* get_test_data<float>() { return pf; }

template<typename T>
int test_gemv_zero_v_cpp(){
    Shape s(M,N);
    Matrix<T> A(s, 1.0);
    Vector<T> x(N, 0.0);
    Vector<T> y(M, 1.0);

    MatrixVectorProduct("N", 1.0, A, x, 1, 0.0, y, 1);

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(M, 1.0);

    MatrixVectorProduct("N", 1.0, A, x, 1, 1.0, y, 1);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    MatrixVectorProduct(A, x, y, "N", 1.0, 1.0);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    A = Matrix<T>(Shape(N,M), 1.0);

    MatrixVectorProduct(A, x, y, "T"); //Check transpose

    sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, make_check_msg(__func__, get_type_name<T>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
}
template<typename T>
int test_complex_gemv_zero_v_cpp(){
    Shape s(M,N);
    Matrix<T> A(s, 1.0);
    Vector<T> x(N, 0.0);
    Vector<T> y(M, 1.0);
    MatrixVectorProduct("N", 1.0, A, x, 1, 0.0, y, 1);
    auto sum_ = y.sum();
    if (!check(sum_, T(0.0, 0.0), make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(M, T{0.0,1.0});
    std::cout << y.sum() << std::endl;

    MatrixVectorProduct("N", 1.0, A, x, 1, T{0.0, -1.0}, y, 1);

    sum_ = y.sum();

    if (!check(sum_, T(M*1.0, 0.0), make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;
    y = Vector<T>(M, T{0.0,1.0});
    MatrixVectorProduct(A, x, y, "N", 1.0, T{0.0, -1.0});

    sum_ = y.sum();
    std::cout << y.sum() << std::endl;
    if (!check(sum_, T(M*1.0,0.0), make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    A = Matrix<T>(Shape(N,M), T(0.0, 1.0));
    x = Vector<T>(N, 1.0);
    MatrixVectorProduct(A, x, y, "C", T(0.0,1.0)); //Check complex conjugate

    sum_ = y.sum();

    if (!check(sum_, T(N*M*1.0, 0.0), make_check_msg(__func__, get_type_name<T>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
}


template<typename T>
int test_gemv_v_cpp(){
    Shape s(3,3);
    Matrix<T> A(s, get_test_data<T>(), false);
    Vector<T> x({1.0, 2.0, 3.0});
    Vector<T> y(3, 0.0);

    MatrixVectorProduct("N", 1.0, A, x, 1, 0.0, y, 1);

    Vector<T> vres({1.0, 8.0, 26.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(3, 1.0);

    MatrixVectorProduct("N", 1.0, A, x, 1, 1.0, y, 1);

    vres = Vector<T>({2.0, 9.0, 27.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    MatrixVectorProduct(A, x, y, "N", 2.0, 0.0);

    vres = Vector<T>({2.0, 16.0, 52.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template<typename T>
int test_symv_zero_v_cpp(){
    Shape s(M,M);
    Matrix<T> A(s, 1.0);
    Vector<T> x(M, 0.0);
    Vector<T> y(M, 1.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 0.0, y, 1);

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(M, 1.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 1.0, y, 1);
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    SymMatrixVectorProduct(A, x, y, 1.0, 1.0);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template<typename T>
int test_symv_v_cpp(){
    Shape s(3,3);
    Matrix<T> A(s, get_test_data<T>(), false);
    Vector<T> x({1.0, 2.0, 3.0});
    Vector<T> y(3, 0.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 0.0, y, 1);

    Vector<T> vres({24.0, 26.0, 26.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(3, 1.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 1.0, y, 1);

    vres = Vector<T>({25.0, 27.0, 27.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    SymMatrixVectorProduct(A, x, y, 2.0, 0.0);

    vres = Vector<T>({48.0, 52.0, 52.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template<typename T>
int test_symv_zero_lowtri_cpp(){
    Shape s(M,M);
    LowTriMatrix<T> A(s, 1.0);
    Vector<T> x(M, 0.0);
    Vector<T> y(M, 1.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 0.0, y, 1);

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(M, 1.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 1.0, y, 1);
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    SymMatrixVectorProduct(A, x, y, 1.0, 1.0);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template<typename T>
int test_symv_lowtri_cpp(){
    T* vdtri_ = new T[6] {1.0, 4.0, 5.0, 2.0, 6.0, 3.0};
    LowTriMatrix<T> A(3, vdtri_, std::is_same_v<T, float>);
    Vector<T> x({1.0, 2.0, 3.0});
    Vector<T> y(3, 0.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 0.0, y, 1);

    Vector<T> vres({24.0, 26.0, 26.0});
    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;
    y = Vector<T>(3, 1.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 1.0, y, 1);

    vres = Vector<T>({25.0, 27.0, 27.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    SymMatrixVectorProduct(A, x, y, 2.0, 0.0);

    vres = Vector<T>({48.0, 52.0, 52.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template<typename T>
int test_tpmv_cpp(){
    T* vdtri_ = new T[6] {1.0, 4.0, 5.0, 2.0, 6.0, 3.0};
    LowTriMatrix<T> A(3, vdtri_);
    Vector<T> x({1.0, 2.0, 3.0});

    LowTriMatrixVectorProduct("N", CblasNonUnit, A, x, 1);

    Vector<T> vres({1.0, 8.0, 26.0});

    if (!check(x.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    x = Vector<T>({1.0, 2.0, 3.0});
    LowTriMatrixVectorProduct("N", CblasNonUnit, A, x, 1);

    if (!check(x.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    x = Vector<T>({1.0, 2.0, 3.0});
    LowTriMatrixVectorProduct("T", CblasNonUnit, A, x, 1);

    vres = Vector<T>({24.0, 22.0, 9.0});

    if (!check(x.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}



template<typename T>
int test_gemv_zero_v_c(){
    Shape s(M,N);
    Matrix<T> A(s, 1.0);
    Vector<T> x(N, 0.0);
    Vector<T> y(M, 1.0);

    MatrixVectorProduct("N", M, N, 1.0, A.data(), x.data(), 1, 0.0, y.data(), 1);

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(M, 1.0);

    MatrixVectorProduct("N", M, N, 1.0, A.data(), x.data(), 1, 1.0, y.data(), 1);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    MatrixVectorProduct(M, N, A.data(), x.data(), y.data(), "N", 1.0, 1.0);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    A = Matrix<T>(Shape(N,M), 1.0);

    MatrixVectorProduct(N, M, A.data(), x.data(), y.data(), "T"); //Check transpose

    sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, make_check_msg(__func__, get_type_name<T>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
}

template<typename T>
int test_gemv_v_c(){
    Shape s(3,3);
    Matrix<T> A(s, get_test_data<T>(), false);
    Vector<T> x({1.0, 2.0, 3.0});
    Vector<T> y(3, 0.0);

    MatrixVectorProduct("N", 3, 3, 1.0, A.data(), x.data(), 1, 0.0, y.data(), 1);

    Vector<T> vres({1.0, 8.0, 26.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;
    return TEST_PASS;

    y = Vector<T>(M, 1.0);

    MatrixVectorProduct("N", 3, 3, 1.0, A.data(), x.data(), 1, 1.0, y.data(), 1);

    vres = Vector<T>({2.0, 9.0, 27.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    MatrixVectorProduct(3, 3, A.data(), x.data(), y.data(), "N", 2.0, 0.0);

    vres = Vector<T>({2.0, 16.0, 52.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template<typename T>
int test_symv_zero_v_c(){
    Shape s(M,M);
    Matrix<T> A(s, 1.0);
    Vector<T> x(M, 0.0);
    Vector<T> y(M, 1.0);

    SymMatrixVectorProduct(M, (T)1.0, A.data(), x.data(), 1, (T)0.0, y.data(), 1);

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(M, 1.0);

    SymMatrixVectorProduct(M, 1.0, A.data(), x.data(), 1, 1.0, y.data(), 1);
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    SymMatrixVectorProduct(M, A.data(), x.data(), y.data(), 1.0, 1.0);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template<typename T>
int test_symv_v_c(){
    Shape s(3,3);
    Matrix<T> A(s, get_test_data<T>(), false);
    Vector<T> x({1.0, 2.0, 3.0});
    Vector<T> y(3, 0.0);

    SymMatrixVectorProduct(3, 1.0, A.data(), x.data(), 1, 0.0, y.data(), 1);

    Vector<T> vres({24.0, 26.0, 26.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;
    return TEST_PASS;

    y = Vector<T>(M, 1.0);

    SymMatrixVectorProduct(3, 1.0, A.data(), x.data(), 1, 1.0, y.data(), 1);

    vres = Vector<T>({25.0, 27.0, 27.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    SymMatrixVectorProduct(3, A.data(), x.data(), y.data(), 2.0, 0.0);

    vres = Vector<T>({48.0, 52.0, 52.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_ger(){
    Vector<T> x({1.0, 2.0, 3.0});
    Vector<T> y({2.0, 3.0});
    Matrix<T> A(Shape(3, 2), 0.0);

    OuterVectorProduct(x, y, A);

    Vector<T> vres({2.0, 4.0, 6.0, 3.0, 6.0, 9.0});

    if (!check(A.data(), vres.data(), 6, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    // Test with alpha = 2.0
    A = Matrix<T>(Shape(3, 2), 0.0);
    OuterVectorProduct(x, y, A, 1, 1, 2.0);

    vres = Vector<T>({4.0, 8.0, 12.0, 6.0, 12.0, 18.0});

    if (!check(A.data(), vres.data(), 6, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    try {
        Matrix<T> A(Shape(4,2),0.0);
        OuterVectorProduct(x, y, A);
        return TEST_FAIL; // Should not reach here
        std::cerr << "Error: No exception thrown for dimension mismatch in outer product test." << std::endl;
    }
    catch (std::invalid_argument& e) {
        // Expected exception caught
    }
    catch (const std::exception& e) {
        return TEST_FAIL; // Unexpected exception type
        std::cerr << "Error: Unexpected exception type caught in outer product dimension mismatch test: " << e.what() << std::endl;
    }

    return TEST_PASS;
}

// ============================================================================
// Main
// ============================================================================

int main(){
    int total_failures = 0;
    total_failures += test_gemv_zero_v_cpp<double>();
    total_failures += test_gemv_zero_v_cpp<float>();
    total_failures += test_complex_gemv_zero_v_cpp<complex_double>();
    total_failures += test_complex_gemv_zero_v_cpp<complex_float>();
    total_failures += test_gemv_v_cpp<double>();
    total_failures += test_gemv_v_cpp<float>();
    total_failures += test_symv_zero_v_cpp<double>();
    total_failures += test_symv_zero_v_cpp<float>();
    total_failures += test_symv_v_cpp<double>();
    total_failures += test_symv_v_cpp<float>();
    total_failures += test_symv_zero_lowtri_cpp<double>();
    total_failures += test_symv_zero_lowtri_cpp<float>();
    total_failures += test_symv_lowtri_cpp<double>();
    total_failures += test_symv_lowtri_cpp<float>();
    total_failures += test_tpmv_cpp<double>();
    total_failures += test_tpmv_cpp<float>();
    total_failures += test_gemv_zero_v_c<double>();
    total_failures += test_gemv_zero_v_c<float>();
    total_failures += test_gemv_v_c<double>();
    total_failures += test_gemv_v_c<float>();
    total_failures += test_symv_zero_v_c<double>();
    total_failures += test_symv_zero_v_c<float>();
    total_failures += test_symv_v_c<double>();
    total_failures += test_symv_v_c<float>();
    total_failures += test_ger<double>();
    total_failures += test_ger<float>();

    if (total_failures > 0) {
        std::cerr << "cpu/blas/level2 tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All cpu/blas/level2 tests passed!" << std::endl;
    return TEST_PASS;
};
