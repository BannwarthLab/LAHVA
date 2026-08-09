#include "test_common.h"
#include "array_utils.hpp"

using namespace lahva::gpu;
using lahva::Shape;
using lahva::CudaRuntime;
#define M 10
#define N 5

template<typename T>
using MLow = LowTriMatrix<T, lahva::StdAllocator<T>, lahva::CudaDeviceAllocator<T>>;

float vf[9] = {1.0, 4.0, 5.0, 0.0, 2.0, 6.0, 0.0, 0.0, 3.0};
double vd[9] = {1.0, 4.0, 5.0, 0.0, 2.0, 6.0, 0.0, 0.0, 3.0};
float *pf = vf;
double *pd = vd;

template<typename T> T* get_test_data();
template<> double* get_test_data<double>() { return pd; }
template<> float* get_test_data<float>() { return pf; }

template <typename T>
int test_gemv_zero_v_cpp(CudaRuntime& cudart){
    Shape s(M,N);
    Matrix<T> A(s, (T)1.0);
    Vector<T> x(N, (T)0.0);
    Vector<T> y(M, (T)1.0);

    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, 1, (T)0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(M, (T)1.0);

    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, 1, (T)1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    MatrixVectorProduct(cudart, A, x, y, "N", (T)1.0, (T)1.0);
    y.copy2host(cudart);
    A.copy2host(cudart);
    cudart.synchronize();

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    A = Matrix<T>(Shape(N,M), (T)1.0);

    MatrixVectorProduct(cudart, A, x, y, "T"); //Check transpose
    y.copy2host(cudart);
    cudart.synchronize();
    sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, make_check_msg(__func__, get_type_name<T>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
};

template<typename T>
int test_complex_gemv_zero_v_cpp(CudaRuntime& cudart){
    Shape s(M,N);
    Matrix<T> A(s, T(1.0,0.0));
    Vector<T> x(N, T(0.0, 0.0));
    Vector<T> y(M, T(1.0, 0.0));
    MatrixVectorProduct("N", 1.0, A, x, 1, 0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    auto sum_ = y.sum();
    if (!check(sum_, T(0.0, 0.0), make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(M, T{0.0,1.0});

    MatrixVectorProduct("N", 1.0, A, x, 1, T{0.0, -1.0}, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    sum_ = y.sum();

    if (!check(sum_, T(M*1.0, 0.0), make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;
    y = Vector<T>(M, T{0.0,1.0});
    MatrixVectorProduct(A, x, y, "N", 1.0, T{0.0, -1.0});
    y.copy2host(cudart);
    cudart.synchronize();
    sum_ = y.sum();

    if (!check(sum_, T(M*1.0,0.0), make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    A = Matrix<T>(Shape(N,M), T(0.0, 1.0));
    x = Vector<T>(N, T(1.0, 0.0));
    A.updateGPUvalues(cudart);
    x.updateGPUvalues(cudart);
    MatrixVectorProduct(A, x, y, "C", T(0.0,1.0)); //Check complex conjugate
    y.copy2host(cudart);
    sum_ = y.sum();

    if (!check(sum_, T(N*M*1.0, 0.0), make_check_msg(__func__, get_type_name<T>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_symv_zero_v_cpp(CudaRuntime& cudart){
    Shape s(M,M);
    Matrix<T> A(s, (T)1.0);
    Vector<T> x(M, (T)0.0);
    Vector<T> y(M, (T)1.0);

    SymMatrixVectorProduct(cudart, (T)1.0, A, x, 1, (T)0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(M, (T)1.0);

    SymMatrixVectorProduct(cudart, (T)1.0, A, x, 1, (T)1.0, y, 1);
    y.copy2host(cudart);
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    SymMatrixVectorProduct(cudart, A, x, y, (T)1.0, (T)1.0);
    y.copy2host(cudart);
    A.copy2host(cudart);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_symv_v_cpp(CudaRuntime& cudart){
    Shape s(3,3);
    Matrix<T> A(s, get_test_data<T>(), false);
    Vector<T> x({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> y(3, (T)0.0);

    SymMatrixVectorProduct(cudart, (T)1.0, A, x, 1, (T)0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    Vector<T> vres({(T)24.0, (T)26.0, (T)26.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(3, (T)1.0);

    SymMatrixVectorProduct(cudart, (T)1.0, A, x, 1, (T)1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<T>({(T)25.0, (T)27.0, (T)27.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    SymMatrixVectorProduct(cudart, A, x, y, (T)2.0, (T)0.0);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<T>({(T)48.0, (T)52.0, (T)52.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gemv_v_cpp(CudaRuntime& cudart){
    Shape s(3,3);
    Matrix<T> A(s, get_test_data<T>(), false);
    Vector<T> x({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> y(3, (T)0.0);

    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, 1, (T)0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    Vector<T> vres({(T)1.0, (T)8.0, (T)26.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(3, (T)1.0);

    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, 1, (T)1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<T>({(T)2.0, (T)9.0, (T)27.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    MatrixVectorProduct(cudart, A, x, y, "N", (T)2.0, (T)0.0);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<T>({(T)2.0, (T)16.0, (T)52.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_spmv_v_cpp(CudaRuntime& cudart){
    T* vdtri_ = new T[6] {(T)1.0, (T)4.0, (T)5.0, (T)2.0, (T)6.0, (T)3.0};
    MLow<T> A(3, vdtri_);
    Vector<T> x({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> y(3, (T)0.0);

    SymMatrixVectorProduct(cudart, (T)1.0, A, x, 1, (T)0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    Vector<T> vres({(T)24.0, (T)26.0, (T)26.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(3, (T)1.0);

    SymMatrixVectorProduct(cudart, (T)1.0, A, x, 1, (T)1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<T>({(T)25.0, (T)27.0, (T)27.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    SymMatrixVectorProduct(cudart, A, x, y, (T)2.0, (T)0.0);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<T>({(T)48.0, (T)52.0, (T)52.0});

    if (!check(y.data(), vres.data(), 3, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_outer_product(CudaRuntime& cudart){
    Vector<T> x({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> y({(T)2.0, (T)3.0});
    Matrix<T> A(Shape(3, 2), (T)0.0);

    OuterVectorProduct(cudart, x, y, A);
    A.copy2host(cudart);
    cudart.synchronize();

    Vector<T> vres({(T)2.0, (T)4.0, (T)6.0, (T)3.0, (T)6.0, (T)9.0});

    if (!check(A.data(), vres.data(), 6, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    // Test with alpha = 2.0
    A = Matrix<T>(Shape(3, 2), (T)0.0);
    OuterVectorProduct(cudart, x, y, A, 1, 1, (T)2.0);
    A.copy2host(cudart);
    cudart.synchronize();

    vres = Vector<T>({(T)4.0, (T)8.0, (T)12.0, (T)6.0, (T)12.0, (T)18.0});

    if (!check(A.data(), vres.data(), 6, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    try {
        Matrix<T> A(Shape(4,2),0.0);
        OuterVectorProduct(cudart, x, y, A);
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
    CudaRuntime cudart;

    total_failures += test_gemv_zero_v_cpp<double>(cudart);
    total_failures += test_gemv_zero_v_cpp<float>(cudart);

    total_failures += test_symv_zero_v_cpp<double>(cudart);
    total_failures += test_symv_zero_v_cpp<float>(cudart);

    total_failures += test_gemv_v_cpp<double>(cudart);
    total_failures += test_gemv_v_cpp<float>(cudart);

    total_failures += test_symv_v_cpp<double>(cudart);
    total_failures += test_symv_v_cpp<float>(cudart);

    total_failures += test_spmv_v_cpp<double>(cudart);
    total_failures += test_spmv_v_cpp<float>(cudart);

    total_failures += test_complex_gemv_zero_v_cpp<complex_double>(cudart);
    total_failures += test_complex_gemv_zero_v_cpp<complex_float>(cudart);

    total_failures += test_outer_product<double>(cudart);
    total_failures += test_outer_product<float>(cudart);

    if (total_failures > 0) {
        std::cerr << "gpu/blas/level2 tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All gpu/blas/level2 tests passed!" << std::endl;
    return TEST_PASS;
};