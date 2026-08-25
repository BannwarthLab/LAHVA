#include "test_common.h"
#include "array_utils.hpp"

using namespace lahva::gpu;
using lahva::Shape;
using lahva::CudaRuntime;
using lahva::CudaHostAllocator;
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

    if (!check(sum_, 0.0, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(M, (T)1.0);

    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, 1, (T)1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    MatrixVectorProduct(cudart, A, x, y, "N", (T)1.0, (T)1.0);
    y.copy2host(cudart);
    A.copy2host(cudart);
    cudart.synchronize();

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    A = Matrix<T>(Shape(N,M), (T)1.0);

    MatrixVectorProduct(cudart, A, x, y, "T"); //Check transpose
    y.copy2host(cudart);
    cudart.synchronize();
    sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;

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
    if (!check(sum_, T(0.0, 0.0), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(M, T{0.0,1.0});

    MatrixVectorProduct("N", 1.0, A, x, 1, T{0.0, -1.0}, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    sum_ = y.sum();

    if (!check(sum_, T(M*1.0, 0.0), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    y = Vector<T>(M, T{0.0,1.0});
    MatrixVectorProduct(A, x, y, "N", 1.0, T{0.0, -1.0});
    y.copy2host(cudart);
    cudart.synchronize();
    sum_ = y.sum();

    if (!check(sum_, T(M*1.0,0.0), check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    A = Matrix<T>(Shape(N,M), T(0.0, 1.0));
    x = Vector<T>(N, T(1.0, 0.0));
    A.updateGPUvalues(cudart);
    x.updateGPUvalues(cudart);
    MatrixVectorProduct(A, x, y, "C", T(0.0,1.0)); //Check complex conjugate
    y.copy2host(cudart);
    sum_ = y.sum();

    if (!check(sum_, T(N*M*1.0, 0.0), check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;

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

    if (!check(sum_, 0.0, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(M, (T)1.0);

    SymMatrixVectorProduct(cudart, (T)1.0, A, x, 1, (T)1.0, y, 1);
    y.copy2host(cudart);
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    SymMatrixVectorProduct(cudart, A, x, y, (T)1.0, (T)1.0);
    y.copy2host(cudart);
    A.copy2host(cudart);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

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

    if (!check(y.data(), vres.data(), 3, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(3, (T)1.0);

    SymMatrixVectorProduct(cudart, (T)1.0, A, x, 1, (T)1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<T>({(T)25.0, (T)27.0, (T)27.0});

    if (!check(y.data(), vres.data(), 3, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    SymMatrixVectorProduct(cudart, A, x, y, (T)2.0, (T)0.0);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<T>({(T)48.0, (T)52.0, (T)52.0});

    if (!check(y.data(), vres.data(), 3, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

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

    if (!check(y.data(), vres.data(), 3, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(3, (T)1.0);

    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, 1, (T)1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<T>({(T)2.0, (T)9.0, (T)27.0});

    if (!check(y.data(), vres.data(), 3, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    MatrixVectorProduct(cudart, A, x, y, "N", (T)2.0, (T)0.0);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<T>({(T)2.0, (T)16.0, (T)52.0});

    if (!check(y.data(), vres.data(), 3, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

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

    if (!check(y.data(), vres.data(), 3, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    y = Vector<T>(3, (T)1.0);

    SymMatrixVectorProduct(cudart, (T)1.0, A, x, 1, (T)1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<T>({(T)25.0, (T)27.0, (T)27.0});

    if (!check(y.data(), vres.data(), 3, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    SymMatrixVectorProduct(cudart, A, x, y, (T)2.0, (T)0.0);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<T>({(T)48.0, (T)52.0, (T)52.0});

    if (!check(y.data(), vres.data(), 3, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

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

    if (!check(A.data(), vres.data(), 6, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    // Test with alpha = 2.0
    A = Matrix<T>(Shape(3, 2), (T)0.0);
    OuterVectorProduct(cudart, x, y, A, 1, 1, (T)2.0);
    A.copy2host(cudart);
    cudart.synchronize();

    vres = Vector<T>({(T)4.0, (T)8.0, (T)12.0, (T)6.0, (T)12.0, (T)18.0});

    if (!check(A.data(), vres.data(), 6, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    try {
        Matrix<T> A(Shape(4,2),0.0);
        OuterVectorProduct(cudart, x, y, A);
        std::cerr << check_msg(get_type_name<T>(), "check 3") << std::endl;
        return TEST_FAIL; // Should not reach here
    }
    catch (std::invalid_argument& e) {
        // Expected exception caught
    }
    catch (const std::exception& e) {
        std::cerr << check_msg(get_type_name<T>(), "check 4") << std::endl;
        return TEST_FAIL; // Unexpected exception type
    }



    return TEST_PASS;
}

template <typename T>
int test_gpu_blockdiag_gemv_simple(CudaRuntime& cudart) {

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 3, 2, 4}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {5, 7, 6, 8}));
    BlockDiagMatrix<T> A(blocks);

    Vector<T, CudaHostAllocator<T>> x(4);
    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;
    x[3] = 4.0;

    Vector<T, CudaHostAllocator<T>> y(4, 0.0);

    MatrixVectorProduct(cudart, "N", static_cast<T>(1.0), A, x, static_cast<T>(0.0), y);
    y.copy2host(cudart);
    cudart.synchronize();

    if (!check(y[0], static_cast<T>(5.0), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(y[1], static_cast<T>(11.0), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(y[2], static_cast<T>(39.0), check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check(y[3], static_cast<T>(53.0), check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_blockdiag_gemv_with_beta(CudaRuntime& cudart) {

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    BlockDiagMatrix<T> A(blocks);

    Vector<T, CudaHostAllocator<T>> x(2, static_cast<T>(1.0));
    Vector<T, CudaHostAllocator<T>> y(2, static_cast<T>(2.0));

    MatrixVectorProduct(cudart, "N", static_cast<T>(1.0), A, x, static_cast<T>(2.0), y);
    y.copy2host(cudart);
    cudart.synchronize();

    if (!check(y[0], static_cast<T>(8.0), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(y[1], static_cast<T>(10.0), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_blockdiag_gemv_varying_blocks(CudaRuntime& cudart) {

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{1, 1}, {2}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{1, 1}, {3}));
    BlockDiagMatrix<T> A(blocks);

    Vector<T, CudaHostAllocator<T>> x(4);
    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;
    x[3] = 4.0;

    Vector<T, CudaHostAllocator<T>> y(4, static_cast<T>(0.0));

    MatrixVectorProduct(cudart, "N", static_cast<T>(1.0), A, x, static_cast<T>(0.0), y);
    y.copy2host(cudart);
    cudart.synchronize();

    if (!check(y[0], static_cast<T>(2.0), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(y[1], static_cast<T>(11.0), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(y[2], static_cast<T>(16.0), check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check(y[3], static_cast<T>(12.0), check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_blockdiag_gemv_transpose(CudaRuntime& cudart) {

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 3, 2, 4}));
    BlockDiagMatrix<T> A(blocks);

    Vector<T, CudaHostAllocator<T>> x(2);
    x[0] = 1.0;
    x[1] = 2.0;

    Vector<T, CudaHostAllocator<T>> y(2, static_cast<T>(0.0));

    MatrixVectorProduct(cudart, "T", static_cast<T>(1.0), A, x, static_cast<T>(0.0), y);
    y.copy2host(cudart);
    cudart.synchronize();

    if (!check(y[0], static_cast<T>(7.0), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(y[1], static_cast<T>(10.0), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_blockdiag_gemv_transpose_with_beta(CudaRuntime& cudart) {

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    BlockDiagMatrix<T> A(blocks);

    Vector<T, CudaHostAllocator<T>> x(2, static_cast<T>(1.0));
    Vector<T, CudaHostAllocator<T>> y(2, static_cast<T>(2.0));

    MatrixVectorProduct(cudart, "T", static_cast<T>(1.0), A, x, static_cast<T>(2.0), y);
    y.copy2host(cudart);
    cudart.synchronize();

    if (!check(y[0], static_cast<T>(7.0), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(y[1], static_cast<T>(11.0), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_blockdiag_gemv_transpose_multiple_blocks(CudaRuntime& cudart) {

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 3, 2, 4}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {5, 7, 6, 8}));
    BlockDiagMatrix<T> A(blocks);

    Vector<T, CudaHostAllocator<T>> x(4);
    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;
    x[3] = 4.0;

    Vector<T, CudaHostAllocator<T>> y(4, static_cast<T>(0.0));

    MatrixVectorProduct(cudart, "T", static_cast<T>(1.0), A, x, static_cast<T>(0.0), y);
    y.copy2host(cudart);
    cudart.synchronize();

    if (!check(y[0], static_cast<T>(7.0), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(y[1], static_cast<T>(10.0), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(y[2], static_cast<T>(43.0), check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check(y[3], static_cast<T>(50.0), check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// BlockMatrix GEMV Tests (via cuSPARSE)
// ============================================================================

template <typename T>
int test_gpu_blockmatrix_gemv_bsr_format(CudaRuntime& cudart) {

    // Create a BlockMatrix with uniform 3x3 blocks (BSR format)
    BlockMatrix<T> A;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{3, 3}, {
        (T)1, (T)2, (T)3,
        (T)4, (T)5, (T)6,
        (T)7, (T)8, (T)9
    });
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, {
        (T)2, (T)4, (T)6,
        (T)8, (T)10, (T)12,
        (T)14, (T)16, (T)18
    });

    A.set_block(0, 0, block1);
    A.set_block(3, 3, block2);

    Vector<T, CudaHostAllocator<T>> x(6);
    x[0] = (T)1; x[1] = (T)2; x[2] = (T)3;
    x[3] = (T)1; x[4] = (T)1; x[5] = (T)1;

    Vector<T, CudaHostAllocator<T>> y(6, (T)0);

    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, (T)0.0, y);

    y.copy2host(cudart);
    cudart.synchronize();

    if (!check(y[0], (T)30.0, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(y[1], (T)36.0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(y[2], (T)42.0, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check(y[3], (T)24.0, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;
    if (!check(y[4], (T)30.0, check_msg(get_type_name<T>(), "check 5"))) return TEST_FAIL;
    if (!check(y[5], (T)36.0, check_msg(get_type_name<T>(), "check 6"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_blockmatrix_gemv_bsr_with_beta(CudaRuntime& cudart) {

    // Create a BlockMatrix with uniform 2x2 blocks
    BlockMatrix<T> A;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, {
        (T)1, (T)2,
        (T)3, (T)4
    });

    A.set_block(0, 0, block1);

    Vector<T, CudaHostAllocator<T>> x(2, (T)1);
    Vector<T, CudaHostAllocator<T>> y(2, (T)2);

    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, (T)2.0, y);

    y.copy2host(cudart);
    cudart.synchronize();

    if (!check(y[0], (T)8.0, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(y[1], (T)10.0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_blockmatrix_gemv_csr_format(CudaRuntime& cudart) {

    // Create a BlockMatrix with non-uniform blocks (CSR format)
    BlockMatrix<T> A;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 3}, {
        (T)1, (T)2, (T)3,
        (T)4, (T)5, (T)6
    });
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 2}, {
        (T)1, (T)2,
        (T)3, (T)4,
        (T)5, (T)6
    });

    A.set_block(0, 0, block1);
    A.set_block(2, 3, block2);

    Vector<T, CudaHostAllocator<T>> x(5);
    x[0] = (T)1; x[1] = (T)2; x[2] = (T)3; x[3] = (T)1; x[4] = (T)1;

    Vector<T, CudaHostAllocator<T>> y(5, (T)0);

    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, (T)0.0, y);

    y.copy2host(cudart);
    cudart.synchronize();

    if (!check(y[0], (T)22.0, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(y[1], (T)28.0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(y[2], (T)5.0, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check(y[3], (T)7.0, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;
    if (!check(y[4], (T)9.0, check_msg(get_type_name<T>(), "check 5"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_blockmatrix_gemv_csr_with_beta(CudaRuntime& cudart) {

    // Create a BlockMatrix with non-uniform blocks and beta scaling
    BlockMatrix<T> A;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, {
        (T)1, (T)2,
        (T)3, (T)4
    });
    Matrix<T, CudaHostAllocator<T>> block2(Shape{1, 2}, {
        (T)2, (T)3
    });

    A.set_block(0, 0, block1);
    A.set_block(2, 2, block2);

    Vector<T, CudaHostAllocator<T>> x(4, (T)1);
    Vector<T, CudaHostAllocator<T>> y(3, (T)2);

    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, (T)2.0, y);

    y.copy2host(cudart);
    cudart.synchronize();

    if (!check(y[0], (T)8.0, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(y[1], (T)10.0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(y[2], (T)9.0, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_blockmatrix_gemv_transpose_bsr(CudaRuntime& cudart) {

    // Create a BlockMatrix with uniform 2x2 blocks and test transpose
    BlockMatrix<T> A;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, {
        (T)1, (T)2,
        (T)3, (T)4
    });

    A.set_block(0, 0, block1);

    Vector<T, CudaHostAllocator<T>> x(2, (T)1);
    Vector<T, CudaHostAllocator<T>> y(2, (T)0);

    MatrixVectorProduct(cudart, "T", (T)1.0, A, x, (T)0.0, y);

    y.copy2host(cudart);
    cudart.synchronize();

    if (!check(y[0], (T)3.0, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(y[1], (T)7.0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_blockmatrix_gemv_transpose_csr(CudaRuntime& cudart) {

    // Create a BlockMatrix with non-uniform blocks and test transpose
    BlockMatrix<T> A;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 3}, {
        (T)1, (T)2, (T)3,
        (T)4, (T)5, (T)6
    });

    A.set_block(0, 0, block1);

    Vector<T, CudaHostAllocator<T>> x(2, (T)1);
    Vector<T, CudaHostAllocator<T>> y(3, (T)0);

    MatrixVectorProduct(cudart, "T", (T)1.0, A, x, (T)0.0, y);

    y.copy2host(cudart);
    cudart.synchronize();

    if (!check(y[0], (T)3.0, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(y[1], (T)7.0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(y[2], (T)11.0, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Main
// ============================================================================

int main(){
    int total_failures = 0;
    CudaRuntime cudart;

    // Basic GEMV tests
    total_failures += test_gemv_zero_v_cpp<double>(cudart);
    total_failures += test_gemv_zero_v_cpp<float>(cudart);

    // SymV tests
    total_failures += test_symv_zero_v_cpp<double>(cudart);
    total_failures += test_symv_zero_v_cpp<float>(cudart);
    total_failures += test_gemv_v_cpp<double>(cudart);
    total_failures += test_gemv_v_cpp<float>(cudart);
    total_failures += test_symv_v_cpp<double>(cudart);
    total_failures += test_symv_v_cpp<float>(cudart);
    total_failures += test_spmv_v_cpp<double>(cudart);
    total_failures += test_spmv_v_cpp<float>(cudart);

    // Complex GEMV tests
    total_failures += test_complex_gemv_zero_v_cpp<complex_double>(cudart);
    total_failures += test_complex_gemv_zero_v_cpp<complex_float>(cudart);

    // Outer product tests
    total_failures += test_outer_product<double>(cudart);
    total_failures += test_outer_product<float>(cudart);

    // BlockDiagMatrix GEMV tests
    total_failures += test_gpu_blockdiag_gemv_simple<double>(cudart);
    total_failures += test_gpu_blockdiag_gemv_with_beta<double>(cudart); 
    total_failures += test_gpu_blockdiag_gemv_varying_blocks<double>(cudart);
    total_failures += test_gpu_blockdiag_gemv_transpose<double>(cudart);
    total_failures += test_gpu_blockdiag_gemv_transpose_with_beta<double>(cudart);
    total_failures += test_gpu_blockdiag_gemv_transpose_multiple_blocks<double>(cudart);
    total_failures += test_gpu_blockdiag_gemv_simple<float>(cudart);
    total_failures += test_gpu_blockdiag_gemv_with_beta<float>(cudart);
    total_failures += test_gpu_blockdiag_gemv_varying_blocks<float>(cudart);
    total_failures += test_gpu_blockdiag_gemv_transpose<float>(cudart);
    total_failures += test_gpu_blockdiag_gemv_transpose_with_beta<float>(cudart);
    total_failures += test_gpu_blockdiag_gemv_transpose_multiple_blocks<float>(cudart);

    // BlockMatrix GEMV with BSR format (uniform blocks)
    total_failures += test_gpu_blockmatrix_gemv_bsr_format<double>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_bsr_format<float>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_bsr_with_beta<double>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_bsr_with_beta<float>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_transpose_bsr<double>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_transpose_bsr<float>(cudart);

    // BlockMatrix GEMV with CSR format (non-uniform blocks)
    total_failures += test_gpu_blockmatrix_gemv_csr_format<double>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_csr_format<float>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_csr_with_beta<double>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_csr_with_beta<float>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_transpose_csr<double>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_transpose_csr<float>(cudart);

    if (total_failures > 0) {
        std::cerr << "gpu/blas/level2 tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All gpu/blas/level2 tests passed!" << std::endl;
    return TEST_PASS;
};