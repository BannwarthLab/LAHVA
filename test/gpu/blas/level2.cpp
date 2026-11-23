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

// MLow frees data it owns through StdAllocator, so data must not come from new[]
template<typename T>
T* make_lowtri_data() {
    T* data = lahva::StdAllocator<T>().allocate(6);
    const T values[6] = {(T)1.0, (T)4.0, (T)5.0, (T)2.0, (T)6.0, (T)3.0};
    std::copy(values, values + 6, data);
    return data;
}

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
    T* vdtri_ = make_lowtri_data<T>();
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

// Checks y = 2 * op(A) * x + 3 * y against a naive reference, independent of BLAS
template <typename T>
int check_blockdiag_gemv(CudaRuntime& cudart, const BlockDiagMatrix<T> &A, const char *Ta) {

    const bool trans = Ta[0] != 'N';
    const size_t n_in = trans ? A.shape().first : A.shape().second;
    const size_t n_out = trans ? A.shape().second : A.shape().first;

    Vector<T, CudaHostAllocator<T>> x(n_in);
    for (size_t i = 0; i < n_in; i++) {
        x[i] = static_cast<T>(i + 1);
    }
    Vector<T, CudaHostAllocator<T>> y(n_out, static_cast<T>(1));

    std::vector<T> expected(n_out, static_cast<T>(3));
    for (size_t b = 0; b < A.num_blocks(); b++) {
        Shape shape = A.get_block_shape(b);
        for (size_t i = 0; i < shape.first; i++) {
            for (size_t j = 0; j < shape.second; j++) {
                const size_t row = A.get_row_offsets()[b] + i;
                const size_t col = A.get_col_offsets()[b] + j;
                expected[trans ? col : row] += static_cast<T>(2) * A.get_block(b)(i, j) * x[trans ? row : col];
            }
        }
    }

    MatrixVectorProduct(cudart, Ta, static_cast<T>(2), A, x, static_cast<T>(3), y);
    y.copy2host(cudart);
    cudart.synchronize();

    const std::string desc = std::string("Ta=") + Ta;
    if (!check(y.data(), expected.data(), static_cast<int>(n_out), check_msg(get_type_name<T>(), desc))) return TEST_FAIL;

    return TEST_PASS;
}

// Row and column offsets differ between the blocks, so swapping them changes the result
template <typename T>
int test_gpu_blockdiag_gemv_multi_block_transpose(CudaRuntime& cudart, const char *Ta) {

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    T value = static_cast<T>(1);
    for (const Shape &shape : {Shape{2, 3}, Shape{3, 1}, Shape{1, 2}, Shape{1, 2}}) {
        Matrix<T, CudaHostAllocator<T>> block(shape);
        for (size_t i = 0; i < shape.first; i++) {
            for (size_t j = 0; j < shape.second; j++) {
                block(i, j) = value++;
            }
        }
        blocks.push_back(block);
    }
    BlockDiagMatrix<T> A(blocks);

    return check_blockdiag_gemv(cudart, A, Ta);
}

template <typename T>
Matrix<T, CudaHostAllocator<T>> make_filled_block(size_t n_rows, size_t n_cols) {
    Matrix<T, CudaHostAllocator<T>> block(Shape{n_rows, n_cols});
    for (size_t i = 0; i < n_rows; i++) {
        for (size_t j = 0; j < n_cols; j++) {
            block(i, j) = static_cast<T>(i + j + 1);
        }
    }
    return block;
}

// x and y of a block can have more elements than a kernel block has threads, both when
// they follow the rows of the block and when they follow the columns
template <typename T>
int test_gpu_blockdiag_gemv_block_larger_than_kernel_block(CudaRuntime& cudart) {

    const size_t n_large = cudart.blockSize() + 8;

    std::vector<Matrix<T, CudaHostAllocator<T>>> tall;
    tall.push_back(make_filled_block<T>(n_large, 4));
    tall.push_back(make_filled_block<T>(2, 2));
    if (check_blockdiag_gemv(cudart, BlockDiagMatrix<T>(tall), "N") != TEST_PASS) return TEST_FAIL;
    if (check_blockdiag_gemv(cudart, BlockDiagMatrix<T>(tall), "T") != TEST_PASS) return TEST_FAIL;

    std::vector<Matrix<T, CudaHostAllocator<T>>> wide;
    wide.push_back(make_filled_block<T>(4, n_large));
    wide.push_back(make_filled_block<T>(2, 2));
    if (check_blockdiag_gemv(cudart, BlockDiagMatrix<T>(wide), "N") != TEST_PASS) return TEST_FAIL;
    if (check_blockdiag_gemv(cudart, BlockDiagMatrix<T>(wide), "T") != TEST_PASS) return TEST_FAIL;

    return TEST_PASS;
}

// The device offsets used to unpack y must follow blocks added after an earlier product
template <typename T>
int test_gpu_blockdiag_gemv_after_add_block(CudaRuntime& cudart) {

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(make_filled_block<T>(1, 1));
    blocks.push_back(make_filled_block<T>(1, 1));
    BlockDiagMatrix<T> A(blocks);

    if (check_blockdiag_gemv(cudart, A, "N") != TEST_PASS) return TEST_FAIL;

    A.add_block(make_filled_block<T>(3, 3), 0);

    return check_blockdiag_gemv(cudart, A, "N");
}

// Naive reference, independent of BLAS: op(A) * x
template <typename T, typename MatrixType>
std::vector<T> reference_op_times_x(const MatrixType &A, const char *Ta, const std::vector<T> &x) {
    const bool trans = Ta[0] != 'N';
    const size_t n_rows = A.shape().first;
    const size_t n_cols = A.shape().second;

    std::vector<T> result(trans ? n_cols : n_rows, static_cast<T>(0));
    for (size_t i = 0; i < n_rows; i++) {
        for (size_t j = 0; j < n_cols; j++) {
            result[trans ? j : i] += A(i, j) * x[trans ? i : j];
        }
    }
    return result;
}

template <typename T>
Vector<T, CudaHostAllocator<T>> make_filled_vector(size_t n) {
    Vector<T, CudaHostAllocator<T>> x(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = static_cast<T>(i + 1);
    }
    return x;
}

template <typename T>
std::vector<T> to_std_vector(const Vector<T, CudaHostAllocator<T>> &v) {
    return std::vector<T>(v.data(), v.data() + v.size());
}

template <typename T>
void copy_result_to_host(CudaRuntime& cudart, Vector<T, CudaHostAllocator<T>> &v) {
    cudart.synchronize();
    v.copy2host(cudart);
    cudart.synchronize();
}

template <typename T>
BlockDiagMatrix<T> make_square_blockdiag() {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(make_filled_block<T>(2, 2));
    blocks.push_back(make_filled_block<T>(3, 3));
    return BlockDiagMatrix<T>(blocks);
}

// y only holds its value on the device when it is used as the input of the second product
template <typename T>
int test_gpu_blockdiag_gemv_input_from_earlier_product(CudaRuntime& cudart) {

    const BlockDiagMatrix<T> A = make_square_blockdiag<T>();
    const Vector<T, CudaHostAllocator<T>> x = make_filled_vector<T>(5);
    const std::vector<T> y_expected = reference_op_times_x(A, "N", to_std_vector(x));
    std::vector<T> z_expected = reference_op_times_x(A, "N", y_expected);

    Vector<T, CudaHostAllocator<T>> y(5, (T)0);
    Vector<T, CudaHostAllocator<T>> z(5, (T)0);
    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, (T)0.0, y);
    MatrixVectorProduct(cudart, "N", (T)1.0, A, y, (T)0.0, z);
    copy_result_to_host(cudart, z);

    if (!check(z.data(), z_expected.data(), 5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    return TEST_PASS;
}

// y only holds its value on the device when the second product scales it with beta
template <typename T>
int test_gpu_blockdiag_gemv_beta_from_earlier_product(CudaRuntime& cudart) {

    const BlockDiagMatrix<T> A = make_square_blockdiag<T>();
    const Vector<T, CudaHostAllocator<T>> x = make_filled_vector<T>(5);
    std::vector<T> expected = reference_op_times_x(A, "N", to_std_vector(x));
    for (T &value : expected) {
        value *= static_cast<T>(2);
    }

    Vector<T, CudaHostAllocator<T>> y(5, (T)0);
    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, (T)0.0, y);
    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, (T)1.0, y);
    copy_result_to_host(cudart, y);

    if (!check(y.data(), expected.data(), 5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_blockdiag_gemv_after_element_write(CudaRuntime& cudart) {

    BlockDiagMatrix<T> A = make_square_blockdiag<T>();
    const BlockDiagMatrix<T> &A_const = A;
    const Vector<T, CudaHostAllocator<T>> x = make_filled_vector<T>(5);

    Vector<T, CudaHostAllocator<T>> y_first(5, (T)0);
    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, (T)0.0, y_first);
    copy_result_to_host(cudart, y_first);

    A(0, 0) = (T)100;
    std::vector<T> expected = reference_op_times_x(A_const, "N", to_std_vector(x));

    Vector<T, CudaHostAllocator<T>> y(5, (T)0);
    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, (T)0.0, y);
    copy_result_to_host(cudart, y);

    if (!check(y.data(), expected.data(), 5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    return TEST_PASS;
}

// For real values "C" must give the same result as "T"
template <typename T>
int test_gpu_blockmatrix_gemv_conjugate_transpose(CudaRuntime& cudart) {

    BlockMatrix<T> A;
    A.set_block(0, 0, make_filled_block<T>(2, 2));
    A.set_block(2, 2, make_filled_block<T>(3, 3));

    const Vector<T, CudaHostAllocator<T>> x = make_filled_vector<T>(5);
    std::vector<T> expected = reference_op_times_x(A, "T", to_std_vector(x));

    Vector<T, CudaHostAllocator<T>> y(5, (T)0);
    MatrixVectorProduct(cudart, "C", (T)1.0, A, x, (T)0.0, y);
    copy_result_to_host(cudart, y);

    if (!check(y.data(), expected.data(), 5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_blockmatrix_gemv_uniform_rectangular_blocks(CudaRuntime& cudart) {

    BlockMatrix<T> A;
    A.set_block(0, 0, make_filled_block<T>(2, 3));
    A.set_block(2, 3, make_filled_block<T>(2, 3));
    A.set_block(4, 6, make_filled_block<T>(2, 3));

    const Vector<T, CudaHostAllocator<T>> x = make_filled_vector<T>(9);
    std::vector<T> expected = reference_op_times_x(A, "N", to_std_vector(x));

    Vector<T, CudaHostAllocator<T>> y(6, (T)0);
    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, (T)0.0, y);
    copy_result_to_host(cudart, y);

    if (!check(y.data(), expected.data(), 6, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    return TEST_PASS;
}

// The operations must not depend on the allocators, including the output of one
// product being the input of the next
template <typename T>
int test_gpu_blockdiag_gemv_async_allocator(CudaRuntime& cudart) {

    using AsyncAllocator = lahva::CudaDeviceAsyncAllocator<T>;
    using AsyncVector = Vector<T, CudaHostAllocator<T>, AsyncAllocator>;

    const BlockDiagMatrix<T> A_default = make_square_blockdiag<T>();
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    for (size_t b = 0; b < A_default.num_blocks(); b++) {
        blocks.push_back(A_default.get_block(b));
    }
    const BlockDiagMatrix<T, CudaHostAllocator<T>, AsyncAllocator> A(blocks);

    const std::vector<T> x_values = to_std_vector(make_filled_vector<T>(5));
    const std::vector<T> y_expected = reference_op_times_x(A, "N", x_values);
    std::vector<T> z_expected = reference_op_times_x(A, "N", y_expected);

    AsyncVector x(5);
    for (size_t i = 0; i < 5; i++) {
        x[i] = x_values[i];
    }
    AsyncVector y(5, (T)0);
    AsyncVector z(5, (T)0);

    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, (T)0.0, y);
    MatrixVectorProduct(cudart, A, y, z);
    cudart.synchronize();
    z.copy2host(cudart);
    cudart.synchronize();

    if (!check(z.data(), z_expected.data(), 5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gemv_strided(CudaRuntime& cudart) {

    Matrix<T> A(Shape{4, 4}, (T)0);
    A(0, 0) = 1; A(1, 0) = 3; A(0, 1) = 2; A(1, 1) = 4;
    A(2, 2) = 5; A(3, 2) = 7; A(2, 3) = 6; A(3, 3) = 8;

    Vector<T> x(8, (T)0);
    for (int i = 0; i < 4; i++) {
        x[2 * i] = static_cast<T>(i + 1);
    }
    Vector<T> y(12, (T)-1);

    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, 2, (T)0.0, y, 3);
    y.copy2host(cudart);
    cudart.synchronize();

    const T expected[4] = {5, 11, 39, 53};
    for (int i = 0; i < 4; i++) {
        if (!check(y[3 * i], expected[i], check_msg(get_type_name<T>(), "element at stride 3"))) return TEST_FAIL;
    }

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
int test_gpu_blockmatrix_gemv_const_runtime(const CudaRuntime& cudart) {

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
int test_gpu_blockdiag_gemv_const_runtime(const CudaRuntime& cudart) {

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

    MatrixVectorProduct(cudart, A, x, y);
    y.copy2host(cudart);
    cudart.synchronize();

    if (!check(y[0], static_cast<T>(5.0), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(y[1], static_cast<T>(11.0), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(y[2], static_cast<T>(39.0), check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check(y[3], static_cast<T>(53.0), check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;

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

// ============================================================================
// BlockMatrix GEMV sparse representation caching
// ============================================================================

template <typename T>
int test_gpu_blockmatrix_gemv_cached(CudaRuntime& cudart) {

    BlockMatrix<T> A;
    A.set_block(0, 0, Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {(T)1, (T)3, (T)2, (T)4}));
    A.set_block(2, 1, Matrix<T, CudaHostAllocator<T>>(Shape{2, 3}, {(T)1, (T)2, (T)3, (T)4, (T)5, (T)6}));
    const BlockMatrix<T> &Aref = A;

    Vector<T, CudaHostAllocator<T>> x(4);
    for (int i = 0; i < 4; ++i) x[i] = static_cast<T>(i + 1);

    auto check_result = [&](const char *msg) {
        Vector<T, CudaHostAllocator<T>> y(4, (T)0);
        MatrixVectorProduct(cudart, "N", (T)1.0, A, x, (T)0.0, y);
        y.copy2host(cudart);
        cudart.synchronize();
        for (size_t i = 0; i < 4; ++i) {
            T ref = 0;
            for (size_t j = 0; j < 4; ++j) ref += Aref(i, j) * x[j];
            if (!check(y[i], ref, check_msg(get_type_name<T>(), msg))) return false;
        }
        return true;
    };

    if (!check_result("first product")) return TEST_FAIL;
    if (!check_result("repeated product")) return TEST_FAIL;
    if (!check((int)A.sparse_build_count(), 1, check_msg(get_type_name<T>(), "repeated products should build once"))) return TEST_FAIL;

    A(3, 2) = (T)-10;
    if (!check_result("after value change")) return TEST_FAIL;
    if (!check((int)A.sparse_build_count(), 1, check_msg(get_type_name<T>(), "value change should not rebuild"))) return TEST_FAIL;
    if (!check((int)A.sparse_values_refresh_count(), 1, check_msg(get_type_name<T>(), "value change should refresh once"))) return TEST_FAIL;

    A.set_block(0, 3, Matrix<T, CudaHostAllocator<T>>(Shape{1, 1}, {(T)7}));
    if (!check_result("after adding a block")) return TEST_FAIL;
    if (!check((int)A.sparse_build_count(), 2, check_msg(get_type_name<T>(), "new block should rebuild"))) return TEST_FAIL;

    return TEST_PASS;
}

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
    for (const char *Ta : {"N", "T", "C"}) {
        total_failures += test_gpu_blockdiag_gemv_multi_block_transpose<double>(cudart, Ta);
        total_failures += test_gpu_blockdiag_gemv_multi_block_transpose<float>(cudart, Ta);
    }
    total_failures += test_gpu_blockdiag_gemv_block_larger_than_kernel_block<double>(cudart);
    total_failures += test_gpu_blockdiag_gemv_block_larger_than_kernel_block<float>(cudart);
    total_failures += test_gpu_blockdiag_gemv_after_add_block<double>(cudart);
    total_failures += test_gpu_blockdiag_gemv_after_add_block<float>(cudart);

    // Stride tests
    total_failures += test_gemv_strided<double>(cudart);
    total_failures += test_gemv_strided<float>(cudart);

    // Block-diagonal GEMV with data on the device and element writes
    total_failures += test_gpu_blockdiag_gemv_input_from_earlier_product<double>(cudart);
    total_failures += test_gpu_blockdiag_gemv_input_from_earlier_product<float>(cudart);
    total_failures += test_gpu_blockdiag_gemv_beta_from_earlier_product<double>(cudart);
    total_failures += test_gpu_blockdiag_gemv_beta_from_earlier_product<float>(cudart);
    total_failures += test_gpu_blockdiag_gemv_after_element_write<double>(cudart);
    total_failures += test_gpu_blockdiag_gemv_after_element_write<float>(cudart);

    // BlockMatrix GEMV with "C" and with uniform rectangular blocks
    total_failures += test_gpu_blockmatrix_gemv_conjugate_transpose<double>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_conjugate_transpose<float>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_uniform_rectangular_blocks<double>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_uniform_rectangular_blocks<float>(cudart);

    // Block-diagonal GEMV with other allocators
    total_failures += test_gpu_blockdiag_gemv_async_allocator<double>(cudart);
    total_failures += test_gpu_blockdiag_gemv_async_allocator<float>(cudart);

    // BlockMatrix GEMV with BSR format (uniform blocks)
    total_failures += test_gpu_blockmatrix_gemv_bsr_format<double>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_bsr_format<float>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_bsr_with_beta<double>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_bsr_with_beta<float>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_const_runtime<double>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_const_runtime<float>(cudart);
    total_failures += test_gpu_blockdiag_gemv_const_runtime<double>(cudart);
    total_failures += test_gpu_blockdiag_gemv_const_runtime<float>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_transpose_bsr<double>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_transpose_bsr<float>(cudart);

    // BlockMatrix GEMV with CSR format (non-uniform blocks)
    total_failures += test_gpu_blockmatrix_gemv_csr_format<double>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_csr_format<float>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_csr_with_beta<double>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_csr_with_beta<float>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_transpose_csr<double>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_transpose_csr<float>(cudart);

    // Sparse representation caching
    total_failures += test_gpu_blockmatrix_gemv_cached<double>(cudart);
    total_failures += test_gpu_blockmatrix_gemv_cached<float>(cudart);

    if (total_failures > 0) {
        std::cerr << "gpu/blas/level2 tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All gpu/blas/level2 tests passed!" << std::endl;
    return TEST_PASS;
};