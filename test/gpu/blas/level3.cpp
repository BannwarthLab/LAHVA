#include "test_common.h"
#include "array_utils.hpp"

using namespace lahva::gpu;
using lahva::Shape;
using lahva::CudaRuntime;
using lahva::CudaHostAllocator;

// ============================================================================
// Matrix-Matrix Product Tests
// ============================================================================

template <typename T>
int test_gemm_zero_v_gpu(CudaRuntime& cudart){

    Shape sres(10, 5);
    Shape sa(10, 3);
    Shape sb(3, 5);

    Matrix<T> A(sa, (T)1.0);
    Matrix<T> B(sb, (T)0.0);
    Matrix<T> C(sres, (T)1.0);

    MatrixMatrixProduct(cudart, "N", "N", (T)1.0, A, B, (T)0.0, C);

    cudart.synchronize();
    C.copy2host(cudart);

    Matrix<T> Mres(sres, (T)0.0);

    if (!check(C.data(), Mres.data(), 50, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gemm_identity_v_gpu(CudaRuntime& cudart){

    Shape sq(5, 5);
    Matrix<T> A(sq, (T)1.0);
    Matrix<T> B(sq, (T)0.0);
    Matrix<T> C(sq, (T)0.0);

    for (int i = 0; i < 5; i++) {
        B.data()[i * 5 + i] = (T)1.0;
    }

    MatrixMatrixProduct(cudart, "N", "N", (T)1.0, A, B, (T)0.0, C);

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 25; i++) {
        if (!check(C.data()[i], (T)1.0, check_msg(get_type_name<T>(), "")))
            return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_gemm_beta_nonzero(CudaRuntime& cudart){

    Shape sq(3, 3);
    Matrix<T> A(sq, (T)2.0);
    Matrix<T> B(sq, (T)0.5);
    Matrix<T> C(sq, (T)10.0);

    MatrixMatrixProduct(cudart, "N", "N", (T)1.0, A, B, (T)0.5, C);

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 9; i++) {
        T expected = ((T)2.0 * (T)0.5 * (T)3) + ((T)0.5 * (T)10.0);
        if (!check(C.data()[i], expected, check_msg(get_type_name<T>(), "")))
            return TEST_FAIL;
    }

    return TEST_PASS;
}

// ============================================================================
// Symmetric Matrix Product Tests
// ============================================================================

template <typename T>
int test_symm_left_side(CudaRuntime& cudart){

    Shape sq(4, 4);
    Shape sb(4, 5);
    Shape sc(4, 5);

    Matrix<T> A(sq, (T)1.0);
    Matrix<T> B(sb, (T)2.0);
    Matrix<T> C(sc, (T)0.0);

    // Ensure matrices are on device
    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);

    SymMatrixMatrixProduct(cudart, CUBLAS_SIDE_LEFT, (T)1.0, A, B, (T)0.0, C);

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 20; i++) {
        T expected = (T)1.0 * (T)4.0 * (T)2.0;
        if (!check(C.data()[i], expected, check_msg(get_type_name<T>(), "")))
            return TEST_FAIL;
    }

    return TEST_PASS;
}

// ============================================================================
// Alternative Parameter Order Tests
// ============================================================================

template <typename T>
int test_gemm_alt_param_order(CudaRuntime& cudart){

    Shape sres(10, 5);
    Shape sa(10, 3);
    Shape sb(3, 5);

    Matrix<T> A(sa, (T)1.0);
    Matrix<T> B(sb, (T)0.0);
    Matrix<T> C(sres, (T)1.0);

    // Call with alternate parameter order: A, B, C, alpha, beta, Ta, Tb
    MatrixMatrixProduct(cudart, A, B, C, (T)1.0, (T)0.0, "N", "N");

    cudart.synchronize();
    C.copy2host(cudart);

    Matrix<T> Mres(sres, (T)0.0);

    if (!check(C.data(), Mres.data(), 50, check_msg(get_type_name<T>(), "")))
        return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gemm_alt_with_alpha(CudaRuntime& cudart){

    Shape sq(4, 4);
    Matrix<T> A(sq, (T)1.0);
    Matrix<T> B(sq, (T)1.0);
    Matrix<T> C(sq, (T)0.0);

    // Call with alternate parameter order and alpha scaling
    MatrixMatrixProduct(cudart, A, B, C, (T)2.5, (T)0.0, "N", "N");

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 16; i++) {
        T expected = (T)2.5 * (T)4.0;
        if (!check(C.data()[i], expected, check_msg(get_type_name<T>(), ""))) {
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

template <typename T>
int test_gemm_alt_with_beta(CudaRuntime& cudart){

    Shape sq(3, 3);
    Matrix<T> A(sq, (T)2.0);
    Matrix<T> B(sq, (T)0.5);
    Matrix<T> C(sq, (T)10.0);

    // Call with alternate parameter order and beta
    MatrixMatrixProduct(cudart, A, B, C, (T)1.0, (T)0.5, "N", "N");

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 9; i++) {
        T expected = ((T)2.0 * (T)0.5 * (T)3) + ((T)0.5 * (T)10.0);
        if (!check(C.data()[i], expected, check_msg(get_type_name<T>(), ""))) {
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

// ============================================================================
// SymMatrixMatrixProduct SIDE_RIGHT Tests
// ============================================================================

template <typename T>
int test_symm_right_side(CudaRuntime& cudart){

    Shape sq(5, 5);

    Matrix<T> A(sq, (T)1.5);
    Matrix<T> B(sq, (T)2.0);
    Matrix<T> C(sq, (T)0.0);

    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);

    SymMatrixMatrixProduct(cudart, CUBLAS_SIDE_RIGHT, (T)1.0, A, B, (T)0.0, C);

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 25; i++) {
        T expected = (T)2.0 * (T)5.0 * (T)1.5;
        if (!check(C.data()[i], expected, check_msg(get_type_name<T>(), ""))) {
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

// ============================================================================
// Extended Coverage Tests
// ============================================================================

template <typename T>
int test_gemm_large_sizes(CudaRuntime& cudart){

    Shape sres(32, 32);
    Shape sa(32, 16);
    Shape sb(16, 32);

    Matrix<T> A(sa, (T)0.75);
    Matrix<T> B(sb, (T)1.5);
    Matrix<T> C(sres, (T)0.0);

    MatrixMatrixProduct(cudart, "N", "N", (T)1.0, A, B, (T)0.0, C);

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 1024; i++) {
        T expected = (T)0.75 * (T)16.0 * (T)1.5;
        if (!check(C.data()[i], expected, check_msg(get_type_name<T>(), ""))) {
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

template <typename T>
int test_gemm_alt_large(CudaRuntime& cudart){

    Shape sres(32, 32);
    Shape sa(32, 16);
    Shape sb(16, 32);

    Matrix<T> A(sa, (T)0.75);
    Matrix<T> B(sb, (T)1.5);
    Matrix<T> C(sres, (T)0.0);

    MatrixMatrixProduct(cudart, A, B, C, (T)1.0, (T)0.0, "N", "N");

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 1024; i++) {
        T expected = (T)0.75 * (T)16.0 * (T)1.5;
        if (!check(C.data()[i], expected, check_msg(get_type_name<T>(), ""))) {
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

template <typename T>
int test_gemm_alpha_scale(CudaRuntime& cudart){

    Shape sq(4, 4);
    Matrix<T> A(sq, (T)1.0);
    Matrix<T> B(sq, (T)1.0);
    Matrix<T> C(sq, (T)0.0);

    MatrixMatrixProduct(cudart, "N", "N", (T)2.5, A, B, (T)0.0, C);

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 16; i++) {
        T expected = (T)2.5 * (T)4.0;
        if (!check(C.data()[i], expected, check_msg(get_type_name<T>(), ""))) {
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

// ============================================================================
// SymMatrixMatrixProduct Alternative Order Tests
// ============================================================================

template <typename T>
int test_symm_alt_order(CudaRuntime& cudart){

    Shape sq(4, 4);
    Shape sb(4, 5);
    Shape sc(4, 5);

    Matrix<T> A(sq, (T)1.0);
    Matrix<T> B(sb, (T)2.0);
    Matrix<T> C(sc, (T)0.0);

    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);

    SymMatrixMatrixProduct(cudart, A, B, C, (T)1.0, (T)0.0, CUBLAS_SIDE_LEFT);

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 20; i++) {
        T expected = (T)1.0 * (T)4.0 * (T)2.0;
        if (!check(C.data()[i], expected, check_msg(get_type_name<T>(), ""))) {
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

template <typename T>
int test_symm_alt_order_right(CudaRuntime& cudart){

    Shape sq(5, 5);
    Matrix<T> A(sq, (T)1.5);
    Matrix<T> B(sq, (T)2.0);
    Matrix<T> C(sq, (T)0.0);

    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);

    SymMatrixMatrixProduct(cudart, A, B, C, (T)1.0, (T)0.0, CUBLAS_SIDE_RIGHT);

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 25; i++) {
        T expected = (T)2.0 * (T)5.0 * (T)1.5;
        if (!check(C.data()[i], expected, check_msg(get_type_name<T>(), ""))) {
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

// ============================================================================
// TF32 Variant Tests (single precision only - kept as-is)
// ============================================================================

int test_dgemm_tf32_zero(CudaRuntime& cudart){

    Shape sres(10, 5);
    Shape sa(10, 3);
    Shape sb(3, 5);

    Matrix<float> A(sa, 1.0f);
    Matrix<float> B(sb, 0.0f);
    Matrix<float> C(sres, 1.0f);

    MatrixMatrixProductTF32(cudart, "N", "N", 1.0f, A, B, 0.0f, C);

    cudart.synchronize();
    C.copy2host(cudart);

    Matrix<float> Mres(sres, 0.0f);

    if (!check(C.data(), Mres.data(), 50, check_msg(get_type_name<float>(), ""))) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

int test_dgemm_tf32_identity(CudaRuntime& cudart){

    Shape sq(5, 5);
    Matrix<float> A(sq, 1.0f);
    Matrix<float> B(sq, 0.0f);
    Matrix<float> C(sq, 0.0f);

    for (int i = 0; i < 5; i++) {
        B.data()[i * 5 + i] = 1.0f;
    }

    MatrixMatrixProductTF32(cudart, "N", "N", 1.0f, A, B, 0.0f, C);

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 25; i++) {
        if (!check(C.data()[i], 1.0f, check_msg(get_type_name<float>(), ""))) {
            return TEST_FAIL;
            break;
        }
    }

    return TEST_PASS;
}

int test_dgemm_tf32_alt_param(CudaRuntime& cudart){

    Shape sres(10, 5);
    Shape sa(10, 3);
    Shape sb(3, 5);

    Matrix<float> A(sa, 1.0f);
    Matrix<float> B(sb, 0.0f);
    Matrix<float> C(sres, 1.0f);

    // Call with alternate parameter order
    MatrixMatrixProductTF32(cudart, A, B, C, 1.0f, 0.0f, "N", "N");

    cudart.synchronize();
    C.copy2host(cudart);

    Matrix<float> Mres(sres, 0.0f);

    if (!check(C.data(), Mres.data(), 50, check_msg(get_type_name<float>(), ""))) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

int test_dgemm_tf32_with_alpha(CudaRuntime& cudart){

    Shape sq(4, 4);
    Matrix<float> A(sq, 1.0f);
    Matrix<float> B(sq, 1.0f);
    Matrix<float> C(sq, 0.0f);

    MatrixMatrixProductTF32(cudart, "N", "N", 2.5f, A, B, 0.0f, C);

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 16; i++) {
        float expected = 2.5f * 4.0f;
        if (!check(C.data()[i], expected, check_msg(get_type_name<float>(), ""))) {
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

int test_tf32_alt_with_beta(CudaRuntime& cudart){

    Shape sq(3, 3);
    Matrix<float> A(sq, 2.0f);
    Matrix<float> B(sq, 0.5f);
    Matrix<float> C(sq, 10.0f);

    MatrixMatrixProductTF32(cudart, A, B, C, 1.0f, 0.5f, "N", "N");

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 9; i++) {
        float expected = (2.0f * 0.5f * 3) + (0.5f * 10.0f);
        if (!check(C.data()[i], expected, check_msg(get_type_name<float>(), ""))) {
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

int test_tf32_large_matrices(CudaRuntime& cudart){

    Shape sres(32, 32);
    Shape sa(32, 16);
    Shape sb(16, 32);

    Matrix<float> A(sa, 0.75f);
    Matrix<float> B(sb, 1.5f);
    Matrix<float> C(sres, 0.0f);

    MatrixMatrixProductTF32(cudart, "N", "N", 1.0f, A, B, 0.0f, C);

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 1024; i++) {
        float expected = 0.75f * 16.0f * 1.5f;
        if (!check(C.data()[i], expected, check_msg(get_type_name<float>(), ""))) {
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

// ============================================================================
// Complex Number Tests (ZGEMM, CGEMM)
// ============================================================================

template <typename T>
int test_complex_gemm_basic(CudaRuntime& cudart){

    Shape sq(2, 2);
    Matrix<T> A(sq, T(1.0, 0.0));
    Matrix<T> B(sq, T(1.0, 0.0));
    Matrix<T> C(sq, T(0.0, 0.0));

    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);

    MatrixMatrixProduct(cudart, "N", "N", T(1.0, 0.0), A, B, T(0.0, 0.0), C);

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 4; i++) {
        T expected(2.0, 0.0);
        if (!check(C.data()[i].real(), expected.real(), check_msg(get_type_name<T>(), "check 1")) ||
            !check(C.data()[i].imag(), expected.imag(), check_msg(get_type_name<T>(), "check 2"))) {
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

template <typename T>
int test_complex_gemm_alt_order(CudaRuntime& cudart){

    Shape sq(2, 2);
    Matrix<T> A(sq, T(1.0, 0.0));
    Matrix<T> B(sq, T(1.0, 0.0));
    Matrix<T> C(sq, T(0.0, 0.0));

    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);

    MatrixMatrixProduct(cudart, A, B, C, T(1.0, 0.0), T(0.0, 0.0), "N", "N");

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 4; i++) {
        T expected(2.0, 0.0);
        if (!check(C.data()[i].real(), expected.real(), check_msg(get_type_name<T>(), "check 1")) ||
            !check(C.data()[i].imag(), expected.imag(), check_msg(get_type_name<T>(), "check 2"))) {
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

template <typename T>
int test_complex_gemm_with_beta(CudaRuntime& cudart){

    Shape sq(2, 2);
    Matrix<T> A(sq, T(1.0, 0.0));
    Matrix<T> B(sq, T(1.0, 0.0));
    Matrix<T> C(sq, T(2.0, 0.0));

    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);

    MatrixMatrixProduct(cudart, "N", "N", T(1.0, 0.0), A, B, T(0.5, 0.0), C);

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 4; i++) {
        T expected(2.0 + 0.5 * 2.0, 0.0);
        if (!check(C.data()[i].real(), expected.real(), check_msg(get_type_name<T>(), "check 1")) ||
            !check(C.data()[i].imag(), expected.imag(), check_msg(get_type_name<T>(), "check 2"))) {
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

// ============================================================================
// FP16 Mixed Precision Tests
// ============================================================================

int test_fp16_basic(CudaRuntime& cudart){

    Shape sq(2, 2);
    Matrix<__half> A(sq, __half(1.0f));
    Matrix<__half> B(sq, __half(1.0f));
    Matrix<float> C(sq, 0.0f);

    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);

    MatrixMatrixProductFP16(cudart, "N", "N", 1.0f, A, B, 0.0f, C);

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 4; i++) {
        float expected = 2.0f;
        if (!check(C.data()[i], expected, check_msg(get_type_name<float>(), ""))) {
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

int test_fp16_alt_order(CudaRuntime& cudart){

    Shape sq(2, 2);
    Matrix<__half> A(sq, __half(1.0f));
    Matrix<__half> B(sq, __half(1.0f));
    Matrix<float> C(sq, 0.0f);

    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);

    MatrixMatrixProductFP16(cudart, A, B, C, 1.0f, 0.0f, "N", "N");

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 4; i++) {
        float expected = 2.0f;
        if (!check(C.data()[i], expected, check_msg(get_type_name<float>(), ""))) {
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

int test_fp16_with_alpha(CudaRuntime& cudart){

    Shape sq(3, 3);
    Matrix<__half> A(sq, __half(1.0f));
    Matrix<__half> B(sq, __half(1.0f));
    Matrix<float> C(sq, 0.0f);

    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);

    MatrixMatrixProductFP16(cudart, "N", "N", 2.5f, A, B, 0.0f, C);

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 9; i++) {
        float expected = 2.5f * 3.0f;
        if (!check(C.data()[i], expected, check_msg(get_type_name<float>(), ""))) {
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

int test_fp16_with_beta(CudaRuntime& cudart){

    Shape sq(2, 2);
    Matrix<__half> A(sq, __half(1.0f));
    Matrix<__half> B(sq, __half(1.0f));
    Matrix<float> C(sq, 5.0f);

    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);

    MatrixMatrixProductFP16(cudart, "N", "N", 1.0f, A, B, 0.5f, C);

    cudart.synchronize();
    C.copy2host(cudart);

    for (int i = 0; i < 4; i++) {
        float expected = 2.0f + 0.5f * 5.0f;
        if (!check(C.data()[i], expected, check_msg(get_type_name<float>(), ""))) {
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

// ============================================================================
// MixedPrecisionMatrix Operational Tests
// ============================================================================

template <typename T>
void fill_random_mp(Matrix<T>& m) {
    std::random_device rd;
    std::minstd_rand eng(rd());
    std::normal_distribution<> distr(0.0, 0.1);

    for (size_t i = 0; i < m.shape().first; i++) {
        for (size_t j = 0; j < m.shape().second; j++) {
            if (i == j)
                m(i, j) = 1.0 + std::abs(distr(eng));
            else
                m(i, j) = std::abs(distr(eng));
        }
    }
}

template <typename T>
int test_mp_matrix_gemm_basic(CudaRuntime& cudart) {

    Shape shape(8, 8);
    MixedPrecisionMatrix<T> A(shape);
    MixedPrecisionMatrix<T> B(shape);
    MixedPrecisionMatrix<T> C(shape);

    fill_random_mp(A);
    fill_random_mp(B);

    for (size_t i = 0; i < C.size(); i++) {
        C.data()[i] = (T)0.0;
    }

    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);
    cudart.synchronize();

    MPRuntime mp_rt;
    mp_rt.nsplits_FP32 = 4;
    mp_rt.fast_mode = true;

    MatrixMatrixProduct(cudart, mp_rt, A, B, C, (T)1.0, (T)0.0);
    cudart.synchronize();
    C.copy2host(cudart);
    cudart.synchronize();

    double sum = 0.0;
    for (size_t i = 0; i < C.size(); i++) {
        sum += std::abs((double)C.data()[i]);
    }

    if (sum < 1e-10) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_mp_matrix_copy_to_device(CudaRuntime& cudart) {

    Shape shape(4, 4);
    MixedPrecisionMatrix<T> m(shape);

    for (size_t i = 0; i < m.size(); i++) {
        m.data()[i] = (T)2.5;
    }

    m.copy2device(cudart);
    cudart.synchronize();

    m.data()[0] = (T)0.0;
    m.copy2host(cudart);
    cudart.synchronize();


    if (!check(m.data()[0], (T)2.5, check_msg(get_type_name<T>(), ""))) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_mp_matrix_scaling_operation(CudaRuntime& cudart) {

    Shape shape(4, 4);
    MixedPrecisionMatrix<T> A(shape);
    MixedPrecisionMatrix<T> B(shape);
    MixedPrecisionMatrix<T> C(shape);

    fill_random_mp(A);
    fill_random_mp(B);

    for (size_t i = 0; i < C.size(); i++) {
        C.data()[i] = (T)0.0;
    }

    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);
    cudart.synchronize();

    MPRuntime mp_rt;
    mp_rt.nsplits_FP32 = 4;
    mp_rt.fast_mode = true;

    MatrixMatrixProduct(cudart, mp_rt, A, B, C, (T)0.5, (T)0.0);
    cudart.synchronize();
    C.copy2host(cudart);
    cudart.synchronize();

    double sum = 0.0;
    for (size_t i = 0; i < C.size(); i++) {
        sum += std::abs((double)C.data()[i]);
    }

    if (sum < 1e-10) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_mp_matrix_accumulation(CudaRuntime& cudart) {

    Shape shape(4, 4);
    MixedPrecisionMatrix<T> A(shape);
    MixedPrecisionMatrix<T> B(shape);
    MixedPrecisionMatrix<T> C(shape);

    fill_random_mp(A);
    fill_random_mp(B);

    for (size_t i = 0; i < C.size(); i++) {
        C.data()[i] = (T)1.0;
    }

    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);
    cudart.synchronize();

    MPRuntime mp_rt;
    mp_rt.nsplits_FP32 = 4;
    mp_rt.fast_mode = true;

    MatrixMatrixProduct(cudart, mp_rt, A, B, C, (T)1.0, (T)1.0);
    cudart.synchronize();
    C.copy2host(cudart);
    cudart.synchronize();

    double sum = 0.0;
    for (size_t i = 0; i < C.size(); i++) {
        sum += (double)C.data()[i];
    }

    if (sum <= 16.0) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_mp_matrix_identity_multiplication(CudaRuntime& cudart) {

    Shape shape(4, 4);
    MixedPrecisionMatrix<T> I(shape);
    MixedPrecisionMatrix<T> A(shape);
    MixedPrecisionMatrix<T> C(shape);

    for (size_t i = 0; i < I.size(); i++) {
        I.data()[i] = (T)0.0;
    }
    for (size_t i = 0; i < I.shape().first; i++) {
        I(i, i) = (T)1.0;
    }

    fill_random_mp(A);

    for (size_t i = 0; i < C.size(); i++) {
        C.data()[i] = (T)0.0;
    }

    I.copy2device(cudart);
    A.copy2device(cudart);
    C.copy2device(cudart);
    cudart.synchronize();

    MPRuntime mp_rt;
    mp_rt.nsplits_FP32 = 4;
    mp_rt.fast_mode = true;

    MatrixMatrixProduct(cudart, mp_rt, I, A, C, (T)1.0, (T)0.0);
    cudart.synchronize();
    C.copy2host(cudart);
    cudart.synchronize();

    bool close = true;
    for (size_t i = 0; i < A.size(); i++) {
        if (std::abs((double)C.data()[i] - (double)A.data()[i]) > 1e-4) {
            close = false;
            break;
        }
    }

    if (!close) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_mp_matrix_transpose_consistency(CudaRuntime& cudart) {

    Shape shape(4, 4);
    MixedPrecisionMatrix<T> A(shape);
    MixedPrecisionMatrix<T> B(shape);
    MixedPrecisionMatrix<T> C1(shape);
    MixedPrecisionMatrix<T> C2(shape);

    fill_random_mp(A);
    fill_random_mp(B);

    for (size_t i = 0; i < C1.size(); i++) {
        C1.data()[i] = (T)0.0;
        C2.data()[i] = (T)0.0;
    }

    A.copy2device(cudart);
    B.copy2device(cudart);
    C1.copy2device(cudart);
    C2.copy2device(cudart);
    cudart.synchronize();

    MPRuntime mp_rt;
    mp_rt.nsplits_FP32 = 4;
    mp_rt.fast_mode = true;

    MatrixMatrixProduct(cudart, mp_rt, A, B, C1, (T)1.0, (T)0.0);
    cudart.synchronize();

    MatrixMatrixProduct(cudart, mp_rt, B, A, C2, (T)1.0, (T)0.0, "T", "T");
    cudart.synchronize();

    C1.copy2host(cudart);
    C2.copy2host(cudart);
    cudart.synchronize();

    bool consistent = true;
    for (size_t i = 0; i < C1.shape().first && consistent; i++) {
        for (size_t j = 0; j < C1.shape().second && consistent; j++) {
            if (std::abs((double)C1(i, j) - (double)C2(j, i)) > 1e-4) {
                consistent = false;
            }
        }
    }

    if (!consistent) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_mp_matrix_nonbatch_gemm(CudaRuntime& cudart) {
    MPRuntime mp_rt;
    mp_rt.nsplits_FP32 = 4;
    mp_rt.fast_mode = false;
    mp_rt.batch_mode = false;

    Shape shape(8, 8);
    MixedPrecisionMatrix<T> A(shape);
    MixedPrecisionMatrix<T> B(shape);
    MixedPrecisionMatrix<T> C(shape);

    fill_random_mp(A);
    fill_random_mp(B);

    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);
    cudart.synchronize();

    MatrixMatrixProduct(cudart, mp_rt, A, B, C, (T)1.0, (T)0.0);
    cudart.synchronize();
    C.copy2host(cudart);
    cudart.synchronize();

    MixedPrecisionMatrix<T> C_ref(shape);
    C_ref.copy2device(cudart);
    MatrixMatrixProduct(cudart, A, B, C_ref, (T)1.0, (T)0.0);
    C_ref.copy2host(cudart);
    cudart.synchronize();

    
    if (!check(C.data(), C_ref.data(), C.size(), check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_mp_matrix_nonbatch_with_alpha(CudaRuntime& cudart) {
    MPRuntime mp_rt;
    mp_rt.nsplits_FP32 = 4;
    mp_rt.fast_mode = false;
    mp_rt.batch_mode = false;

    Shape shape(8, 8);
    MixedPrecisionMatrix<T> A(shape);
    MixedPrecisionMatrix<T> B(shape);
    MixedPrecisionMatrix<T> C(shape);

    fill_random_mp(A);
    fill_random_mp(B);

    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);
    cudart.synchronize();

    MatrixMatrixProduct(cudart, mp_rt, A, B, C, (T)2.5, (T)0.0);
    cudart.synchronize();
    C.copy2host(cudart);
    cudart.synchronize();

    MixedPrecisionMatrix<T> C_ref(shape);
    C_ref.copy2device(cudart);
    MatrixMatrixProduct(cudart, A, B, C_ref, (T)2.5, (T)0.0);
    C_ref.copy2host(cudart);
    cudart.synchronize();

    if (!check(C.data(), C_ref.data(), C.size(), check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_mp_matrix_nonbatch_with_beta(CudaRuntime& cudart) {
    MPRuntime mp_rt;
    mp_rt.nsplits_FP32 = 4;
    mp_rt.fast_mode = false;
    mp_rt.batch_mode = false;

    Shape shape(8, 8);
    MixedPrecisionMatrix<T> A(shape);
    MixedPrecisionMatrix<T> B(shape);
    MixedPrecisionMatrix<T> C(shape);
    MixedPrecisionMatrix<T> C_ref(shape);

    fill_random_mp(A);
    fill_random_mp(B);
    fill_random_mp(C);

    lahva::cpu::CopyVectors(C, C_ref);

    A.copy2device(cudart);
    B.copy2device(cudart);
    C.copy2device(cudart);
    cudart.synchronize();

    MatrixMatrixProduct(cudart, mp_rt, A, B, C, (T)1.0, (T)0.5);
    cudart.synchronize();
    C.copy2host(cudart);
    cudart.synchronize();


    C_ref.copy2device(cudart);
    cudart.synchronize();
    MatrixMatrixProduct(cudart, A, B, C_ref, (T)1.0, (T)0.5);
    C_ref.copy2host(cudart);
    cudart.synchronize();

    if (!check(C.data(), C_ref.data(), C.size(), check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Main
// ============================================================================

int main(){
    int total_failures = 0;
    CudaRuntime cudart;

    // General matrix-matrix product tests
    total_failures += test_gemm_zero_v_gpu<double>(cudart);
    total_failures += test_gemm_zero_v_gpu<float>(cudart);
    total_failures += test_gemm_identity_v_gpu<double>(cudart);
    total_failures += test_gemm_identity_v_gpu<float>(cudart);
    total_failures += test_gemm_beta_nonzero<double>(cudart);
    total_failures += test_gemm_beta_nonzero<float>(cudart);
    total_failures += test_gemm_alpha_scale<double>(cudart);
    total_failures += test_gemm_alpha_scale<float>(cudart);

    // Symmetric matrix product tests - LEFT side
    total_failures += test_symm_left_side<double>(cudart);
    total_failures += test_symm_left_side<float>(cudart);

    // Symmetric matrix product tests - RIGHT side
    total_failures += test_symm_right_side<double>(cudart);
    total_failures += test_symm_right_side<float>(cudart);

    // Alternative parameter order tests for general matrix product
    total_failures += test_gemm_alt_param_order<double>(cudart);
    total_failures += test_gemm_alt_param_order<float>(cudart);
    total_failures += test_gemm_alt_with_alpha<double>(cudart);
    total_failures += test_gemm_alt_with_alpha<float>(cudart);
    total_failures += test_gemm_alt_with_beta<double>(cudart);
    total_failures += test_gemm_alt_with_beta<float>(cudart);

    // Symmetric matrix product tests - alternative parameter order
    total_failures += test_symm_alt_order<double>(cudart);
    total_failures += test_symm_alt_order<float>(cudart);
    total_failures += test_symm_alt_order_right<double>(cudart);
    total_failures += test_symm_alt_order_right<float>(cudart);

    // TF32 precision tests (single precision only)
    total_failures += test_dgemm_tf32_zero(cudart);
    total_failures += test_dgemm_tf32_identity(cudart);
    total_failures += test_dgemm_tf32_alt_param(cudart);
    total_failures += test_dgemm_tf32_with_alpha(cudart);
    total_failures += test_tf32_alt_with_beta(cudart);
    total_failures += test_tf32_large_matrices(cudart);

    // Extended coverage tests - larger matrices
    total_failures += test_gemm_large_sizes<double>(cudart);
    total_failures += test_gemm_large_sizes<float>(cudart);
    total_failures += test_gemm_alt_large<double>(cudart);
    total_failures += test_gemm_alt_large<float>(cudart);

    // Complex number tests
    total_failures += test_complex_gemm_basic<complex_double>(cudart);
    total_failures += test_complex_gemm_alt_order<complex_double>(cudart);
    total_failures += test_complex_gemm_basic<complex_float>(cudart);
    total_failures += test_complex_gemm_alt_order<complex_float>(cudart);
    total_failures += test_complex_gemm_with_beta<complex_double>(cudart);
    total_failures += test_complex_gemm_with_beta<complex_float>(cudart);

    // FP16 mixed precision tests
    total_failures += test_fp16_basic(cudart);
    total_failures += test_fp16_alt_order(cudart);
    total_failures += test_fp16_with_alpha(cudart);
    total_failures += test_fp16_with_beta(cudart);

    // MixedPrecisionMatrix operational tests
    total_failures += test_mp_matrix_gemm_basic<double>(cudart);
    total_failures += test_mp_matrix_gemm_basic<float>(cudart);
    total_failures += test_mp_matrix_copy_to_device<double>(cudart);
    total_failures += test_mp_matrix_copy_to_device<float>(cudart);
    total_failures += test_mp_matrix_scaling_operation<double>(cudart);
    total_failures += test_mp_matrix_scaling_operation<float>(cudart);
    total_failures += test_mp_matrix_accumulation<double>(cudart);
    total_failures += test_mp_matrix_accumulation<float>(cudart);
    total_failures += test_mp_matrix_identity_multiplication<double>(cudart);
    total_failures += test_mp_matrix_identity_multiplication<float>(cudart);
    total_failures += test_mp_matrix_transpose_consistency<double>(cudart);
    total_failures += test_mp_matrix_transpose_consistency<float>(cudart);

    if (total_failures > 0) {
        std::cerr << "gpu/blas/level3 tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All gpu/blas/level3 tests passed!" << std::endl;
    return TEST_PASS;
};