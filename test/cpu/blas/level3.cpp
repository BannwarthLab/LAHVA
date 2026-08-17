#include "test_common.h"
#include "array_utils.hpp"

using namespace lahva::cpu;
using lahva::Shape;

#define M 10
#define N 5 
#define K 3

template <typename T>
int test_gemm_zero_v_cpp() {
    Shape sres(M,N);
    Shape sa(M,K);
    Shape sb(K,N);
    Matrix<T> A(sa, 1.0);
    Matrix<T> B(sb, 0.0);
    Matrix<T> C(sres, 1.0);

    MatrixMatrixProduct("N", "N", (T)1.0, A, B, (T)0.0, C);

    Matrix<T> Mres(sres, 0.0);

    if (!check(C.data(), Mres.data(), M*N, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Additional GEMM Tests with Different Transpose Options
// ============================================================================

template <typename T>
int test_gemm_with_scaling() {
    Shape sres(M,N);
    Shape sa(M,K);
    Shape sb(K,N);

    Matrix<T> A(sa, (T)1.0);
    Matrix<T> B(sb, (T)2.0);
    Matrix<T> C(sres, (T)5.0);

    MatrixMatrixProduct("N", "N", (T)0.0, A, B, (T)1.0, C);

    // Result = 0.0*(A*B) + 1.0*C_old = 5.0
    if (!check((double)C(0,0), 5.0, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gemm_negative_alpha() {
    Shape sres(M,N);
    Shape sa(M,K);
    Shape sb(K,N);

    Matrix<T> A(sa, (T)1.0);
    Matrix<T> B(sb, (T)2.0);
    Matrix<T> C(sres, (T)1.0);

    MatrixMatrixProduct("N", "N", (T)(-1.0), A, B, (T)1.0, C);

    // Result = -1.0*(A*B) + 1.0*C = -(2*K) + 1 = 1 - 2*K
    if (!check((double)C(0,0), 1.0 - 2.0*K, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gemm_with_beta() {
    Shape sres(M,N);
    Shape sa(M,K);
    Shape sb(K,N);

    Matrix<T> A(sa, (T)1.0);
    Matrix<T> B(sb, (T)2.0);
    Matrix<T> C(sres, (T)3.0);

    MatrixMatrixProduct("N", "N", (T)1.0, A, B, (T)2.0, C);

    // Result = A*B + 2*C = (2*K + 6)
    if (!check((double)C(0,0), 2.0*K + 6.0, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gemm_with_alpha_beta() {
    Shape sres(M,N);
    Shape sa(M,K);
    Shape sb(K,N);

    Matrix<T> A(sa, (T)1.0);
    Matrix<T> B(sb, (T)2.0);
    Matrix<T> C(sres, (T)3.0);

    MatrixMatrixProduct("N", "N", (T)0.5, A, B, (T)2.0, C);

    // Result = 0.5*(A*B) + 2*C = 0.5*(2*K) + 6 = K + 6
    if (!check((double)C(0,0), 0.5*2.0*K + 6.0, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// SymMatrixMatrixProduct Tests
// ============================================================================

template <typename T>
int test_symm_left() {
    Shape sa(5, 5);
    Shape sb(5, 3);
    Shape sres(5, 3);

    Matrix<T> A(sa, (T)1.0);
    Matrix<T> B(sb, (T)2.0);
    Matrix<T> C(sres, (T)0.0);

    SymMatrixMatrixProduct(CblasLeft, (T)1.0, A, B, (T)0.0, C);

    // C = A*B where A is symmetric
    if (!check((double)C(0,0), 2.0 * 5, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_symm_right() {
    // For CblasRight: C = alpha * B * A + beta * C where A is symmetric n x n
    // We use A as 3x3, B as 3x3, C as 3x3 to satisfy check_size_mm
    Shape sa(3, 3);
    Shape sb(3, 3);
    Shape sres(3, 3);

    Matrix<T> A(sa, (T)1.0);
    Matrix<T> B(sb, (T)2.0);
    Matrix<T> C(sres, (T)0.0);

    SymMatrixMatrixProduct(CblasRight, (T)1.0, A, B, (T)0.0, C);

    // C = B*A where both are 3x3, result is 3x3
    // C[0,0] = sum(B[0,k] * A[k,0]) = 2*1 + 2*1 + 2*1 = 6
    if (!check((double)C(0,0), 2.0 * 3, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_symm_with_beta() {
    Shape sa(5, 5);
    Shape sb(5, 3);
    Shape sres(5, 3);

    Matrix<T> A(sa, (T)1.0);
    Matrix<T> B(sb, (T)2.0);
    Matrix<T> C(sres, (T)3.0);

    SymMatrixMatrixProduct(CblasLeft, (T)1.0, A, B, (T)2.0, C);

    // C = A*B + 2*C_old = (2*5) + 2*(3) = 16
    if (!check((double)C(0,0), 10.0 + 6.0, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// C-level Pointer-based GEMM Tests
// ============================================================================

template <typename T>
int test_c_gemm_pointer() {

    const int m = 3, n = 2, k = 4;
    T A[m*k];
    T B[k*n];
    T C[m*n];

    for(int i = 0; i < m*k; i++) A[i] = (T)1.0;
    for(int i = 0; i < k*n; i++) B[i] = (T)2.0;
    for(int i = 0; i < m*n; i++) C[i] = (T)0.0;

    MatrixMatrixProduct("N", "N", m, n, k, (T)1.0, A, B, (T)0.0, C);

    if (!check((double)C[0], 2.0 * k, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_c_gemm_negative_alpha() {

    const int m = 3, n = 2, k = 4;
    T A[m*k];
    T B[k*n];
    T C[m*n];

    for(int i = 0; i < m*k; i++) A[i] = (T)1.0;
    for(int i = 0; i < k*n; i++) B[i] = (T)2.0;
    for(int i = 0; i < m*n; i++) C[i] = (T)5.0;

    MatrixMatrixProduct("N", "N", m, n, k, (T)(-1.0), A, B, (T)1.0, C);

    // Result = -1.0*(A*B) + 1.0*C_old = -(2*k) + 5 = 5 - 8 = -3
    if (!check((double)C[0], 5.0 - 2.0*k, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_c_gemm_with_scaling() {

    const int m = 3, n = 2, k = 4;
    T A[m*k];
    T B[k*n];
    T C[m*n];

    for(int i = 0; i < m*k; i++) A[i] = (T)1.0;
    for(int i = 0; i < k*n; i++) B[i] = (T)2.0;
    for(int i = 0; i < m*n; i++) C[i] = (T)3.0;

    MatrixMatrixProduct("N", "N", m, n, k, (T)0.5, A, B, (T)2.0, C);

    // Result = 0.5*(A*B) + 2*C_old = 0.5*(2*k) + 2*3 = k + 6
    if (!check((double)C[0], 0.5*2.0*k + 6.0, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_c_gemm_with_beta() {

    const int m = 3, n = 2, k = 4;
    T A[m*k];
    T B[k*n];
    T C[m*n];

    for(int i = 0; i < m*k; i++) A[i] = (T)1.0;
    for(int i = 0; i < k*n; i++) B[i] = (T)2.0;
    for(int i = 0; i < m*n; i++) C[i] = (T)5.0;

    MatrixMatrixProduct("N", "N", m, n, k, (T)1.0, A, B, (T)3.0, C);

    // Result = A*B + 3*C_old = (2*k) + 3*5 = 2*4 + 15 = 23
    if (!check((double)C[0], 2.0*k + 15.0, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// C-level SymMatrixMatrixProduct Tests (both branches)
// ============================================================================

template <typename T>
int test_c_symm_left() {

    const int m = 5, n = 3;
    T A[m*m];
    T B[m*n];
    T C[m*n];

    for(int i = 0; i < m*m; i++) A[i] = (T)1.0;
    for(int i = 0; i < m*n; i++) B[i] = (T)2.0;
    for(int i = 0; i < m*n; i++) C[i] = (T)0.0;

    SymMatrixMatrixProduct(CblasLeft, m, n, (T)1.0, A, B, (T)0.0, C);

    // C = A*B where A is m x m
    if (!check((double)C[0], 2.0 * m, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_c_symm_right() {

    const int m = 3, n = 3;
    T A[n*n];
    T B[m*n];
    T C[m*n];

    for(int i = 0; i < n*n; i++) A[i] = (T)1.0;
    for(int i = 0; i < m*n; i++) B[i] = (T)2.0;
    for(int i = 0; i < m*n; i++) C[i] = (T)0.0;

    SymMatrixMatrixProduct(CblasRight, m, n, (T)1.0, A, B, (T)0.0, C);

    // C = B*A where A is n x n
    if (!check((double)C[0], 2.0 * n, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_c_symm_with_beta_left() {

    const int m = 5, n = 3;
    T A[m*m];
    T B[m*n];
    T C[m*n];

    for(int i = 0; i < m*m; i++) A[i] = (T)1.0;
    for(int i = 0; i < m*n; i++) B[i] = (T)2.0;
    for(int i = 0; i < m*n; i++) C[i] = (T)4.0;

    SymMatrixMatrixProduct(CblasLeft, m, n, (T)1.0, A, B, (T)2.0, C);

    // C = A*B + 2*C_old = (2*m) + 2*(4) = 10 + 8 = 18
    if (!check((double)C[0], 2.0*m + 8.0, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_c_symm_with_beta_right() {

    const int m = 3, n = 3;
    T A[n*n];
    T B[m*n];
    T C[m*n];

    for(int i = 0; i < n*n; i++) A[i] = (T)1.0;
    for(int i = 0; i < m*n; i++) B[i] = (T)2.0;
    for(int i = 0; i < m*n; i++) C[i] = (T)4.0;

    SymMatrixMatrixProduct(CblasRight, m, n, (T)1.0, A, B, (T)2.0, C);

    // C = B*A + 2*C_old = (2*n) + 2*(4) = 6 + 8 = 14
    if (!check((double)C[0], 2.0*n + 8.0, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template<typename T>
int test_blockdiag_gemm_simple() {
    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 3, 2, 4}));
    blocks.push_back(Matrix<T>(Shape{2, 2}, {5, 7, 6, 8}));

    BlockDiagMatrix<T> A(blocks);

    Matrix<T> B(Shape{4, 3});
    for (int i = 0; i < 12; i++) {
        B.data()[i] = static_cast<T>(i + 1);
    }

    Matrix<T> C(Shape{4, 3}, static_cast<T>(0));

    MatrixMatrixProduct(A, B, C);

    if (!check(C(0, 0), static_cast<T>(5), check_msg(get_type_name<T>(), "C(0,0) should be 5"))) return TEST_FAIL;
    if (!check(C(0, 1), static_cast<T>(17), check_msg(get_type_name<T>(), "C(0,1) should be 17"))) return TEST_FAIL;
    if (!check(C(0, 2), static_cast<T>(29), check_msg(get_type_name<T>(), "C(0,2) should be 29"))) return TEST_FAIL;
    if (!check(C(2, 0), static_cast<T>(39), check_msg(get_type_name<T>(), "C(2,0) should be 39"))) return TEST_FAIL;
    if (!check(C(3, 2), static_cast<T>(173), check_msg(get_type_name<T>(), "C(3,2) should be 173"))) return TEST_FAIL;

    return TEST_PASS;
}

template<typename T>
int test_blockdiag_gemm_with_beta() {
    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 2, 3, 4}));

    BlockDiagMatrix<T> A(blocks);

    Matrix<T> B(Shape{2, 2}, static_cast<T>(1));
    Matrix<T> C(Shape{2, 2}, static_cast<T>(2));

    MatrixMatrixProduct(A, B, C, static_cast<T>(1), static_cast<T>(2));

    if (!check(C(0, 0), static_cast<T>(8), check_msg(get_type_name<T>(), "C(0,0) should be 8"))) return TEST_FAIL;
    if (!check(C(1, 1), static_cast<T>(10), check_msg(get_type_name<T>(), "C(1,1) should be 10"))) return TEST_FAIL;

    return TEST_PASS;
}

template<typename T>
int test_blockdiag_gemm_scaling() {
    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 2, 3, 4}));

    BlockDiagMatrix<T> A(blocks);

    Matrix<T> B(Shape{2, 2}, static_cast<T>(1));
    Matrix<T> C(Shape{2, 2}, static_cast<T>(0));

    MatrixMatrixProduct(A, B, C, static_cast<T>(2), static_cast<T>(0));

    if (!check(C(0, 0), static_cast<T>(8), check_msg(get_type_name<T>(), "C(0,0) should be 8"))) return TEST_FAIL;
    if (!check(C(1, 1), static_cast<T>(12), check_msg(get_type_name<T>(), "C(1,1) should be 12"))) return TEST_FAIL;

    return TEST_PASS;
}

template<typename T>
int test_blockdiag_gemm_varying_blocks() {
    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{1, 1}, {2}));
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 2, 3, 4}));

    BlockDiagMatrix<T> A(blocks);

    Matrix<T> B(Shape{3, 2});
    for (int i = 0; i < 6; i++) {
        B.data()[i] = static_cast<T>(i + 1);
    }

    Matrix<T> C(Shape{3, 2}, static_cast<T>(0));

    MatrixMatrixProduct(A, B, C);

    if (!check(C(0, 0), static_cast<T>(2), check_msg(get_type_name<T>(), "C(0,0) should be 2"))) return TEST_FAIL;
    if (!check(C(1, 0), static_cast<T>(11), check_msg(get_type_name<T>(), "C(1,0) should be 11"))) return TEST_FAIL;
    if (!check(C(2, 1), static_cast<T>(34), check_msg(get_type_name<T>(), "C(2,1) should be 34"))) return TEST_FAIL;

    return TEST_PASS;
}

template<typename T>
int test_blockdiag_gemm_transpose_a() {
    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 3}, {1, 4, 2, 5, 3, 6}));

    BlockDiagMatrix<T> A(blocks);

    Matrix<T> B(Shape{2, 2});
    B(0, 0) = static_cast<T>(1); B(0, 1) = static_cast<T>(2);
    B(1, 0) = static_cast<T>(3); B(1, 1) = static_cast<T>(4);

    Matrix<T> C(Shape{3, 2}, static_cast<T>(0));

    MatrixMatrixProduct(A, B, C, static_cast<T>(1), static_cast<T>(0), "T", "N");

    if (!check(C(0, 0), static_cast<T>(13), check_msg(get_type_name<T>(), "C(0,0) should be 13"))) return TEST_FAIL;
    if (!check(C(1, 0), static_cast<T>(17), check_msg(get_type_name<T>(), "C(1,0) should be 17"))) return TEST_FAIL;
    if (!check(C(2, 1), static_cast<T>(30), check_msg(get_type_name<T>(), "C(2,1) should be 30"))) return TEST_FAIL;

    return TEST_PASS;
}

template<typename T>
int test_blockdiag_gemm_transpose_b() {
    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 3, 2, 4}));

    BlockDiagMatrix<T> A(blocks);

    Matrix<T> B(Shape{3, 2});
    B(0, 0) = static_cast<T>(1); B(0, 1) = static_cast<T>(2);
    B(1, 0) = static_cast<T>(3); B(1, 1) = static_cast<T>(4);
    B(2, 0) = static_cast<T>(5); B(2, 1) = static_cast<T>(6);

    Matrix<T> C(Shape{2, 3}, static_cast<T>(0));

    MatrixMatrixProduct(A, B, C, static_cast<T>(1), static_cast<T>(0), "N", "T");

    if (!check(C(0, 0), static_cast<T>(5), check_msg(get_type_name<T>(), "C(0,0) should be 5"))) return TEST_FAIL;
    if (!check(C(0, 1), static_cast<T>(11), check_msg(get_type_name<T>(), "C(0,1) should be 11"))) return TEST_FAIL;
    if (!check(C(1, 2), static_cast<T>(39), check_msg(get_type_name<T>(), "C(1,2) should be 39"))) return TEST_FAIL;

    return TEST_PASS;
}

template<typename T>
int test_blockdiag_gemm_both_transpose() {
    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{3, 2}, {1, 3, 5, 2, 4, 6}));

    BlockDiagMatrix<T> A(blocks);

    Matrix<T> B(Shape{3, 3});
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            B(i, j) = static_cast<T>(i * 3 + j + 1);
        }
    }

    Matrix<T> C(Shape{2, 3}, static_cast<T>(0));

    MatrixMatrixProduct(A, B, C, static_cast<T>(1), static_cast<T>(0), "T", "T");

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            T val = C(i, j);
            if (std::isnan(val) || std::isinf(val)) {
                std::cout << "C(" << i << "," << j << ") is NaN or Inf!" << std::endl;
                std::cerr << check_msg(get_type_name<T>(), "test condition failed") << std::endl;
                return TEST_FAIL;
            }
        }
    }

    return TEST_PASS;
}

template<typename T>
int test_dense_times_blockdiag_gemm_simple() {
    Matrix<T> A(Shape{3, 2});
    A(0, 0) = static_cast<T>(1); A(0, 1) = static_cast<T>(2);
    A(1, 0) = static_cast<T>(3); A(1, 1) = static_cast<T>(4);
    A(2, 0) = static_cast<T>(5); A(2, 1) = static_cast<T>(6);

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 3}, {1, 4, 2, 5, 3, 6}));

    BlockDiagMatrix<T> B(blocks);

    Matrix<T> C(Shape{3, 3}, static_cast<T>(0));

    MatrixMatrixProduct(A, B, C);

    if (!check(C(0, 0), static_cast<T>(9), check_msg(get_type_name<T>(), "C(0,0) should be 9"))) return TEST_FAIL;
    if (!check(C(1, 1), static_cast<T>(26), check_msg(get_type_name<T>(), "C(1,1) should be 26"))) return TEST_FAIL;
    if (!check(C(2, 2), static_cast<T>(51), check_msg(get_type_name<T>(), "C(2,2) should be 51"))) return TEST_FAIL;

    return TEST_PASS;
}

template<typename T>
int test_dense_times_blockdiag_gemm_with_alpha_beta() {
    Matrix<T> A(Shape{2, 2});
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 3; A(1, 1) = 4;

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 3, 2, 4}));

    BlockDiagMatrix<T> B(blocks);

    Matrix<T> C(Shape{2, 2}, static_cast<T>(1));

    MatrixMatrixProduct(A, B, C, static_cast<T>(0.5), static_cast<T>(2));

    if (!check(C(0, 0), static_cast<T>(5.5), check_msg(get_type_name<T>(), "C(0,0) should be 5.5"))) return TEST_FAIL;
    if (!check(C(1, 1), static_cast<T>(13), check_msg(get_type_name<T>(), "C(1,1) should be 13"))) return TEST_FAIL;

    return TEST_PASS;
}

template<typename T>
int test_dense_times_blockdiag_gemm_transpose() {
    Matrix<T> A(Shape{2, 3});
    A(0, 0) = static_cast<T>(1); A(0, 1) = static_cast<T>(2); A(0, 2) = static_cast<T>(3);
    A(1, 0) = static_cast<T>(4); A(1, 1) = static_cast<T>(5); A(1, 2) = static_cast<T>(6);

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 3, 2, 4}));

    BlockDiagMatrix<T> B(blocks);

    Matrix<T> C(Shape{3, 2}, static_cast<T>(0));

    MatrixMatrixProduct(A, B, C, static_cast<T>(1), static_cast<T>(0), "T", "N");

    if (!check(C(0, 0), static_cast<T>(13), check_msg(get_type_name<T>(), "C(0,0) should be 13"))) return TEST_FAIL;
    if (!check(C(1, 0), static_cast<T>(17), check_msg(get_type_name<T>(), "C(1,0) should be 17"))) return TEST_FAIL;
    if (!check(C(2, 1), static_cast<T>(30), check_msg(get_type_name<T>(), "C(2,1) should be 30"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_dense_mixed_shapes_gemm() {
    std::vector<Matrix<T>> blocks;
    Matrix<T> b1(Shape{2, 2});
    b1(0, 0) = 1.0; b1(0, 1) = 2.0;
    b1(1, 0) = 3.0; b1(1, 1) = 4.0;
    blocks.push_back(b1);
    blocks.push_back(b1);  // Same block twice
    blocks.push_back(b1);  // Same block three times

    Matrix<T> b2(Shape{3, 3});
    b2(0, 0) = 5.0; b2(0, 1) = 0.0; b2(0, 2) = 0.0;
    b2(1, 0) = 0.0; b2(1, 1) = 5.0; b2(1, 2) = 0.0;
    b2(2, 0) = 0.0; b2(2, 1) = 0.0; b2(2, 2) = 5.0;
    blocks.push_back(b2);  // Different shaped block

    BlockDiagMatrix<T> A(blocks);

    Matrix<T> B(Shape{9, 2}, static_cast<T>(1.0));
    Matrix<T> C(Shape{9, 2}, static_cast<T>(0));

    MatrixMatrixProduct("N", "N", static_cast<T>(1.0), A, B, static_cast<T>(0.0), C);

    if (!check(C(0, 0), static_cast<T>(3), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(C(1, 0), static_cast<T>(7), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(C(2, 0), static_cast<T>(3), check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check(C(6, 0), static_cast<T>(5), check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_dense_blockdiag_mixed_shapes_gemm() {
    std::vector<Matrix<T>> blocks;
    Matrix<T> b1(Shape{2, 2});
    b1(0, 0) = 1.0; b1(0, 1) = 2.0;
    b1(1, 0) = 3.0; b1(1, 1) = 4.0;
    blocks.push_back(b1);
    blocks.push_back(b1);  // Same block twice
    blocks.push_back(b1);  // Same block three times

    Matrix<T> b2(Shape{3, 3});
    b2(0, 0) = 5.0; b2(0, 1) = 0.0; b2(0, 2) = 0.0;
    b2(1, 0) = 0.0; b2(1, 1) = 5.0; b2(1, 2) = 0.0;
    b2(2, 0) = 0.0; b2(2, 1) = 0.0; b2(2, 2) = 5.0;
    blocks.push_back(b2);  // Different shaped block

    BlockDiagMatrix<T> A(blocks);

    Matrix<T> A_dense(Shape{9, 9}, static_cast<T>(1.0));
    Matrix<T> C(Shape{9, 9}, static_cast<T>(0));

    MatrixMatrixProduct("N", "N", static_cast<T>(1.0), A_dense, A, static_cast<T>(0.0), C);

    if (!check(C(0, 0), static_cast<T>(4), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(C(0, 1), static_cast<T>(6), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(C(6, 6), static_cast<T>(5), check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Main
// ============================================================================

int main(){
    int total_failures = 0;

    // Basic GEMM tests with double and float
    total_failures += test_gemm_zero_v_cpp<double>();
    total_failures += test_gemm_zero_v_cpp<float>();

    // Additional parameter tests
    total_failures += test_gemm_with_scaling<double>();
    total_failures += test_gemm_with_scaling<float>();
    total_failures += test_gemm_negative_alpha<double>();
    total_failures += test_gemm_negative_alpha<float>();

    // Scaling factor tests
    total_failures += test_gemm_with_beta<double>();
    total_failures += test_gemm_with_beta<float>();
    total_failures += test_gemm_with_alpha_beta<double>();
    total_failures += test_gemm_with_alpha_beta<float>();

    // SymMatrixMatrixProduct tests
    total_failures += test_symm_left<double>();
    total_failures += test_symm_left<float>();
    total_failures += test_symm_right<double>();
    total_failures += test_symm_right<float>();
    total_failures += test_symm_with_beta<double>();
    total_failures += test_symm_with_beta<float>();

    // C-level pointer-based tests
    total_failures += test_c_gemm_pointer<double>();
    total_failures += test_c_gemm_pointer<float>();
    total_failures += test_c_gemm_negative_alpha<double>();
    total_failures += test_c_gemm_negative_alpha<float>();
    total_failures += test_c_gemm_with_scaling<double>();
    total_failures += test_c_gemm_with_scaling<float>();
    total_failures += test_c_gemm_with_beta<double>();
    total_failures += test_c_gemm_with_beta<float>();

    // C-level SymMatrixMatrixProduct tests (both branches)
    total_failures += test_c_symm_left<double>();
    total_failures += test_c_symm_left<float>();
    total_failures += test_c_symm_right<double>();
    total_failures += test_c_symm_right<float>();
    total_failures += test_c_symm_with_beta_left<double>();
    total_failures += test_c_symm_with_beta_left<float>();
    total_failures += test_c_symm_with_beta_right<double>();
    total_failures += test_c_symm_with_beta_right<float>();

    // BlockDiagMatrix GEMM tests
    total_failures += test_blockdiag_gemm_simple<double>();
    total_failures += test_blockdiag_gemm_simple<float>();
    total_failures += test_blockdiag_gemm_with_beta<double>();
    total_failures += test_blockdiag_gemm_with_beta<float>();
    total_failures += test_blockdiag_gemm_scaling<double>();
    total_failures += test_blockdiag_gemm_scaling<float>();
    total_failures += test_blockdiag_gemm_varying_blocks<double>();
    total_failures += test_blockdiag_gemm_varying_blocks<float>();
    total_failures += test_blockdiag_gemm_transpose_a<double>();
    total_failures += test_blockdiag_gemm_transpose_a<float>();
    total_failures += test_blockdiag_gemm_transpose_b<double>();
    total_failures += test_blockdiag_gemm_transpose_b<float>();
    total_failures += test_blockdiag_gemm_both_transpose<double>();
    total_failures += test_blockdiag_gemm_both_transpose<float>();
    total_failures += test_dense_times_blockdiag_gemm_simple<double>();
    total_failures += test_dense_times_blockdiag_gemm_simple<float>();
    total_failures += test_dense_times_blockdiag_gemm_with_alpha_beta<double>();
    total_failures += test_dense_times_blockdiag_gemm_with_alpha_beta<float>();
    total_failures += test_dense_times_blockdiag_gemm_transpose<double>();
    total_failures += test_dense_times_blockdiag_gemm_transpose<float>();
    total_failures += test_blockdiag_dense_mixed_shapes_gemm<double>();
    total_failures += test_blockdiag_dense_mixed_shapes_gemm<float>();
    total_failures += test_dense_blockdiag_mixed_shapes_gemm<double>();
    total_failures += test_dense_blockdiag_mixed_shapes_gemm<float>();

    if (total_failures > 0) {
        std::cerr << "cpu/blas/level3 tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All cpu/blas/level3 tests passed!" << std::endl;
    return TEST_PASS;
};
