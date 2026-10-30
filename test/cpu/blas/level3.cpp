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

// Non-square operands
template <typename T>
int test_gemm_transpose(const char *Ta, const char *Tb) {
    bool ta = Ta[0] == 'T';
    bool tb = Tb[0] == 'T';

    Matrix<T> A(ta ? Shape(K,M) : Shape(M,K));
    Matrix<T> B(tb ? Shape(N,K) : Shape(K,N));
    Matrix<T> C(Shape(M,N), (T)0.0);
    for (size_t i = 0; i < A.size(); i++) A.data()[i] = (T)(1 + i);
    for (size_t i = 0; i < B.size(); i++) B.data()[i] = (T)(1 + 2 * i);

    MatrixMatrixProduct(Ta, Tb, (T)1.0, A, B, (T)0.0, C);

    Matrix<T> expected(Shape(M,N), (T)0.0);
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t l = 0; l < K; l++) {
                expected(i,j) += (ta ? A(l,i) : A(i,l)) * (tb ? B(j,l) : B(l,j));
            }
        }
    }

    if (!check(C.data(), expected.data(), M*N, check_msg(get_type_name<T>(), std::string(Ta) + Tb))) return TEST_FAIL;

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

    if (!check(C(0, 0), static_cast<T>(5), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(C(0, 1), static_cast<T>(17), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(C(0, 2), static_cast<T>(29), check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check(C(2, 0), static_cast<T>(39), check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;
    if (!check(C(3, 2), static_cast<T>(173), check_msg(get_type_name<T>(), "check 5"))) return TEST_FAIL;

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

    if (!check(C(0, 0), static_cast<T>(8), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(C(1, 1), static_cast<T>(10), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

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

    if (!check(C(0, 0), static_cast<T>(8), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(C(1, 1), static_cast<T>(12), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

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

    if (!check(C(0, 0), static_cast<T>(2), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(C(1, 0), static_cast<T>(11), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(C(2, 1), static_cast<T>(34), check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

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

    if (!check(C(0, 0), static_cast<T>(13), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(C(1, 0), static_cast<T>(17), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(C(2, 1), static_cast<T>(30), check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

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

    if (!check(C(0, 0), static_cast<T>(5), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(C(0, 1), static_cast<T>(11), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(C(1, 2), static_cast<T>(39), check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

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

    Matrix<T> expected(Shape{2, 3}, {22, 28, 49, 64, 76, 100});

    if (!check(C(0, 0), static_cast<T>(22), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(C(0, 1), static_cast<T>(49), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(C(0, 2), static_cast<T>(76), check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check(C(1, 0), static_cast<T>(28), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(C(1, 1), static_cast<T>(64), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(C(1, 2), static_cast<T>(100), check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

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

    if (!check(C(0, 0), static_cast<T>(9), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(C(1, 1), static_cast<T>(26), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(C(2, 2), static_cast<T>(51), check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

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

    if (!check(C(0, 0), static_cast<T>(5.5), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(C(1, 1), static_cast<T>(13), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

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

    if (!check(C(0, 0), static_cast<T>(13), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(C(1, 0), static_cast<T>(17), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(C(2, 1), static_cast<T>(30), check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

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
// BlockDiagMatrix GEMM Transpose Tests with Several Rectangular Blocks
// ============================================================================

// Row and column offsets differ between these blocks, so swapping them changes the result
template <typename T>
BlockDiagMatrix<T> make_rectangular_blockdiag() {
    std::vector<Matrix<T>> blocks;
    T value = static_cast<T>(1);
    for (const Shape &shape : {Shape{2, 3}, Shape{3, 1}, Shape{1, 2}, Shape{1, 2}}) {
        Matrix<T> block(shape);
        for (size_t i = 0; i < shape.first; i++) {
            for (size_t j = 0; j < shape.second; j++) {
                block(i, j) = value++;
            }
        }
        blocks.push_back(block);
    }
    return BlockDiagMatrix<T>(blocks);
}

template <typename T>
Matrix<T> blockdiag_to_dense(const BlockDiagMatrix<T> &a) {
    Matrix<T> dense(a.shape(), static_cast<T>(0));
    for (size_t b = 0; b < a.num_blocks(); b++) {
        Shape shape = a.get_block_shape(b);
        for (size_t i = 0; i < shape.first; i++) {
            for (size_t j = 0; j < shape.second; j++) {
                dense(a.get_row_offsets()[b] + i, a.get_col_offsets()[b] + j) = a.get_block(b)(i, j);
            }
        }
    }
    return dense;
}

inline Shape op_shape(const Shape &shape, const char *trans) {
    return (trans[0] == 'N') ? shape : Shape{shape.second, shape.first};
}

template <typename T>
Matrix<T> make_filled_matrix(const Shape &shape) {
    Matrix<T> mat(shape);
    for (size_t i = 0; i < shape.first * shape.second; i++) {
        mat.data()[i] = static_cast<T>(1 + (i * 7) % 11);
    }
    return mat;
}

// Naive reference, independent of BLAS: C = alpha * op(A) * op(B) + beta * C
template <typename T>
void reference_gemm(const char *Ta, const char *Tb, const T alpha, const Matrix<T> &a, const Matrix<T> &b,
                    const T beta, Matrix<T> &c) {
    const size_t k = op_shape(a.shape(), Ta).second;
    for (size_t i = 0; i < c.shape().first; i++) {
        for (size_t j = 0; j < c.shape().second; j++) {
            T sum = static_cast<T>(0);
            for (size_t l = 0; l < k; l++) {
                const T a_il = (Ta[0] == 'N') ? a(i, l) : a(l, i);
                const T b_lj = (Tb[0] == 'N') ? b(l, j) : b(j, l);
                sum += a_il * b_lj;
            }
            c(i, j) = alpha * sum + beta * c(i, j);
        }
    }
}

template <typename T>
int test_blockdiag_gemm_multi_block_transpose(const char *Ta, const char *Tb) {
    const BlockDiagMatrix<T> A = make_rectangular_blockdiag<T>();
    const Shape op_a = op_shape(A.shape(), Ta);
    const size_t n_cols = 4;

    const Matrix<T> B = make_filled_matrix<T>(op_shape(Shape{op_a.second, n_cols}, Tb));
    Matrix<T> C(Shape{op_a.first, n_cols}, static_cast<T>(1));
    Matrix<T> expected(Shape{op_a.first, n_cols}, static_cast<T>(1));

    MatrixMatrixProduct(Ta, Tb, static_cast<T>(2), A, B, static_cast<T>(3), C);
    reference_gemm(Ta, Tb, static_cast<T>(2), blockdiag_to_dense(A), B, static_cast<T>(3), expected);

    const std::string desc = std::string("Ta=") + Ta + " Tb=" + Tb;
    if (!check(C.data(), expected.data(), static_cast<int>(op_a.first * n_cols), check_msg(get_type_name<T>(), desc))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_dense_times_blockdiag_gemm_multi_block_transpose(const char *Ta, const char *Tb) {
    const BlockDiagMatrix<T> B = make_rectangular_blockdiag<T>();
    const Shape op_b = op_shape(B.shape(), Tb);
    const size_t n_rows = 4;

    const Matrix<T> A = make_filled_matrix<T>(op_shape(Shape{n_rows, op_b.first}, Ta));
    Matrix<T> C(Shape{n_rows, op_b.second}, static_cast<T>(1));
    Matrix<T> expected(Shape{n_rows, op_b.second}, static_cast<T>(1));

    MatrixMatrixProduct(Ta, Tb, static_cast<T>(2), A, B, static_cast<T>(3), C);
    reference_gemm(Ta, Tb, static_cast<T>(2), A, blockdiag_to_dense(B), static_cast<T>(3), expected);

    const std::string desc = std::string("Ta=") + Ta + " Tb=" + Tb;
    if (!check(C.data(), expected.data(), static_cast<int>(n_rows * op_b.second), check_msg(get_type_name<T>(), desc))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// BlockDiagMatrix times BlockDiagMatrix GEMM Tests
// ============================================================================

template <typename T>
BlockDiagMatrix<T> make_blockdiag(const std::vector<Shape> &shapes, const char *trans = "N") {
    std::vector<Matrix<T>> blocks;
    T value = static_cast<T>(1);
    for (const Shape &shape : shapes) {
        Matrix<T> block(op_shape(shape, trans));
        for (size_t i = 0; i < block.shape().first * block.shape().second; i++) {
            block.data()[i] = value++;
        }
        blocks.push_back(block);
    }
    return BlockDiagMatrix<T>(blocks);
}

// Compares C = 2 * op(A) * op(B) + 3 * C with a naive product of the densified matrices.
// The shapes are those of op(A) and op(B), so that they conform for every transpose option.
template <typename T>
int check_blockdiag_times_blockdiag(const char *Ta, const char *Tb, const std::vector<Shape> &op_a_shapes, const std::vector<Shape> &op_b_shapes) {
    std::vector<Shape> c_shapes;
    for (size_t i = 0; i < op_a_shapes.size(); i++) {
        c_shapes.push_back(Shape{op_a_shapes[i].first, op_b_shapes[i].second});
    }

    const BlockDiagMatrix<T> A = make_blockdiag<T>(op_a_shapes, Ta);
    const BlockDiagMatrix<T> B = make_blockdiag<T>(op_b_shapes, Tb);
    if (product_block_shapes(Ta, Tb, A, B) != c_shapes) {
        std::cerr << check_msg(get_type_name<T>(), "product_block_shapes returned the wrong shapes") << std::endl;
        return TEST_FAIL;
    }
    BlockDiagMatrix<T> C = make_blockdiag<T>(c_shapes);
    Matrix<T> expected = blockdiag_to_dense(C);

    MatrixMatrixProduct(Ta, Tb, static_cast<T>(2), A, B, static_cast<T>(3), C);
    reference_gemm(Ta, Tb, static_cast<T>(2), blockdiag_to_dense(A), blockdiag_to_dense(B), static_cast<T>(3), expected);

    const Matrix<T> C_dense = blockdiag_to_dense(C);
    const Shape s = C_dense.shape();
    const std::string desc = std::string("Ta=") + Ta + " Tb=" + Tb;
    if (!check(C_dense.data(), expected.data(), static_cast<int>(s.first * s.second), check_msg(get_type_name<T>(), desc))) return TEST_FAIL;

    return TEST_PASS;
}

// Runs of equal shapes, a lone block, and rectangular blocks whose row and column offsets differ
template <typename T>
int test_blockdiag_times_blockdiag_mixed_shapes(const char *Ta, const char *Tb) {
    return check_blockdiag_times_blockdiag<T>(Ta, Tb,
        {Shape{2, 3}, Shape{2, 3}, Shape{3, 1}, Shape{1, 2}, Shape{1, 2}},
        {Shape{3, 4}, Shape{3, 4}, Shape{1, 2}, Shape{2, 2}, Shape{2, 2}});
}

template <typename T>
int test_blockdiag_times_blockdiag_uniform(const char *Ta, const char *Tb) {
    return check_blockdiag_times_blockdiag<T>(Ta, Tb,
        {Shape{2, 3}, Shape{2, 3}, Shape{2, 3}},
        {Shape{3, 2}, Shape{3, 2}, Shape{3, 2}});
}

// All blocks of A have the same shape, so only B can end a run of equal shapes
template <typename T>
int test_blockdiag_times_blockdiag_run_ended_by_b() {
    return check_blockdiag_times_blockdiag<T>("N", "N",
        {Shape{2, 2}, Shape{2, 2}, Shape{2, 2}},
        {Shape{2, 3}, Shape{2, 1}, Shape{2, 1}});
}

template <typename T>
int test_blockdiag_times_blockdiag_convenience() {
    const std::vector<Shape> shapes = {Shape{2, 2}, Shape{3, 3}};
    const BlockDiagMatrix<T> A = make_blockdiag<T>(shapes);
    const BlockDiagMatrix<T> B = make_blockdiag<T>(shapes);
    BlockDiagMatrix<T> C = make_blockdiag<T>(shapes);
    Matrix<T> expected(C.shape(), static_cast<T>(0));

    MatrixMatrixProduct(A, B, C);
    reference_gemm("N", "N", static_cast<T>(1), blockdiag_to_dense(A), blockdiag_to_dense(B), static_cast<T>(0), expected);

    const Matrix<T> C_dense = blockdiag_to_dense(C);
    if (!check(C_dense.data(), expected.data(), 25, check_msg(get_type_name<T>(), "default arguments"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_times_blockdiag_empty() {
    BlockDiagMatrix<T> A, B, C;

    MatrixMatrixProduct("N", "N", static_cast<T>(1), A, B, static_cast<T>(0), C);

    if (!check(static_cast<T>(C.num_blocks()), static_cast<T>(0), check_msg(get_type_name<T>(), "no blocks"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int check_blockdiag_times_blockdiag_throws(const char *Ta, const std::vector<Shape> &a_shapes, const std::vector<Shape> &b_shapes, const std::vector<Shape> &c_shapes) {
    const BlockDiagMatrix<T> A = make_blockdiag<T>(a_shapes);
    const BlockDiagMatrix<T> B = make_blockdiag<T>(b_shapes);
    BlockDiagMatrix<T> C = make_blockdiag<T>(c_shapes);

    try {
        MatrixMatrixProduct(Ta, "N", static_cast<T>(1), A, B, static_cast<T>(0), C);
    } catch (const std::invalid_argument &) {
        return TEST_PASS;
    }
    std::cerr << "MatrixMatrixProduct did not reject mismatched block-diagonal matrices" << std::endl;
    return TEST_FAIL;
}

template <typename T>
int test_blockdiag_times_blockdiag_mismatch() {
    // Different partitions with the same total size
    if (check_blockdiag_times_blockdiag_throws<T>("N", {Shape{2, 2}, Shape{2, 2}}, {Shape{1, 1}, Shape{3, 3}}, {Shape{2, 1}, Shape{2, 3}}) != TEST_PASS) return TEST_FAIL;
    // Different block counts
    if (check_blockdiag_times_blockdiag_throws<T>("N", {Shape{2, 2}}, {Shape{2, 2}, Shape{2, 2}}, {Shape{2, 2}}) != TEST_PASS) return TEST_FAIL;
    // Output block with the wrong shape
    if (check_blockdiag_times_blockdiag_throws<T>("N", {Shape{2, 3}}, {Shape{3, 2}}, {Shape{3, 3}}) != TEST_PASS) return TEST_FAIL;
    // Only conforms when A is not transposed
    if (check_blockdiag_times_blockdiag_throws<T>("T", {Shape{2, 3}}, {Shape{3, 2}}, {Shape{2, 2}}) != TEST_PASS) return TEST_FAIL;

    return TEST_PASS;
}

// An output matrix built from the returned shapes must be accepted by the product
template <typename T>
int test_product_block_shapes_builds_output() {
    const BlockDiagMatrix<T> A = make_blockdiag<T>({Shape{2, 3}, Shape{3, 1}});
    const BlockDiagMatrix<T> B = make_blockdiag<T>({Shape{2, 4}, Shape{3, 2}});

    BlockDiagMatrix<T> C(product_block_shapes("T", "N", A, B), static_cast<T>(0));
    Matrix<T> expected(C.shape(), static_cast<T>(0));

    MatrixMatrixProduct("T", "N", static_cast<T>(1), A, B, static_cast<T>(0), C);
    reference_gemm("T", "N", static_cast<T>(1), blockdiag_to_dense(A), blockdiag_to_dense(B), static_cast<T>(0), expected);

    const Matrix<T> C_dense = blockdiag_to_dense(C);
    const Shape s = C_dense.shape();
    if (!check(C_dense.data(), expected.data(), static_cast<int>(s.first * s.second), check_msg(get_type_name<T>(), "built from the shapes"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_product_block_shapes_empty() {
    const BlockDiagMatrix<T> A, B;

    if (!product_block_shapes("N", "N", A, B).empty()) {
        std::cerr << check_msg(get_type_name<T>(), "expected no shapes") << std::endl;
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int check_product_block_shapes_throws(const char *Ta, const std::vector<Shape> &a_shapes, const std::vector<Shape> &b_shapes) {
    const BlockDiagMatrix<T> A = make_blockdiag<T>(a_shapes);
    const BlockDiagMatrix<T> B = make_blockdiag<T>(b_shapes);

    try {
        product_block_shapes(Ta, "N", A, B);
    } catch (const std::invalid_argument &) {
        return TEST_PASS;
    }
    std::cerr << check_msg(get_type_name<T>(), "product_block_shapes did not reject mismatched matrices") << std::endl;
    return TEST_FAIL;
}

template <typename T>
int test_product_block_shapes_mismatch() {
    // Different partitions with the same total size
    if (check_product_block_shapes_throws<T>("N", {Shape{2, 2}, Shape{2, 2}}, {Shape{1, 1}, Shape{3, 3}}) != TEST_PASS) return TEST_FAIL;
    // Different block counts
    if (check_product_block_shapes_throws<T>("N", {Shape{2, 2}}, {Shape{2, 2}, Shape{2, 2}}) != TEST_PASS) return TEST_FAIL;
    // Only conforms when A is not transposed
    if (check_product_block_shapes_throws<T>("T", {Shape{2, 3}}, {Shape{3, 2}}) != TEST_PASS) return TEST_FAIL;

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

    // Transpose tests with non-square operands
    for (const char *Ta : {"N", "T"}) {
        for (const char *Tb : {"N", "T"}) {
            total_failures += test_gemm_transpose<double>(Ta, Tb);
            total_failures += test_gemm_transpose<float>(Ta, Tb);
        }
    }

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

    // BlockDiagMatrix GEMM transpose tests with several rectangular blocks
    for (const char *Ta : {"N", "T", "C"}) {
        for (const char *Tb : {"N", "T", "C"}) {
            total_failures += test_blockdiag_gemm_multi_block_transpose<double>(Ta, Tb);
            total_failures += test_blockdiag_gemm_multi_block_transpose<float>(Ta, Tb);
            total_failures += test_dense_times_blockdiag_gemm_multi_block_transpose<double>(Ta, Tb);
            total_failures += test_dense_times_blockdiag_gemm_multi_block_transpose<float>(Ta, Tb);
            total_failures += test_blockdiag_times_blockdiag_mixed_shapes<double>(Ta, Tb);
            total_failures += test_blockdiag_times_blockdiag_mixed_shapes<float>(Ta, Tb);
            total_failures += test_blockdiag_times_blockdiag_uniform<double>(Ta, Tb);
            total_failures += test_blockdiag_times_blockdiag_uniform<float>(Ta, Tb);
        }
    }

    // BlockDiagMatrix times BlockDiagMatrix GEMM tests
    total_failures += test_blockdiag_times_blockdiag_run_ended_by_b<double>();
    total_failures += test_blockdiag_times_blockdiag_run_ended_by_b<float>();
    total_failures += test_blockdiag_times_blockdiag_convenience<double>();
    total_failures += test_blockdiag_times_blockdiag_convenience<float>();
    total_failures += test_blockdiag_times_blockdiag_empty<double>();
    total_failures += test_blockdiag_times_blockdiag_empty<float>();
    total_failures += test_blockdiag_times_blockdiag_mismatch<double>();
    total_failures += test_blockdiag_times_blockdiag_mismatch<float>();
    total_failures += test_product_block_shapes_builds_output<double>();
    total_failures += test_product_block_shapes_builds_output<float>();
    total_failures += test_product_block_shapes_empty<double>();
    total_failures += test_product_block_shapes_empty<float>();
    total_failures += test_product_block_shapes_mismatch<double>();
    total_failures += test_product_block_shapes_mismatch<float>();

    if (total_failures > 0) {
        std::cerr << "cpu/blas/level3 tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All cpu/blas/level3 tests passed!" << std::endl;
    return TEST_PASS;
};
