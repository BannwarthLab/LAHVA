#include "test_common.h"
using namespace lahva::cpu;
#define M 10
#define N 5 
#define K 3

const double thr2 = 5.0e-15;
float vf[9] = {1.0, 4.0, 5.0, 0.0, 2.0, 6.0, 0.0, 0.0, 3.0};
double vd[9] = {1.0, 4.0, 5.0, 0.0, 2.0, 6.0, 0.0, 0.0, 3.0};
float *pf = vf;
double *pd = vd;
float vftri[6] = {1.0, 4.0, 5.0, 2.0, 6.0, 3.0};
double vdtri[6] = {1.0, 4.0, 5.0, 2.0, 6.0, 3.0};
float *pft = vftri;
double *pdt = vdtri;
Vector<float> pvf({1.0, 2.0, 3.0}) ;

template <typename T>
int test_gemm_zero_v_cpp() {
    int stat_ = 0;
    Shape sres(M,N);
    Shape sa(M,K);
    Shape sb(K,N);
    Matrix<T> A(sa, 1.0);
    Matrix<T> B(sb, 0.0);
    Matrix<T> C(sres, 1.0);

    MatrixMatrixProduct("N", "N", (T)1.0, A, B, (T)0.0, C);

    Matrix<T> Mres(sres, 0.0);

    if (!check(C.data(), Mres.data(), thr2, M*N,"Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    return stat_;
}

// ============================================================================
// Additional GEMM Tests with Different Transpose Options
// ============================================================================

template <typename T>
int test_gemm_with_scaling() {
    int stat_ = 0;
    Shape sres(M,N);
    Shape sa(M,K);
    Shape sb(K,N);

    Matrix<T> A(sa, (T)1.0);
    Matrix<T> B(sb, (T)2.0);
    Matrix<T> C(sres, (T)5.0);

    MatrixMatrixProduct("N", "N", (T)0.0, A, B, (T)1.0, C);

    // Result = 0.0*(A*B) + 1.0*C_old = 5.0
    if (!check((double)C(0,0), 5.0, thr2, "Error with zero alpha.")) stat_ += 1;

    return stat_;
}

template <typename T>
int test_gemm_negative_alpha() {
    int stat_ = 0;
    Shape sres(M,N);
    Shape sa(M,K);
    Shape sb(K,N);

    Matrix<T> A(sa, (T)1.0);
    Matrix<T> B(sb, (T)2.0);
    Matrix<T> C(sres, (T)1.0);

    MatrixMatrixProduct("N", "N", (T)(-1.0), A, B, (T)1.0, C);

    // Result = -1.0*(A*B) + 1.0*C = -(2*K) + 1 = 1 - 2*K
    if (!check((double)C(0,0), 1.0 - 2.0*K, thr2, "Error with negative alpha.")) stat_ += 1;

    return stat_;
}

template <typename T>
int test_gemm_with_beta() {
    int stat_ = 0;
    Shape sres(M,N);
    Shape sa(M,K);
    Shape sb(K,N);

    Matrix<T> A(sa, (T)1.0);
    Matrix<T> B(sb, (T)2.0);
    Matrix<T> C(sres, (T)3.0);

    MatrixMatrixProduct("N", "N", (T)1.0, A, B, (T)2.0, C);

    // Result = A*B + 2*C = (2*K + 6)
    if (!check((double)C(0,0), 2.0*K + 6.0, thr2, "Error with non-zero beta.")) stat_ += 1;

    return stat_;
}

template <typename T>
int test_gemm_with_alpha_beta() {
    int stat_ = 0;
    Shape sres(M,N);
    Shape sa(M,K);
    Shape sb(K,N);

    Matrix<T> A(sa, (T)1.0);
    Matrix<T> B(sb, (T)2.0);
    Matrix<T> C(sres, (T)3.0);

    MatrixMatrixProduct("N", "N", (T)0.5, A, B, (T)2.0, C);

    // Result = 0.5*(A*B) + 2*C = 0.5*(2*K) + 6 = K + 6
    if (!check((double)C(0,0), 0.5*2.0*K + 6.0, thr2, "Error with alpha and beta.")) stat_ += 1;

    return stat_;
}

// ============================================================================
// SymMatrixMatrixProduct Tests
// ============================================================================

template <typename T>
int test_symm_left() {
    int stat_ = 0;
    Shape sa(5, 5);
    Shape sb(5, 3);
    Shape sres(5, 3);

    Matrix<T> A(sa, (T)1.0);
    Matrix<T> B(sb, (T)2.0);
    Matrix<T> C(sres, (T)0.0);

    SymMatrixMatrixProduct(CblasLeft, (T)1.0, A, B, (T)0.0, C);

    // C = A*B where A is symmetric
    if (!check((double)C(0,0), 2.0 * 5, thr2, "Error in SymMatrixMatrixProduct (left).")) stat_ += 1;

    return stat_;
}

template <typename T>
int test_symm_right() {
    int stat_ = 0;
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
    if (!check((double)C(0,0), 2.0 * 3, thr2, "Error in SymMatrixMatrixProduct (right).")) stat_ += 1;

    return stat_;
}

template <typename T>
int test_symm_with_beta() {
    int stat_ = 0;
    Shape sa(5, 5);
    Shape sb(5, 3);
    Shape sres(5, 3);

    Matrix<T> A(sa, (T)1.0);
    Matrix<T> B(sb, (T)2.0);
    Matrix<T> C(sres, (T)3.0);

    SymMatrixMatrixProduct(CblasLeft, (T)1.0, A, B, (T)2.0, C);

    // C = A*B + 2*C_old = (2*5) + 2*(3) = 16
    if (!check((double)C(0,0), 10.0 + 6.0, thr2, "Error in SymMatrixMatrixProduct with beta.")) stat_ += 1;

    return stat_;
}

// ============================================================================
// C-level Pointer-based GEMM Tests
// ============================================================================

template <typename T>
int test_c_gemm_pointer() {
    int stat_ = 0;

    const int m = 3, n = 2, k = 4;
    T A[m*k];
    T B[k*n];
    T C[m*n];

    for(int i = 0; i < m*k; i++) A[i] = (T)1.0;
    for(int i = 0; i < k*n; i++) B[i] = (T)2.0;
    for(int i = 0; i < m*n; i++) C[i] = (T)0.0;

    MatrixMatrixProduct("N", "N", m, n, k, (T)1.0, A, B, (T)0.0, C);

    if (!check((double)C[0], 2.0 * k, thr2, "Error in C-level gemm.")) stat_ += 1;

    return stat_;
}

template <typename T>
int test_c_gemm_negative_alpha() {
    int stat_ = 0;

    const int m = 3, n = 2, k = 4;
    T A[m*k];
    T B[k*n];
    T C[m*n];

    for(int i = 0; i < m*k; i++) A[i] = (T)1.0;
    for(int i = 0; i < k*n; i++) B[i] = (T)2.0;
    for(int i = 0; i < m*n; i++) C[i] = (T)5.0;

    MatrixMatrixProduct("N", "N", m, n, k, (T)(-1.0), A, B, (T)1.0, C);

    // Result = -1.0*(A*B) + 1.0*C_old = -(2*k) + 5 = 5 - 8 = -3
    if (!check((double)C[0], 5.0 - 2.0*k, thr2, "Error in C-level gemm with negative alpha.")) stat_ += 1;

    return stat_;
}

template <typename T>
int test_c_gemm_with_scaling() {
    int stat_ = 0;

    const int m = 3, n = 2, k = 4;
    T A[m*k];
    T B[k*n];
    T C[m*n];

    for(int i = 0; i < m*k; i++) A[i] = (T)1.0;
    for(int i = 0; i < k*n; i++) B[i] = (T)2.0;
    for(int i = 0; i < m*n; i++) C[i] = (T)3.0;

    MatrixMatrixProduct("N", "N", m, n, k, (T)0.5, A, B, (T)2.0, C);

    // Result = 0.5*(A*B) + 2*C_old = 0.5*(2*k) + 2*3 = k + 6
    if (!check((double)C[0], 0.5*2.0*k + 6.0, thr2, "Error in C-level gemm with scaling.")) stat_ += 1;

    return stat_;
}

template <typename T>
int test_c_gemm_with_beta() {
    int stat_ = 0;

    const int m = 3, n = 2, k = 4;
    T A[m*k];
    T B[k*n];
    T C[m*n];

    for(int i = 0; i < m*k; i++) A[i] = (T)1.0;
    for(int i = 0; i < k*n; i++) B[i] = (T)2.0;
    for(int i = 0; i < m*n; i++) C[i] = (T)5.0;

    MatrixMatrixProduct("N", "N", m, n, k, (T)1.0, A, B, (T)3.0, C);

    // Result = A*B + 3*C_old = (2*k) + 3*5 = 2*4 + 15 = 23
    if (!check((double)C[0], 2.0*k + 15.0, thr2, "Error in C-level gemm with beta.")) stat_ += 1;

    return stat_;
}

// ============================================================================
// C-level SymMatrixMatrixProduct Tests (both branches)
// ============================================================================

template <typename T>
int test_c_symm_left() {
    int stat_ = 0;

    const int m = 5, n = 3;
    T A[m*m];
    T B[m*n];
    T C[m*n];

    for(int i = 0; i < m*m; i++) A[i] = (T)1.0;
    for(int i = 0; i < m*n; i++) B[i] = (T)2.0;
    for(int i = 0; i < m*n; i++) C[i] = (T)0.0;

    SymMatrixMatrixProduct(CblasLeft, m, n, (T)1.0, A, B, (T)0.0, C);

    // C = A*B where A is m x m
    if (!check((double)C[0], 2.0 * m, thr2, "Error in C-level SymMatrixMatrixProduct (left).")) stat_ += 1;

    return stat_;
}

template <typename T>
int test_c_symm_right() {
    int stat_ = 0;

    const int m = 3, n = 3;
    T A[n*n];
    T B[m*n];
    T C[m*n];

    for(int i = 0; i < n*n; i++) A[i] = (T)1.0;
    for(int i = 0; i < m*n; i++) B[i] = (T)2.0;
    for(int i = 0; i < m*n; i++) C[i] = (T)0.0;

    SymMatrixMatrixProduct(CblasRight, m, n, (T)1.0, A, B, (T)0.0, C);

    // C = B*A where A is n x n
    if (!check((double)C[0], 2.0 * n, thr2, "Error in C-level SymMatrixMatrixProduct (right).")) stat_ += 1;

    return stat_;
}

template <typename T>
int test_c_symm_with_beta_left() {
    int stat_ = 0;

    const int m = 5, n = 3;
    T A[m*m];
    T B[m*n];
    T C[m*n];

    for(int i = 0; i < m*m; i++) A[i] = (T)1.0;
    for(int i = 0; i < m*n; i++) B[i] = (T)2.0;
    for(int i = 0; i < m*n; i++) C[i] = (T)4.0;

    SymMatrixMatrixProduct(CblasLeft, m, n, (T)1.0, A, B, (T)2.0, C);

    // C = A*B + 2*C_old = (2*m) + 2*(4) = 10 + 8 = 18
    if (!check((double)C[0], 2.0*m + 8.0, thr2, "Error in C-level SymMatrixMatrixProduct with beta.")) stat_ += 1;

    return stat_;
}

template <typename T>
int test_c_symm_with_beta_right() {
    int stat_ = 0;

    const int m = 3, n = 3;
    T A[n*n];
    T B[m*n];
    T C[m*n];

    for(int i = 0; i < n*n; i++) A[i] = (T)1.0;
    for(int i = 0; i < m*n; i++) B[i] = (T)2.0;
    for(int i = 0; i < m*n; i++) C[i] = (T)4.0;

    SymMatrixMatrixProduct(CblasRight, m, n, (T)1.0, A, B, (T)2.0, C);

    // C = B*A + 2*C_old = (2*n) + 2*(4) = 6 + 8 = 14
    if (!check((double)C[0], 2.0*n + 8.0, thr2, "Error in C-level SymMatrixMatrixProduct with beta.")) stat_ += 1;

    return stat_;
}

template<typename T>
int test_blockdiag_gemm_simple() {
    int failures = 0;
    T tol = get_tolerance<T>();

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 3, 2, 4}));
    blocks.push_back(Matrix<T>(Shape{2, 2}, {5, 7, 6, 8}));

    BlockDiagMatrix<T> A(blocks);

    Matrix<T> B(Shape{4, 3});
    for (int i = 0; i < 12; i++) {
        B.data()[i] = static_cast<T>(i + 1);
    }

    Matrix<T> C(Shape{4, 3}, static_cast<T>(0));

    MatrixMatrixProduct(static_cast<T>(1), A, B, static_cast<T>(0), C);

    if (!check(C(0, 0), static_cast<T>(5), tol, "C(0,0) should be 5")) failures += 1;
    if (!check(C(0, 1), static_cast<T>(17), tol, "C(0,1) should be 17")) failures += 1;
    if (!check(C(0, 2), static_cast<T>(29), tol, "C(0,2) should be 29")) failures += 1;
    if (!check(C(2, 0), static_cast<T>(39), tol, "C(2,0) should be 39")) failures += 1;
    if (!check(C(3, 2), static_cast<T>(173), tol, "C(3,2) should be 173")) failures += 1;

    return failures;
}

template<typename T>
int test_blockdiag_gemm_with_beta() {
    int failures = 0;
    T tol = get_tolerance<T>();

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 2, 3, 4}));

    BlockDiagMatrix<T> A(blocks);

    Matrix<T> B(Shape{2, 2}, static_cast<T>(1));
    Matrix<T> C(Shape{2, 2}, static_cast<T>(2));

    MatrixMatrixProduct(static_cast<T>(1), A, B, static_cast<T>(2), C);

    if (!check(C(0, 0), static_cast<T>(8), tol, "C(0,0) should be 8")) failures += 1;
    if (!check(C(1, 1), static_cast<T>(10), tol, "C(1,1) should be 10")) failures += 1;

    return failures;
}

template<typename T>
int test_blockdiag_gemm_scaling() {
    int failures = 0;
    T tol = get_tolerance<T>();

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 2, 3, 4}));

    BlockDiagMatrix<T> A(blocks);

    Matrix<T> B(Shape{2, 2}, static_cast<T>(1));
    Matrix<T> C(Shape{2, 2}, static_cast<T>(0));

    MatrixMatrixProduct(static_cast<T>(2), A, B, static_cast<T>(0), C);

    if (!check(C(0, 0), static_cast<T>(8), tol, "C(0,0) should be 8")) failures += 1;
    if (!check(C(1, 1), static_cast<T>(12), tol, "C(1,1) should be 12")) failures += 1;

    return failures;
}

template<typename T>
int test_blockdiag_gemm_varying_blocks() {
    int failures = 0;
    T tol = get_tolerance<T>();

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{1, 1}, {2}));
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 2, 3, 4}));

    BlockDiagMatrix<T> A(blocks);

    Matrix<T> B(Shape{3, 2});
    for (int i = 0; i < 6; i++) {
        B.data()[i] = static_cast<T>(i + 1);
    }

    Matrix<T> C(Shape{3, 2}, static_cast<T>(0));

    MatrixMatrixProduct(static_cast<T>(1), A, B, static_cast<T>(0), C);

    if (!check(C(0, 0), static_cast<T>(2), tol, "C(0,0) should be 2")) failures += 1;
    if (!check(C(1, 0), static_cast<T>(11), tol, "C(1,0) should be 11")) failures += 1;
    if (!check(C(2, 1), static_cast<T>(34), tol, "C(2,1) should be 34")) failures += 1;

    return failures;
}

template<typename T>
int test_blockdiag_gemm_transpose_a() {
    int failures = 0;
    T tol = get_tolerance<T>();

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 3}, {1, 4, 2, 5, 3, 6}));

    BlockDiagMatrix<T> A(blocks);

    Matrix<T> B(Shape{2, 2});
    B(0, 0) = static_cast<T>(1); B(0, 1) = static_cast<T>(2);
    B(1, 0) = static_cast<T>(3); B(1, 1) = static_cast<T>(4);

    Matrix<T> C(Shape{3, 2}, static_cast<T>(0));

    MatrixMatrixProduct(A, B, C, static_cast<T>(1), static_cast<T>(0), "T", "N");

    if (!check(C(0, 0), static_cast<T>(13), tol, "C(0,0) should be 13")) failures += 1;
    if (!check(C(1, 0), static_cast<T>(17), tol, "C(1,0) should be 17")) failures += 1;
    if (!check(C(2, 1), static_cast<T>(30), tol, "C(2,1) should be 30")) failures += 1;

    return failures;
}

template<typename T>
int test_blockdiag_gemm_transpose_b() {
    int failures = 0;
    T tol = get_tolerance<T>();

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 3, 2, 4}));

    BlockDiagMatrix<T> A(blocks);

    Matrix<T> B(Shape{3, 2});
    B(0, 0) = static_cast<T>(1); B(0, 1) = static_cast<T>(2);
    B(1, 0) = static_cast<T>(3); B(1, 1) = static_cast<T>(4);
    B(2, 0) = static_cast<T>(5); B(2, 1) = static_cast<T>(6);

    Matrix<T> C(Shape{2, 3}, static_cast<T>(0));

    MatrixMatrixProduct(A, B, C, static_cast<T>(1), static_cast<T>(0), "N", "T");

    if (!check(C(0, 0), static_cast<T>(5), tol, "C(0,0) should be 5")) failures += 1;
    if (!check(C(0, 1), static_cast<T>(11), tol, "C(0,1) should be 11")) failures += 1;
    if (!check(C(1, 2), static_cast<T>(39), tol, "C(1,2) should be 39")) failures += 1;

    return failures;
}

template<typename T>
int test_blockdiag_gemm_both_transpose() {
    int failures = 0;

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
                failures += 1;
                std::cout << "C(" << i << "," << j << ") is NaN or Inf!" << std::endl;
            }
        }
    }

    return failures;
}

template<typename T>
int test_dense_times_blockdiag_gemm_simple() {
    int failures = 0;
    T tol = get_tolerance<T>();

    Matrix<T> A(Shape{3, 2});
    A(0, 0) = static_cast<T>(1); A(0, 1) = static_cast<T>(2);
    A(1, 0) = static_cast<T>(3); A(1, 1) = static_cast<T>(4);
    A(2, 0) = static_cast<T>(5); A(2, 1) = static_cast<T>(6);

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 3}, {1, 4, 2, 5, 3, 6}));

    BlockDiagMatrix<T> B(blocks);

    Matrix<T> C(Shape{3, 3}, static_cast<T>(0));

    MatrixMatrixProduct(static_cast<T>(1), A, B, static_cast<T>(0), C);

    if (!check(C(0, 0), static_cast<T>(9), tol, "C(0,0) should be 9")) failures += 1;
    if (!check(C(1, 1), static_cast<T>(26), tol, "C(1,1) should be 26")) failures += 1;
    if (!check(C(2, 2), static_cast<T>(51), tol, "C(2,2) should be 51")) failures += 1;

    return failures;
}

template<typename T>
int test_dense_times_blockdiag_gemm_with_alpha_beta() {
    int failures = 0;
    T tol = get_tolerance<T>();

    Matrix<T> A(Shape{2, 2});
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 3; A(1, 1) = 4;

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 3, 2, 4}));

    BlockDiagMatrix<T> B(blocks);

    Matrix<T> C(Shape{2, 2}, static_cast<T>(1));

    MatrixMatrixProduct(A, B, C, static_cast<T>(0.5), static_cast<T>(2));

    if (!check(C(0, 0), static_cast<T>(5.5), tol, "C(0,0) should be 5.5")) failures += 1;
    if (!check(C(1, 1), static_cast<T>(13), tol, "C(1,1) should be 13")) failures += 1;

    return failures;
}

template<typename T>
int test_dense_times_blockdiag_gemm_transpose() {
    int failures = 0;
    T tol = get_tolerance<T>();

    Matrix<T> A(Shape{2, 3});
    A(0, 0) = static_cast<T>(1); A(0, 1) = static_cast<T>(2); A(0, 2) = static_cast<T>(3);
    A(1, 0) = static_cast<T>(4); A(1, 1) = static_cast<T>(5); A(1, 2) = static_cast<T>(6);

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 3, 2, 4}));

    BlockDiagMatrix<T> B(blocks);

    Matrix<T> C(Shape{3, 2}, static_cast<T>(0));

    MatrixMatrixProduct(A, B, C, static_cast<T>(1), static_cast<T>(0), "T", "N");

    if (!check(C(0, 0), static_cast<T>(13), tol, "C(0,0) should be 13")) failures += 1;
    if (!check(C(1, 0), static_cast<T>(17), tol, "C(1,0) should be 17")) failures += 1;
    if (!check(C(2, 1), static_cast<T>(30), tol, "C(2,1) should be 30")) failures += 1;

    return failures;
}

template <typename T>
int test_blockdiag_mixed_shapes_gemm() {
    int failures = 0;
    const T tol = std::is_same<T, double>::value ? 1e-14 : 1e-6;

    // Create 3 blocks of shape 2x2 and 1 block of shape 3x3
    // Total size: 3×2 + 3 = 9 rows and cols
    std::vector<Matrix<T>> blocks;

    // Initialize 2x2 blocks with specific values
    Matrix<T> b1(Shape{2, 2});
    b1(0, 0) = 1.0; b1(0, 1) = 2.0;
    b1(1, 0) = 3.0; b1(1, 1) = 4.0;
    blocks.push_back(b1);
    blocks.push_back(b1);  // Same block twice
    blocks.push_back(b1);  // Same block three times

    // Initialize 3x3 block
    Matrix<T> b2(Shape{3, 3});
    b2(0, 0) = 5.0; b2(0, 1) = 0.0; b2(0, 2) = 0.0;
    b2(1, 0) = 0.0; b2(1, 1) = 5.0; b2(1, 2) = 0.0;
    b2(2, 0) = 0.0; b2(2, 1) = 0.0; b2(2, 2) = 5.0;
    blocks.push_back(b2);  // Different shaped block

    BlockDiagMatrix<T> A(blocks);

    // Create dense matrix for blockdiag × dense (9x2)
    Matrix<T> B(Shape{9, 2}, static_cast<T>(1.0));
    Matrix<T> C_blockdiag_dense(Shape{9, 2}, static_cast<T>(0));

    // blockdiag × dense: A * B
    MatrixMatrixProduct("N", "N", static_cast<T>(1.0), A, B, static_cast<T>(0.0), C_blockdiag_dense);

    // Each 2x2 block multiplied by [1,1]^T gives [3, 7]^T
    if (!check(C_blockdiag_dense(0, 0), static_cast<T>(3), tol, "BlockDiag×Dense(0,0)")) failures += 1;
    if (!check(C_blockdiag_dense(1, 0), static_cast<T>(7), tol, "BlockDiag×Dense(1,0)")) failures += 1;
    if (!check(C_blockdiag_dense(2, 0), static_cast<T>(3), tol, "BlockDiag×Dense(2,0)")) failures += 1;
    // 3x3 block (5*I) multiplied by [1,1,1]^T gives [5, 5, 5]^T
    if (!check(C_blockdiag_dense(6, 0), static_cast<T>(5), tol, "BlockDiag×Dense(6,0)")) failures += 1;

    // Create dense matrix for dense × blockdiag (9x9)
    Matrix<T> A_dense(Shape{9, 9}, static_cast<T>(1.0));
    Matrix<T> C_dense_blockdiag(Shape{9, 9}, static_cast<T>(0));

    // dense × blockdiag: A_dense * A
    MatrixMatrixProduct("N", "N", static_cast<T>(1.0), A_dense, A, static_cast<T>(0.0), C_dense_blockdiag);

    // First 2x2 block result: each row [1,1,1,1,1,1,1,1,1] dotted with block column
    // Row 0 of A_dense is all 1s, times first column of block [1,3] = 4
    if (!check(C_dense_blockdiag(0, 0), static_cast<T>(4), tol, "Dense×BlockDiag(0,0)")) failures += 1;
    if (!check(C_dense_blockdiag(0, 1), static_cast<T>(6), tol, "Dense×BlockDiag(0,1)")) failures += 1;
    // 3x3 block result: each row [1,...,1] times scaled identity = [5, 5, 5]
    if (!check(C_dense_blockdiag(6, 6), static_cast<T>(5), tol, "Dense×BlockDiag(6,6)")) failures += 1;

    return failures;
}

int main(){
    int stat = 0;

    // C++ wrapper tests (Matrix<T> interface)
    // Basic GEMM tests with double and float
    stat += test_gemm_zero_v_cpp<double>();
    stat += test_gemm_zero_v_cpp<float>();

    // Additional parameter tests
    stat += test_gemm_with_scaling<double>();
    stat += test_gemm_with_scaling<float>();
    stat += test_gemm_negative_alpha<double>();
    stat += test_gemm_negative_alpha<float>();

    // Scaling factor tests
    stat += test_gemm_with_beta<double>();
    stat += test_gemm_with_beta<float>();
    stat += test_gemm_with_alpha_beta<double>();
    stat += test_gemm_with_alpha_beta<float>();

    // SymMatrixMatrixProduct tests
    stat += test_symm_left<double>();
    stat += test_symm_left<float>();
    stat += test_symm_right<double>();
    stat += test_symm_right<float>();
    stat += test_symm_with_beta<double>();
    stat += test_symm_with_beta<float>();

    // C-level pointer-based tests
    stat += test_c_gemm_pointer<double>();
    stat += test_c_gemm_pointer<float>();
    stat += test_c_gemm_negative_alpha<double>();
    stat += test_c_gemm_negative_alpha<float>();
    stat += test_c_gemm_with_scaling<double>();
    stat += test_c_gemm_with_scaling<float>();
    stat += test_c_gemm_with_beta<double>();
    stat += test_c_gemm_with_beta<float>();

    // C-level SymMatrixMatrixProduct tests (both branches)
    stat += test_c_symm_left<double>();
    stat += test_c_symm_left<float>();
    stat += test_c_symm_right<double>();
    stat += test_c_symm_right<float>();
    stat += test_c_symm_with_beta_left<double>();
    stat += test_c_symm_with_beta_left<float>();
    stat += test_c_symm_with_beta_right<double>();
    stat += test_c_symm_with_beta_right<float>();

    // BlockDiagMatrix GEMM tests
    stat += test_blockdiag_gemm_simple<double>();
    stat += test_blockdiag_gemm_simple<float>();
    stat += test_blockdiag_gemm_with_beta<double>();
    stat += test_blockdiag_gemm_with_beta<float>();
    stat += test_blockdiag_gemm_scaling<double>();
    stat += test_blockdiag_gemm_scaling<float>();
    stat += test_blockdiag_gemm_varying_blocks<double>();
    stat += test_blockdiag_gemm_varying_blocks<float>();
    stat += test_blockdiag_gemm_transpose_a<double>();
    stat += test_blockdiag_gemm_transpose_a<float>();
    stat += test_blockdiag_gemm_transpose_b<double>();
    stat += test_blockdiag_gemm_transpose_b<float>();
    stat += test_blockdiag_gemm_both_transpose<double>();
    stat += test_blockdiag_gemm_both_transpose<float>();
    stat += test_dense_times_blockdiag_gemm_simple<double>();
    stat += test_dense_times_blockdiag_gemm_simple<float>();
    stat += test_dense_times_blockdiag_gemm_with_alpha_beta<double>();
    stat += test_dense_times_blockdiag_gemm_with_alpha_beta<float>();
    stat += test_dense_times_blockdiag_gemm_transpose<double>();
    stat += test_dense_times_blockdiag_gemm_transpose<float>();
    stat += test_blockdiag_mixed_shapes_gemm<double>();
    stat += test_blockdiag_mixed_shapes_gemm<float>();

    if (stat == 0) {
        std::cout << "All CPU Level-3 BLAS tests passed!" << std::endl;
    } else {
        std::cout << "CPU Level-3 BLAS tests: " << stat << " failures" << std::endl;
    }

    return stat;
};
