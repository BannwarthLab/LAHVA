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

    if (stat == 0) {
        std::cout << "All CPU Level-3 BLAS tests passed!" << std::endl;
    } else {
        std::cout << "CPU Level-3 BLAS tests: " << stat << " failures" << std::endl;
    }

    return stat;
};
