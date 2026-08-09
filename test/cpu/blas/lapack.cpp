#include "test_common.h"
#include "array_utils.hpp"

using namespace lahva::cpu;
using lahva::Shape;

// Create a positive definite matrix
template<typename T>
void make_positive_definite(Matrix<T>& m) {
    // m = A * A^T which is always positive definite
    size_t n = m.shape().first;
    Matrix<T> temp(Shape(n, n));
    fill_with_rd_values(temp);

    // Simple approach: m(i,j) = sum_k temp(i,k)*temp(j,k)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            m(i, j) = 0.0;
            for (size_t k = 0; k < n; k++) {
                m(i, j) += temp(i, k) * temp(j, k);
            }
        }
    }
}

// ============================================================================
// Solve Positive Definite System Tests
// ============================================================================

template <typename T>
int test_solve_pos_sys() {

    Shape sa(4, 4);
    Shape sb(4, 2);

    Matrix<T> A(sa);
    Matrix<T> B(sb);

    make_positive_definite(A);
    fill_with_rd_values(B);

    try {
        SolvePosSysLinEquations(A, B);
    } catch (...) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

// ============================================================================
// Solve General Linear System Tests
// ============================================================================

template <typename T>
int test_solve_gen_sys() {

    Shape sa(4, 4);
    Shape sb(4, 2);

    Matrix<T> A(sa);
    Matrix<T> B(sb);

    fill_with_rd_values(A);
    fill_with_rd_values(B);

    try {
        SolveGenSysLinEquations("N", A, B);
    } catch (...) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_solve_gen_sys_transpose() {

    Shape sa(4, 4);
    Shape sb(4, 2);

    Matrix<T> A(sa);
    Matrix<T> B(sb);

    fill_with_rd_values(A);
    fill_with_rd_values(B);

    try {
        // Solve with transposed A
        SolveGenSysLinEquations("T", A, B);
    } catch (...) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_solve_gen_sys_alt_order() {

    Shape sa(4, 4);
    Shape sb(4, 2);

    Matrix<T> A(sa);
    Matrix<T> B(sb);

    fill_with_rd_values(A);
    fill_with_rd_values(B);

    try {
        SolveGenSysLinEquations(A, B, "N");
    } catch (...) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

// ============================================================================
// Solve with Lower Triangular Matrix Tests
// ============================================================================

template <typename T>
int test_solve_lower_tri() {

    Shape sb(4, 2);

    LowTriMatrix<T> A(4);
    Matrix<T> B(sb);

    fill_with_rd_values(A);
    fill_with_rd_values(B);

    try {
        SolveGenSysLinEquations("N", A, B);
    } catch (...) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_solve_lower_tri_transpose() {

    Shape sb(4, 2);

    LowTriMatrix<T> A(4);
    Matrix<T> B(sb);

    fill_with_rd_values(A);
    fill_with_rd_values(B);

    try {
        SolveGenSysLinEquations("T", A, B);
    } catch (...) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

// ============================================================================
// Solve Symmetric System Tests
// ============================================================================

template <typename T>
int test_solve_sym_sys() {

    Shape sb(4, 2);

    LowTriMatrix<T> A(4);
    Matrix<T> B(sb);

    fill_with_rd_values(A);
    fill_with_rd_values(B);

    try {
        SolveSymSysLinEquations(A, B);
    } catch (...) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

// ============================================================================
// Triangular Matrix Inversion Tests
// ============================================================================

template <typename T>
int test_invert_tri_matrix() {

    Shape s(4, 4);
    Matrix<T> A(s);

    fill_with_rd_values(A);
    // Make it lower triangular
    for (size_t i = 0; i < A.shape().first; i++) {
        for (size_t j = i + 1; j < A.shape().second; j++) {
            A(i, j) = (T)0.0;
        }
    }

    try {
        InvertTriMatrix(A);
    } catch (...) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

// ============================================================================
// Symmetric Eigenvalue Decomposition Tests
// ============================================================================

template <typename T>
int test_sym_eigenvalue_decomp() {

    Shape s(4, 4);
    Matrix<T> A(s);
    Vector<T> eigenvalues(4);

    make_positive_definite(A);

    try {
        SymEigenvalueDecomposition(A, eigenvalues, 'V');

        // Check that eigenvalues are valid (should be positive)
        bool valid = true;
        for (size_t i = 0; i < eigenvalues.size(); i++) {
            if (eigenvalues[i] <= 0) {
                valid = false;
                break;
            }
        }

        if (!valid) {
            return TEST_FAIL;
        }
    } catch (...) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_sym_eigenvalue_no_vectors() {

    Shape s(4, 4);
    Matrix<T> A(s);
    Vector<T> eigenvalues(4);

    make_positive_definite(A);

    try {
        // Only compute eigenvalues, not eigenvectors
        SymEigenvalueDecomposition(A, eigenvalues, 'N');
    } catch (...) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

// Create a diagonal dominant matrix (guaranteed non-singular)
template <typename T>
void create_diagonal_dominant_matrix(T* A, int n) {
    // Fill with small random values
    for (int i = 0; i < n*n; i++) {
        A[i] = (T)0.1 + (i % 5) * (T)0.05;
    }
    // Make diagonal dominant
    for (int i = 0; i < n; i++) {
        A[i*n + i] += (T)(2.0 * n);
    }
}

// ============================================================================
// C-Level API Tests (Direct raw pointer interface)
// ============================================================================

template <typename T>
int test_c_solve_gen_sys() {

    const int n = 4;
    const int nrhs = 2;

    T A[n*n];
    T B[n*nrhs];

    create_diagonal_dominant_matrix<T>(A, n);

    for (int i = 0; i < n*nrhs; i++) {
        B[i] = (T)1.0 + (i % 2) * (T)0.5;
    }

    try {
        lahva::cpu::SolveGenSysLinEquations("N", n, A, nrhs, B);
    } catch (...) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_c_solve_gen_sys_default() {
    const int n = 4;
    const int nrhs = 2;

    T A[n*n];
    T B[n*nrhs];

    create_diagonal_dominant_matrix<T>(A, n);

    for (int i = 0; i < n*nrhs; i++) {
        B[i] = (T)1.0 + (i % 2) * (T)0.5;
    }

    try {
        lahva::cpu::SolveGenSysLinEquations(n, A, nrhs, B, "N");
    } catch (...) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_c_solve_gen_sys_transpose() {

    const int n = 4;
    const int nrhs = 2;

    T A[n*n];
    T B[n*nrhs];

    create_diagonal_dominant_matrix<T>(A, n);

    for (int i = 0; i < n*nrhs; i++) {
        B[i] = (T)1.0 + (i % 2) * (T)0.5;
    }

    try {
        lahva::cpu::SolveGenSysLinEquations("T", n, A, nrhs, B);
    } catch (...) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    int total_failures = 0;

    // C-Level API tests
    total_failures += test_c_solve_gen_sys<double>();
    total_failures += test_c_solve_gen_sys<float>();
    total_failures += test_c_solve_gen_sys_default<double>();
    total_failures += test_c_solve_gen_sys_default<float>();
    total_failures += test_c_solve_gen_sys_transpose<double>();
    total_failures += test_c_solve_gen_sys_transpose<float>();

    // Solve positive definite system tests
    total_failures += test_solve_pos_sys<double>();
    total_failures += test_solve_pos_sys<float>();

    // Solve general linear system tests
    total_failures += test_solve_gen_sys<double>();
    total_failures += test_solve_gen_sys<float>();
    total_failures += test_solve_gen_sys_transpose<double>();
    total_failures += test_solve_gen_sys_transpose<float>();
    total_failures += test_solve_gen_sys_alt_order<double>();
    total_failures += test_solve_gen_sys_alt_order<float>();

    // Solve with lower triangular matrix tests
    total_failures += test_solve_lower_tri<double>();
    total_failures += test_solve_lower_tri<float>();
    total_failures += test_solve_lower_tri_transpose<double>();
    total_failures += test_solve_lower_tri_transpose<float>();

    // Solve symmetric system tests
    total_failures += test_solve_sym_sys<double>();
    total_failures += test_solve_sym_sys<float>();

    // Triangular matrix inversion tests
    total_failures += test_invert_tri_matrix<double>();
    total_failures += test_invert_tri_matrix<float>();

    // Symmetric eigenvalue decomposition tests
    total_failures += test_sym_eigenvalue_decomp<double>();
    total_failures += test_sym_eigenvalue_decomp<float>();
    total_failures += test_sym_eigenvalue_no_vectors<double>();
    total_failures += test_sym_eigenvalue_no_vectors<float>();

    if (total_failures > 0) {
        std::cerr << "cpu/blas/lapack tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All cpu/blas/lapack tests passed!" << std::endl;
    return TEST_PASS;
}
