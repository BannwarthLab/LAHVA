#include "test_common.h"
#include "impl/blas/cpu/lapack.hpp"
#include "impl/blas/cpu/lapack.h"
#include <cmath>
#include <random>

using namespace lahva::cpu;

// Use standard tolerances for LAPACK tests
const double TOLERANCE = DEFAULT_DOUBLE_TOLERANCE;

// Fill matrix with random values
template<typename T>
void fill_random(Matrix<T>& m) {
    std::random_device rd;
    std::minstd_rand eng(rd());
    std::uniform_real_distribution<> distr(0.1, 1.0);
    for (size_t i = 0; i < m.size(); i++) {
        m.data()[i] = (T)distr(eng);
    }
}

// Overload for LowTriMatrix
template<typename T>
void fill_random(LowTriMatrix<T>& m) {
    std::random_device rd;
    std::minstd_rand eng(rd());
    std::uniform_real_distribution<> distr(0.1, 1.0);
    for (size_t i = 0; i < m.size(); i++) {
        m.data()[i] = (T)distr(eng);
    }
}

// Create a positive definite matrix
template<typename T>
void make_positive_definite(Matrix<T>& m) {
    // m = A * A^T which is always positive definite
    size_t n = m.shape().first;
    Matrix<T> temp(Shape(n, n));
    fill_random(temp);

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
    int failures = 0;

    Shape sa(4, 4);
    Shape sb(4, 2);

    Matrix<T> A(sa);
    Matrix<T> B(sb);

    make_positive_definite(A);
    fill_random(B);

    try {
        SolvePosSysLinEquations(A, B);
        // If solving succeeded, B now contains the solution
    } catch (...) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Solve General Linear System Tests
// ============================================================================

template <typename T>
int test_solve_gen_sys() {
    int failures = 0;

    Shape sa(4, 4);
    Shape sb(4, 2);

    Matrix<T> A(sa);
    Matrix<T> B(sb);

    fill_random(A);
    fill_random(B);

    try {
        SolveGenSysLinEquations("N", A, B);
    } catch (...) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_solve_gen_sys_transpose() {
    int failures = 0;

    Shape sa(4, 4);
    Shape sb(4, 2);

    Matrix<T> A(sa);
    Matrix<T> B(sb);

    fill_random(A);
    fill_random(B);

    try {
        // Solve with transposed A
        SolveGenSysLinEquations("T", A, B);
    } catch (...) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_solve_gen_sys_alt_order() {
    int failures = 0;

    Shape sa(4, 4);
    Shape sb(4, 2);

    Matrix<T> A(sa);
    Matrix<T> B(sb);

    fill_random(A);
    fill_random(B);

    try {
        SolveGenSysLinEquations(A, B, "N");
    } catch (...) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Solve with Lower Triangular Matrix Tests
// ============================================================================

template <typename T>
int test_solve_lower_tri() {
    int failures = 0;

    Shape sb(4, 2);

    LowTriMatrix<T> A(4);
    Matrix<T> B(sb);

    fill_random(A);
    fill_random(B);

    try {
        SolveGenSysLinEquations("N", A, B);
    } catch (...) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_solve_lower_tri_transpose() {
    int failures = 0;

    Shape sb(4, 2);

    LowTriMatrix<T> A(4);
    Matrix<T> B(sb);

    fill_random(A);
    fill_random(B);

    try {
        SolveGenSysLinEquations("T", A, B);
    } catch (...) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Solve Symmetric System Tests
// ============================================================================

template <typename T>
int test_solve_sym_sys() {
    int failures = 0;

    Shape sb(4, 2);

    LowTriMatrix<T> A(4);
    Matrix<T> B(sb);

    fill_random(A);
    fill_random(B);

    try {
        SolveSymSysLinEquations(A, B);
    } catch (...) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Triangular Matrix Inversion Tests
// ============================================================================

template <typename T>
int test_invert_tri_matrix() {
    int failures = 0;

    Shape s(4, 4);
    Matrix<T> A(s);

    fill_random(A);
    // Make it lower triangular
    for (size_t i = 0; i < A.shape().first; i++) {
        for (size_t j = i + 1; j < A.shape().second; j++) {
            A(i, j) = (T)0.0;
        }
    }

    try {
        InvertTriMatrix(A);
        // Inversion succeeded
    } catch (...) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Symmetric Eigenvalue Decomposition Tests
// ============================================================================

template <typename T>
int test_sym_eigenvalue_decomp() {
    int failures = 0;

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
            failures += 1;
        }
    } catch (...) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_sym_eigenvalue_no_vectors() {
    int failures = 0;

    Shape s(4, 4);
    Matrix<T> A(s);
    Vector<T> eigenvalues(4);

    make_positive_definite(A);

    try {
        // Only compute eigenvalues, not eigenvectors
        SymEigenvalueDecomposition(A, eigenvalues, 'N');
    } catch (...) {
        failures += 1;
    }

    return failures;
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
    int failures = 0;

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
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_c_solve_gen_sys_default() {
    int failures = 0;

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
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_c_solve_gen_sys_transpose() {
    int failures = 0;

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
        failures += 1;
    }

    return failures;
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

    if (total_failures == 0) {
        std::cout << "All CPU LAPACK tests passed!" << std::endl;
    } else {
        std::cout << "CPU LAPACK tests: " << total_failures << " failures" << std::endl;
    }

    return total_failures;
}
