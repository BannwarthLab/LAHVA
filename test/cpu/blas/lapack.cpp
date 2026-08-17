#include "test_common.h"
#include "array_utils.hpp"
#include <iomanip>

#define M 4
#define N 2

using namespace lahva::cpu;
using lahva::Shape;

// ============================================================================
// Multiplication helpers
// ============================================================================

template <typename T>
Matrix<T> multiply_matrices(const Matrix<T>& A, const Matrix<T>& X, bool transpose = false) {
    size_t n = A.shape().first;
    size_t nrhs = X.shape().second;
    Matrix<T> result(Shape(n, nrhs));

    for (size_t j = 0; j < nrhs; j++) {
        for (size_t i = 0; i < n; i++) {
            T sum = 0.0;
            if (!transpose) {
                for (size_t k = 0; k < n; k++) {
                    sum += A(i, k) * X(k, j);
                }
            } else {
                for (size_t k = 0; k < n; k++) {
                    sum += A(k, i) * X(k, j);
                }
            }
            result(i, j) = sum;
        }
    }
    return result;
}

template <typename T>
Matrix<T> multiply_lower_tri(const LowTriMatrix<T>& A, const Matrix<T>& X, bool transpose = false) {
    size_t n = A.shape().first;
    size_t nrhs = X.shape().second;
    Matrix<T> result(Shape(n, nrhs));

    for (size_t j = 0; j < nrhs; j++) {
        for (size_t i = 0; i < n; i++) {
            T sum = 0.0;
            if (!transpose) {
                for (size_t k = 0; k <= i; k++) {
                    sum += A(i, k) * X(k, j);
                }
            } else {
                for (size_t k = i; k < n; k++) {
                    sum += A(k, i) * X(k, j);
                }
            }
            result(i, j) = sum;
        }
    }
    return result;
}

template <typename T>
Matrix<T> multiply_symmetric_matrix(const LowTriMatrix<T>& A, const Matrix<T>& X, bool transpose = false) {
    size_t n = A.shape().first;
    size_t nrhs = X.shape().second;
    Matrix<T> result(Shape(n, nrhs));

    for (size_t j = 0; j < nrhs; j++) {
        for (size_t i = 0; i < n; i++) {
            T sum = 0.0;
            if (!transpose) {
                for (size_t k = 0; k < n; k++) {
                    sum += A(i, k) * X(k, j);
                }
            } else {
                for (size_t k = 0; k < n; k++) {
                    sum += A(k, i) * X(k, j);
                }
            }
            result(i, j) = sum;
        }
    }
    return result;
}

template <typename T>
T* c_multiply_matrices(T* A, T* X, int n, int nrhs, bool transpose = false) {
    T* result = new T[n * nrhs];
    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < n; i++) {
            T sum = 0.0;
            if (!transpose) {
                for (int k = 0; k < n; k++) {
                    sum += A[k*n + i] * X[j*n + k];
                }
            } else {
                // A^T: A^T(i,k) = A(k,i) at A[i*n + k]
                for (int k = 0; k < n; k++) {
                    sum += A[i*n + k] * X[j*n + k];
                }
            }
            result[j*n + i] = sum;
        }
    }
    return result;
}

// ============================================================================
// Matrix helpers
// ============================================================================

template<typename T>
void make_positive_definite(Matrix<T>& m) {
    size_t n = m.shape().first;
    Matrix<T> temp(Shape(n, n));
    fill_with_rd_values(temp);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            m(i, j) = 0.0;
            for (size_t k = 0; k < n; k++) {
                m(i, j) += temp(i, k) * temp(j, k);
            }
        }
    }

    // Add strong diagonal dominance for better conditioning
    for (size_t i = 0; i < n; i++) {
        T sum = 0.0;
        for (size_t j = 0; j < n; j++) {
            if (i != j) sum += std::abs(m(i, j));
        }
        m(i, i) += sum + (T)(100.0 * n);
    }
}

// Create a diagonal dominant matrix (guaranteed non-singular)
template <typename T>
void create_diagonal_dominant_matrix(Matrix<T>& A) {
    size_t n = A.shape().first;

    // Fill with small random values
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            A(i, j) = (T)0.1 + ((i*n + j) % 5) * (T)0.05;
        }
    }

    // Make strongly diagonal dominant
    for (size_t i = 0; i < n; i++) {
        T sum = 0.0;
        for (size_t j = 0; j < n; j++) {
            if (i != j) sum += std::abs(A(i, j));
        }
        A(i, i) = sum + (T)(100.0 * n);
    }
}

// Create a diagonal dominant matrix from raw pointer (column-major)
template <typename T>
void create_diagonal_dominant_matrix(T* A, int n) {
    // Fill with small random values (column-major)
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            A[j*n + i] = (T)0.1 + ((j*n + i) % 5) * (T)0.05;
        }
    }

    // Make strongly diagonal dominant
    for (int i = 0; i < n; i++) {
        T sum = 0.0;
        for (int j = 0; j < n; j++) {
            if (i != j) sum += std::abs(A[j*n + i]);
        }
        A[i*n + i] = sum + (T)(100.0 * n);
    }
}

// ============================================================================
// Solve Positive Definite System Tests
// ============================================================================

template <typename T>
int test_solve_pos_sys() {

    Shape sa(M, M);
    Shape sb(M, N);

    Matrix<T> A(sa);
    Matrix<T> A_copy(sa);
    Matrix<T> B(sb);
    Matrix<T> B_original(sb);

    make_positive_definite(A);
    fill_with_rd_values(B);

    CopyVectors(A, A_copy);
    CopyVectors(B, B_original);

    SolvePosSysLinEquations(A, B);

    // Verify: A_copy * B = B_original
    Matrix<T> product = multiply_matrices(A_copy, B, false);
    size_t n = A_copy.shape().first;
    size_t nrhs = B_original.shape().second;
    if (!check<T, ToleranceType::LAPACK>(product.data(), B_original.data(), n*nrhs,
               check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Solve General Linear System Tests
// ============================================================================

template <typename T>
int test_solve_gen_sys() {

    Shape sa(M, M);
    Shape sb(M, N);

    Matrix<T> A(sa);
    Matrix<T> A_copy(sa);
    Matrix<T> B(sb);
    Matrix<T> B_original(sb);

    fill_with_rd_values(A);
    fill_with_rd_values(B);

    lahva::cpu::CopyVectors(A, A_copy);
    lahva::cpu::CopyVectors(B, B_original);

    SolveGenSysLinEquations("N", A, B);

    // Verify: A_copy * B = B_original
    Matrix<T> product = multiply_matrices(A_copy, B, false);
    size_t n = A_copy.shape().first;
    size_t nrhs = B_original.shape().second;
    if (!check<T, ToleranceType::LAPACK>(product.data(), B_original.data(), n*nrhs,
               check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_solve_gen_sys_transpose() {

    Shape sa(M, M);
    Shape sb(M, N);

    Matrix<T> A(sa);
    Matrix<T> A_copy(sa);
    Matrix<T> B(sb);
    Matrix<T> B_original(sb);

    fill_with_rd_values(A);
    fill_with_rd_values(B);

    CopyVectors(A, A_copy);
    CopyVectors(B, B_original);

    SolveGenSysLinEquations("T", A, B);

    // Verify: A_copy^T * B = B_original
    Matrix<T> product = multiply_matrices(A_copy, B, true);
    size_t n = A_copy.shape().first;
    size_t nrhs = B_original.shape().second;
    if (!check<T, ToleranceType::LAPACK>(product.data(), B_original.data(), n*nrhs,
               check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_solve_gen_sys_alt_order() {

    Shape sa(M, M);
    Shape sb(M, N);

    Matrix<T> A(sa);
    Matrix<T> A_copy(sa);
    Matrix<T> B(sb);
    Matrix<T> B_original(sb);

    fill_with_rd_values(A);
    fill_with_rd_values(B);

    CopyVectors(A, A_copy);
    CopyVectors(B, B_original);

    SolveGenSysLinEquations(A, B, "N");

    // Verify: A_copy * B = B_original
    Matrix<T> product = multiply_matrices(A_copy, B, false);
    size_t n = A_copy.shape().first;
    size_t nrhs = B_original.shape().second;
    if (!check<T, ToleranceType::LAPACK>(product.data(), B_original.data(), n*nrhs,
               check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Solve with Lower Triangular Matrix Tests
// ============================================================================

template <typename T>
int test_solve_lower_tri() {

    Shape sb(M, N);

    LowTriMatrix<T> A(M);
    LowTriMatrix<T> A_copy(M);
    Matrix<T> B(sb);
    Matrix<T> B_original(sb);

    fill_with_rd_values(A);
    fill_with_rd_values(B);

    CopyVectors(A, A_copy);
    CopyVectors(B, B_original);

    SolveGenSysLinEquations("N", A, B);

    // Verify: A_copy * B = B_original
    Matrix<T> product = multiply_lower_tri(A_copy, B, false);
    size_t n = A_copy.shape().first;
    size_t nrhs = B_original.shape().second;
    if (!check<T, ToleranceType::LAPACK>(product.data(), B_original.data(), n*nrhs,
               check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_solve_lower_tri_transpose() {

    Shape sb(M, N);

    LowTriMatrix<T> A(M);
    LowTriMatrix<T> A_copy(M);
    Matrix<T> B(sb);
    Matrix<T> B_original(sb);

    fill_with_rd_values(A);
    fill_with_rd_values(B);

    CopyVectors(A, A_copy);
    CopyVectors(B, B_original);

    SolveGenSysLinEquations("T", A, B);

    // Verify: A_copy^T * B = B_original
    Matrix<T> product = multiply_lower_tri(A_copy, B, true);
    size_t n = A_copy.shape().first;
    size_t nrhs = B_original.shape().second;
    if (!check<T, ToleranceType::LAPACK>(product.data(), B_original.data(), n*nrhs,
               check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Solve Symmetric System Tests
// ============================================================================

template <typename T>
int test_solve_sym_sys() {

    Shape sb(M, N);

    LowTriMatrix<T> A(M);
    LowTriMatrix<T> A_copy(M);
    Matrix<T> B(sb);
    Matrix<T> B_original(sb);

    fill_with_rd_values(A);
    fill_with_rd_values(B);

    CopyVectors(A, A_copy);
    CopyVectors(B, B_original);

    SolveSymSysLinEquations(A, B);

    // Verify: A_copy * B = B_original
    Matrix<T> product = multiply_symmetric_matrix(A_copy, B, false);
    size_t n = A_copy.shape().first;
    size_t nrhs = B_original.shape().second;
    if (!check<T, ToleranceType::LAPACK>(product.data(), B_original.data(), n*nrhs,
               check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Triangular Matrix Inversion Tests
// ============================================================================

template <typename T>
int test_invert_tri_matrix() {

    Shape s(M, M);
    Matrix<T> A(s);
    Matrix<T> A_original(s);

    fill_with_rd_values(A);

    // Make A lower triangular
    size_t n = A.shape().first;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            A(i, j) = (T)0.0;
        }
    }

    lahva::cpu::CopyVectors(A, A_original);

    InvertTriMatrix(A);

    Matrix<T> product = multiply_matrices(A_original, A, false);

    Matrix<T> identity(Shape(n, n), 0.0);
    T one = (T)1.0;
    for (size_t i = 0; i < n; i++) {
        identity(i, i) = one;
    }

    // Verify A_original * A = identity
    if (!check<T, ToleranceType::LAPACK>(product.data(), identity.data(), n*n,
               check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Symmetric Eigenvalue Decomposition Tests
// ============================================================================

template <typename T>
int test_sym_eigenvalue_decomp() {

    Shape s(M, M);
    Matrix<T> A(s);
    Matrix<T> A_original(s);
    Vector<T> eigenvalues(M);

    make_positive_definite(A);

    CopyVectors(A, A_original);

    SymEigenvalueDecomposition(A, eigenvalues, 'V');
    // A now contains eigenvectors, eigenvalues contains eigenvalues

    // Verify: A_original * V = V * Lambda (where V is eigenvectors, Lambda is diagonal of eigenvalues)
    Matrix<T> product = multiply_matrices(A_original, A, false);

    // Create diagonal matrix Lambda and compute V * Lambda
    Matrix<T> lambda_diag(Shape(M, M), T(0.0));
    lambda_diag.set_diagonal(eigenvalues);
    Matrix<T> expected = multiply_matrices(A, lambda_diag, false);

    if (!check<T, ToleranceType::LAPACK>(product.data(), expected.data(), M*M,
               check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_sym_eigenvalue_no_vectors() {

    Shape s(4, 4);
    Matrix<T> A(s);
    Vector<T> eigenvalues(4);

    make_positive_definite(A);

    SymEigenvalueDecomposition(A, eigenvalues, 'N');

    // Check that eigenvalues are positive (for positive definite matrix)
    for (size_t i = 0; i < eigenvalues.size(); i++) {
        if (eigenvalues[i] <= 0) {
            std::cerr << check_msg(get_type_name<T>(), "Negative eigenvalue found") << "at index " << i 
            << ": " << std::setprecision(15) << eigenvalues[i] << std::endl;
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
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
    T B_original[n*nrhs];
    T A_copy[n*n];

    create_diagonal_dominant_matrix<T>(A, n);

    for (int i = 0; i < n*nrhs; i++) {
        B[i] = (T)1.0 + (i % 2) * (T)0.5;
        B_original[i] = B[i];
    }

    lahva::cpu::CopyVectors(n*n, A, A_copy);

    lahva::cpu::SolveGenSysLinEquations("N", n, A, nrhs, B);

    // Verify: A_copy * B = B_original
    T* product = c_multiply_matrices(A_copy, B, n, nrhs, false);
    if (!check<T, ToleranceType::LAPACK>(product, B_original, n*nrhs,
               check_msg(get_type_name<T>(), ""))) {
        delete[] product;
        return TEST_FAIL;
    }
    delete[] product;

    return TEST_PASS;
}

template <typename T>
int test_c_solve_gen_sys_default() {
    const int n = 4;
    const int nrhs = 2;

    T A[n*n];
    T B[n*nrhs];
    T B_original[n*nrhs];
    T A_copy[n*n];

    create_diagonal_dominant_matrix<T>(A, n);

    for (int i = 0; i < n*nrhs; i++) {
        B[i] = (T)1.0 + (i % 2) * (T)0.5;
        B_original[i] = B[i];
    }

    lahva::cpu::CopyVectors(n*n, A, A_copy);

    lahva::cpu::SolveGenSysLinEquations(n, A, nrhs, B, "N");

    // Verify: A_copy * B = B_original
    T* product = c_multiply_matrices(A_copy, B, n, nrhs, false);
    if (!check<T, ToleranceType::LAPACK>(product, B_original, n*nrhs,
               check_msg(get_type_name<T>(), "A*X verification"))) {
        delete[] product;
        return TEST_FAIL;
    }
    delete[] product;

    return TEST_PASS;
}

template <typename T>
int test_c_solve_gen_sys_transpose() {

    const int n = 4;
    const int nrhs = 2;

    T A[n*n];
    T B[n*nrhs];
    T B_original[n*nrhs];
    T A_copy[n*n];

    create_diagonal_dominant_matrix<T>(A, n);

    for (int i = 0; i < n*nrhs; i++) {
        B[i] = (T)1.0 + (i % 2) * (T)0.5;
        B_original[i] = B[i];
    }

    lahva::cpu::CopyVectors(n*n, A, A_copy);

    lahva::cpu::SolveGenSysLinEquations("T", n, A, nrhs, B);

    // Verify: A_copy^T * B = B_original
    T* product = c_multiply_matrices(A_copy, B, n, nrhs, true);
    if (!check<T, ToleranceType::LAPACK>(product, B_original, n*nrhs,
               check_msg(get_type_name<T>(), "A^T*X verification"))) {
        delete[] product;
        return TEST_FAIL;
    }
    delete[] product;

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
