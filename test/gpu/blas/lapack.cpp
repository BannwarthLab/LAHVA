#include "test_common.h"
#include "array_utils.hpp"

#define M 4
#define N 2

using namespace lahva::gpu;
using lahva::Shape;
using lahva::CudaRuntime;

// ============================================================================
// Matrix helpers
// ============================================================================

// Host-side reference product C = A * B for square matrices
template <typename T>
Matrix<T> multiply_matrices(const Matrix<T>& A, const Matrix<T>& B) {
    size_t n = A.shape().first;
    Matrix<T> result(Shape(n, n), (T)0.0);

    for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < n; i++) {
            T sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                sum += A(i, k) * B(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

// Host-side reference product C = op(A) * X for square A (n x n) and X (n x nrhs)
template <typename T>
Matrix<T> multiply_matrices(const Matrix<T>& A, const Matrix<T>& X, bool transpose) {
    size_t n = A.shape().first;
    size_t nrhs = X.shape().second;
    Matrix<T> result(Shape(n, nrhs), (T)0.0);

    for (size_t j = 0; j < nrhs; j++) {
        for (size_t i = 0; i < n; i++) {
            T sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                sum += (transpose ? A(k, i) : A(i, k)) * X(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

// Fill a right-hand side matrix with deterministic values
template <typename T>
void fill_rhs(Matrix<T>& B) {
    for (size_t j = 0; j < B.shape().second; j++) {
        for (size_t i = 0; i < B.shape().first; i++) {
            B(i, j) = (T)1.0 + (T)0.5 * (T)((i + 2*j) % 7);
        }
    }
}

// Create a well-conditioned matrix whose LU factorization requires row pivoting
template <typename T>
void create_pivoting_matrix(Matrix<T>& A) {
    size_t n = A.shape().first;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            A(i, j) = (T)0.1 + ((i*n + j) % 5) * (T)0.05;
        }
    }
    for (size_t i = 0; i < n; i++) {
        T sum = 0.0;
        for (size_t j = 0; j < n; j++) {
            if (i != j) sum += std::abs(A(i, j));
        }
        A(i, i) = sum + (T)(100.0 * n);
    }
    for (size_t j = 0; j < n; j++) {
        std::swap(A(0, j), A(n - 1, j));
    }
}

template <typename T>
Matrix<T> identity_matrix(size_t n) {
    Matrix<T> identity(Shape(n, n), (T)0.0);
    for (size_t i = 0; i < n; i++) {
        identity(i, i) = (T)1.0;
    }
    return identity;
}

// ============================================================================
// General Matrix Inversion Tests
// ============================================================================

template <typename T>
int test_invert_matrix(CudaRuntime& cudart, size_t n) {

    Shape s(n, n);
    Matrix<T> A(s, (T)0.0);
    Matrix<T> A_original(s, (T)0.0);

    create_pivoting_matrix(A);
    create_pivoting_matrix(A_original);

    InvertMatrix(cudart, A);

    cudart.synchronize();
    A.copy2host(cudart);

    Matrix<T> identity = identity_matrix<T>(n);

    // Verify A_original * A^(-1) = I
    Matrix<T> product = multiply_matrices(A_original, A);
    if (!check<T, ToleranceType::LAPACK>(product.data(), identity.data(), n*n,
               check_msg(get_type_name<T>(), "A*inv(A)"))) return TEST_FAIL;

    // Verify A^(-1) * A_original = I
    product = multiply_matrices(A, A_original);
    if (!check<T, ToleranceType::LAPACK>(product.data(), identity.data(), n*n,
               check_msg(get_type_name<T>(), "inv(A)*A"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_invert_matrix_matches_cpu(CudaRuntime& cudart) {

    Shape s(M, M);
    Matrix<T> A_gpu(s, (T)0.0);
    lahva::cpu::Matrix<T> A_cpu(s, (T)0.0);

    create_pivoting_matrix(A_gpu);
    for (size_t j = 0; j < M; j++) {
        for (size_t i = 0; i < M; i++) {
            A_cpu(i, j) = A_gpu(i, j);
        }
    }

    InvertMatrix(cudart, A_gpu);
    lahva::cpu::InvertMatrix(A_cpu);

    cudart.synchronize();
    A_gpu.copy2host(cudart);

    if (!check<T, ToleranceType::LAPACK>(A_gpu.data(), A_cpu.data(), M*M,
               check_msg(get_type_name<T>(), "GPU vs CPU"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_invert_singular_matrix_throws(CudaRuntime& cudart) {

    // Rank-1 matrix: every column is identical
    Matrix<T> A(Shape(M, M), (T)1.0);

    try {
        InvertMatrix(cudart, A);
    } catch (const std::runtime_error&) {
        return TEST_PASS;
    }
    std::cerr << check_msg(get_type_name<T>(), "Singular matrix did not throw") << std::endl;
    return TEST_FAIL;
}

template <typename T>
int test_invert_non_square_matrix_throws(CudaRuntime& cudart) {

    Matrix<T> A(Shape(M, 2), (T)1.0);

    try {
        InvertMatrix(cudart, A);
    } catch (const std::runtime_error&) {
        return TEST_PASS;
    }
    std::cerr << check_msg(get_type_name<T>(), "Non-square matrix did not throw") << std::endl;
    return TEST_FAIL;
}

// ============================================================================
// Solve General Linear System Tests
// ============================================================================

template <typename T>
int test_solve_gen_sys(CudaRuntime& cudart, const char* Ta, size_t n) {

    Matrix<T> A(Shape(n, n), (T)0.0);
    Matrix<T> A_original(Shape(n, n), (T)0.0);
    Matrix<T> B(Shape(n, N), (T)0.0);
    Matrix<T> B_original(Shape(n, N), (T)0.0);

    create_pivoting_matrix(A);
    create_pivoting_matrix(A_original);
    fill_rhs(B);
    fill_rhs(B_original);

    SolveGenSysLinEquations(cudart, Ta, A, B);

    cudart.synchronize();
    B.copy2host(cudart);

    // Verify op(A_original) * X = B_original
    Matrix<T> product = multiply_matrices(A_original, B, Ta[0] == 'T');
    if (!check<T, ToleranceType::LAPACK>(product.data(), B_original.data(), n*N,
               check_msg(get_type_name<T>(), std::string("op(A)*X, Ta=") + Ta))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_solve_gen_sys_alt_order(CudaRuntime& cudart) {

    Matrix<T> A(Shape(M, M), (T)0.0);
    Matrix<T> A_original(Shape(M, M), (T)0.0);
    Matrix<T> B(Shape(M, N), (T)0.0);
    Matrix<T> B_original(Shape(M, N), (T)0.0);

    create_pivoting_matrix(A);
    create_pivoting_matrix(A_original);
    fill_rhs(B);
    fill_rhs(B_original);

    // Default Ta = "N"
    SolveGenSysLinEquations(cudart, A, B);

    cudart.synchronize();
    B.copy2host(cudart);

    // Verify A_original * X = B_original
    Matrix<T> product = multiply_matrices(A_original, B, false);
    if (!check<T, ToleranceType::LAPACK>(product.data(), B_original.data(), M*N,
               check_msg(get_type_name<T>(), "A*X"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_solve_gen_sys_matches_cpu(CudaRuntime& cudart) {

    Matrix<T> A_gpu(Shape(M, M), (T)0.0);
    Matrix<T> B_gpu(Shape(M, N), (T)0.0);
    lahva::cpu::Matrix<T> A_cpu(Shape(M, M), (T)0.0);
    lahva::cpu::Matrix<T> B_cpu(Shape(M, N), (T)0.0);

    create_pivoting_matrix(A_gpu);
    fill_rhs(B_gpu);
    for (size_t j = 0; j < M; j++) {
        for (size_t i = 0; i < M; i++) {
            A_cpu(i, j) = A_gpu(i, j);
        }
    }
    for (size_t j = 0; j < N; j++) {
        for (size_t i = 0; i < M; i++) {
            B_cpu(i, j) = B_gpu(i, j);
        }
    }

    SolveGenSysLinEquations(cudart, "N", A_gpu, B_gpu);
    lahva::cpu::SolveGenSysLinEquations("N", A_cpu, B_cpu);

    cudart.synchronize();
    B_gpu.copy2host(cudart);

    if (!check<T, ToleranceType::LAPACK>(B_gpu.data(), B_cpu.data(), M*N,
               check_msg(get_type_name<T>(), "GPU vs CPU"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_solve_gen_sys_singular_throws(CudaRuntime& cudart) {

    // Rank-1 matrix: every column is identical
    Matrix<T> A(Shape(M, M), (T)1.0);
    Matrix<T> B(Shape(M, N), (T)1.0);

    try {
        SolveGenSysLinEquations(cudart, "N", A, B);
    } catch (const std::runtime_error&) {
        return TEST_PASS;
    }
    std::cerr << check_msg(get_type_name<T>(), "Singular matrix did not throw") << std::endl;
    return TEST_FAIL;
}

template <typename T>
int test_solve_gen_sys_wrong_rhs_throws(CudaRuntime& cudart) {

    Matrix<T> A(Shape(M, M), (T)1.0);
    Matrix<T> B(Shape(M + 1, N), (T)1.0);

    try {
        SolveGenSysLinEquations(cudart, "N", A, B);
    } catch (const std::runtime_error&) {
        return TEST_PASS;
    }
    std::cerr << check_msg(get_type_name<T>(), "Mismatched B did not throw") << std::endl;
    return TEST_FAIL;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    int total_failures = 0;
    CudaRuntime cudart;

    // General matrix inversion tests
    total_failures += test_invert_matrix<double>(cudart, M);
    total_failures += test_invert_matrix<float>(cudart, M);
    total_failures += test_invert_matrix<double>(cudart, 64);
    total_failures += test_invert_matrix<float>(cudart, 64);
    total_failures += test_invert_matrix_matches_cpu<double>(cudart);
    total_failures += test_invert_matrix_matches_cpu<float>(cudart);
    total_failures += test_invert_singular_matrix_throws<double>(cudart);
    total_failures += test_invert_singular_matrix_throws<float>(cudart);
    total_failures += test_invert_non_square_matrix_throws<double>(cudart);
    total_failures += test_invert_non_square_matrix_throws<float>(cudart);

    // Solve general linear system tests
    total_failures += test_solve_gen_sys<double>(cudart, "N", M);
    total_failures += test_solve_gen_sys<float>(cudart, "N", M);
    total_failures += test_solve_gen_sys<double>(cudart, "T", M);
    total_failures += test_solve_gen_sys<float>(cudart, "T", M);
    total_failures += test_solve_gen_sys<double>(cudart, "N", 64);
    total_failures += test_solve_gen_sys<float>(cudart, "N", 64);
    total_failures += test_solve_gen_sys_alt_order<double>(cudart);
    total_failures += test_solve_gen_sys_alt_order<float>(cudart);
    total_failures += test_solve_gen_sys_matches_cpu<double>(cudart);
    total_failures += test_solve_gen_sys_matches_cpu<float>(cudart);
    total_failures += test_solve_gen_sys_singular_throws<double>(cudart);
    total_failures += test_solve_gen_sys_singular_throws<float>(cudart);
    total_failures += test_solve_gen_sys_wrong_rhs_throws<double>(cudart);
    total_failures += test_solve_gen_sys_wrong_rhs_throws<float>(cudart);

    if (total_failures > 0) {
        std::cerr << "gpu/blas/lapack tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All gpu/blas/lapack tests passed!" << std::endl;
    return TEST_PASS;
}
