#include "test_common.h"
#include <cmath>

using namespace lahva::cpu;

// Use strict tolerances for these tests
const double thr2 = DEFAULT_STRICT_DOUBLE_TOLERANCE;

// Test vector operations
int test_vector_sum() {
    int stat_ = 0;

    Vector<double> v({1.0, 2.0, 3.0, 4.0, 5.0});

    double sum_val = sum(5, v.data());

    if (!check(sum_val, 15.0, thr2, "Error computing sum of vector.")) {
        stat_ += 1;
    }

    return stat_;
}

int test_vector_norm() {
    int stat_ = 0;

    Vector<double> v({3.0, 4.0, 0.0});

    double norm_val = norm2(3, v.data());

    if (!check(norm_val, 5.0, thr2, "Error computing norm of vector.")) {
        stat_ += 1;
    }

    return stat_;
}

// ============================================================================
// Helper function for matrix setup
// ============================================================================

template <typename T>
void setup_3x3_matrix(Matrix<T>& A) {
    A(0, 0) = (T)1.0;
    A(0, 1) = (T)2.0;
    A(0, 2) = (T)3.0;
    A(1, 0) = (T)4.0;
    A(1, 1) = (T)5.0;
    A(1, 2) = (T)6.0;
    A(2, 0) = (T)7.0;
    A(2, 1) = (T)8.0;
    A(2, 2) = (T)9.0;
}

// ============================================================================
// ComputeTrace Tests
// ============================================================================

template <typename T>
int test_compute_trace() {
    int stat_ = 0;

    Shape s(3, 3);
    Matrix<T> A(s);
    setup_3x3_matrix(A);

    CPURuntime rt;
    double trace = ComputeTrace(rt, A);
    double expected = 1.0 + 5.0 + 9.0; // 15.0

    if (!check(trace, expected, thr2, "Error computing trace.")) {
        stat_ += 1;
    }

    return stat_;
}

template <typename T>
int test_compute_trace_vector() {
    int stat_ = 0;

    Vector<T> diag({(T)1.0, (T)2.0, (T)3.0, (T)4.0, (T)5.0});

    CPURuntime rt;
    double trace = ComputeTrace(rt, diag);
    double expected = 15.0;

    if (!check(trace, expected, thr2, "Error computing trace from vector.")) {
        stat_ += 1;
    }

    return stat_;
}

// ============================================================================
// FrobeniusNorm Tests
// ============================================================================

template <typename T>
int test_frobenius_norm() {
    int stat_ = 0;

    Shape s(2, 2);
    Matrix<T> A(s);
    A(0, 0) = (T)1.0;
    A(0, 1) = (T)2.0;
    A(1, 0) = (T)2.0;
    A(1, 1) = (T)1.0;

    T norm = FrobeniusNorm(A);
    double expected = std::sqrt(1.0 + 4.0 + 4.0 + 1.0); // sqrt(10)

    if (!check((double)norm, expected, get_tolerance<T>(), "Error computing Frobenius norm.")) {
        stat_ += 1;
    }

    return stat_;
}

template <typename T>
int test_frobenius_norm_difference() {
    int stat_ = 0;

    Shape s(2, 2);
    Matrix<T> A(s);
    A(0, 0) = (T)1.0;
    A(0, 1) = (T)2.0;
    A(1, 0) = (T)2.0;
    A(1, 1) = (T)1.0;

    Matrix<T> B(s);
    B(0, 0) = (T)1.0;
    B(0, 1) = (T)1.0;
    B(1, 0) = (T)1.0;
    B(1, 1) = (T)1.0;

    T norm = FrobeniusNorm(A, B);
    double expected = std::sqrt(0.0 + 1.0 + 1.0 + 0.0); // sqrt(2)

    if (!check((double)norm, expected, get_tolerance<T>(), "Error computing Frobenius norm difference.")) {
        stat_ += 1;
    }

    return stat_;
}

template <typename T>
int test_frobenius_norm_with_runtime() {
    int stat_ = 0;

    Shape s(2, 2);
    Matrix<T> A(s);
    A(0, 0) = (T)1.0;
    A(0, 1) = (T)2.0;
    A(1, 0) = (T)2.0;
    A(1, 1) = (T)1.0;

    CPURuntime rt;
    T norm = FrobeniusNorm(rt, A);
    double expected = std::sqrt(1.0 + 4.0 + 4.0 + 1.0);

    if (!check((double)norm, expected, get_tolerance<T>(), "Error computing Frobenius norm with runtime.")) {
        stat_ += 1;
    }

    return stat_;
}

// ============================================================================
// FrobeniusInnerProduct Tests
// ============================================================================

template <typename T>
int test_frobenius_inner_product() {
    int stat_ = 0;

    Shape s(2, 2);
    Matrix<T> A(s);
    A(0, 0) = (T)1.0;
    A(0, 1) = (T)2.0;
    A(1, 0) = (T)3.0;
    A(1, 1) = (T)4.0;

    Matrix<T> B(s);
    B(0, 0) = (T)2.0;
    B(0, 1) = (T)1.0;
    B(1, 0) = (T)1.0;
    B(1, 1) = (T)2.0;

    T prod = FrobeniusInnerProduct(A, B);
    double expected = 1.0 * 2.0 + 2.0 * 1.0 + 3.0 * 1.0 + 4.0 * 2.0; // 15

    if (!check((double)prod, expected, get_tolerance<T>(), "Error computing Frobenius inner product.")) {
        stat_ += 1;
    }

    return stat_;
}

template <typename T>
int test_frobenius_inner_product_with_runtime() {
    int stat_ = 0;

    Shape s(2, 2);
    Matrix<T> A(s);
    A(0, 0) = (T)1.0;
    A(0, 1) = (T)2.0;
    A(1, 0) = (T)3.0;
    A(1, 1) = (T)4.0;

    Matrix<T> B(s);
    B(0, 0) = (T)2.0;
    B(0, 1) = (T)1.0;
    B(1, 0) = (T)1.0;
    B(1, 1) = (T)2.0;

    CPURuntime rt;
    T prod = FrobeniusInnerProduct(rt, A, B);
    double expected = 15.0;

    if (!check((double)prod, expected, get_tolerance<T>(), "Error computing Frobenius inner product with runtime.")) {
        stat_ += 1;
    }

    return stat_;
}

// ============================================================================
// HadamardProduct Tests
// ============================================================================

template <typename T>
int test_hadamard_product_in_place() {
    int stat_ = 0;

    Shape s(2, 2);
    Matrix<T> A(s);
    A(0, 0) = (T)1.0;
    A(0, 1) = (T)2.0;
    A(1, 0) = (T)3.0;
    A(1, 1) = (T)4.0;

    Matrix<T> B(s);
    B(0, 0) = (T)2.0;
    B(0, 1) = (T)2.0;
    B(1, 0) = (T)2.0;
    B(1, 1) = (T)2.0;

    HadamardProduct(A, B);

    if (!check((double)B(0, 0), 2.0, thr2, "Hadamard product (in-place) failed.")) {
        stat_ += 1;
    }
    if (!check((double)B(0, 1), 4.0, thr2, "Hadamard product (in-place) failed.")) {
        stat_ += 1;
    }
    if (!check((double)B(1, 0), 6.0, thr2, "Hadamard product (in-place) failed.")) {
        stat_ += 1;
    }
    if (!check((double)B(1, 1), 8.0, thr2, "Hadamard product (in-place) failed.")) {
        stat_ += 1;
    }

    return stat_;
}

template <typename T>
int test_hadamard_product_output() {
    int stat_ = 0;

    Shape s(2, 2);
    Matrix<T> A(s);
    A(0, 0) = (T)1.0;
    A(0, 1) = (T)2.0;
    A(1, 0) = (T)3.0;
    A(1, 1) = (T)4.0;

    Matrix<T> B(s);
    B(0, 0) = (T)2.0;
    B(0, 1) = (T)2.0;
    B(1, 0) = (T)2.0;
    B(1, 1) = (T)2.0;

    Matrix<T> C(s);
    HadamardProduct(A, B, C);

    if (!check((double)C(0, 0), 2.0, thr2, "Hadamard product (output) failed.")) {
        stat_ += 1;
    }

    return stat_;
}

template <typename T>
int test_hadamard_product_pointer_no_increment() {
    int stat_ = 0;

    const int n = 4;
    T A[] = {(T)1.0, (T)2.0, (T)3.0, (T)4.0};
    T B[] = {(T)2.0, (T)2.0, (T)2.0, (T)2.0};
    T C[] = {(T)10.0, (T)20.0, (T)30.0, (T)40.0};

    HadamardProduct(n, A, B, C, false);

    if (!check((double)C[0], 2.0, thr2, "Hadamard product (pointer, no increment) failed.")) {
        stat_ += 1;
    }
    if (!check((double)C[1], 4.0, thr2, "Hadamard product (pointer, no increment) failed.")) {
        stat_ += 1;
    }

    return stat_;
}

template <typename T>
int test_hadamard_product_pointer_with_increment() {
    int stat_ = 0;

    const int n = 4;
    T A[] = {(T)1.0, (T)2.0, (T)3.0, (T)4.0};
    T B[] = {(T)2.0, (T)2.0, (T)2.0, (T)2.0};
    T C[] = {(T)10.0, (T)20.0, (T)30.0, (T)40.0};

    HadamardProduct(n, A, B, C, true);

    if (!check((double)C[0], 12.0, thr2, "Hadamard product (pointer, with increment) failed.")) {
        stat_ += 1;
    }
    if (!check((double)C[1], 24.0, thr2, "Hadamard product (pointer, with increment) failed.")) {
        stat_ += 1;
    }

    return stat_;
}

int main() {
    int total_failures = 0;

    // Basic vector operations
    total_failures += test_vector_sum();
    total_failures += test_vector_norm();

    // ComputeTrace tests
    total_failures += test_compute_trace<double>();
    total_failures += test_compute_trace<float>();
    total_failures += test_compute_trace_vector<double>();
    total_failures += test_compute_trace_vector<float>();

    // FrobeniusNorm tests
    total_failures += test_frobenius_norm<double>();
    total_failures += test_frobenius_norm<float>();
    total_failures += test_frobenius_norm_difference<double>();
    total_failures += test_frobenius_norm_difference<float>();
    total_failures += test_frobenius_norm_with_runtime<double>();
    total_failures += test_frobenius_norm_with_runtime<float>();

    // FrobeniusInnerProduct tests
    total_failures += test_frobenius_inner_product<double>();
    total_failures += test_frobenius_inner_product<float>();
    total_failures += test_frobenius_inner_product_with_runtime<double>();
    total_failures += test_frobenius_inner_product_with_runtime<float>();

    // HadamardProduct tests
    total_failures += test_hadamard_product_in_place<double>();
    total_failures += test_hadamard_product_in_place<float>();
    total_failures += test_hadamard_product_output<double>();
    total_failures += test_hadamard_product_output<float>();
    total_failures += test_hadamard_product_pointer_no_increment<double>();
    total_failures += test_hadamard_product_pointer_no_increment<float>();
    total_failures += test_hadamard_product_pointer_with_increment<double>();
    total_failures += test_hadamard_product_pointer_with_increment<float>();

    if (total_failures == 0) {
        std::cout << "All CPU additional Level-1 tests passed!" << std::endl;
    } else {
        std::cout << "CPU additional Level-1 tests: " << total_failures << " failures" << std::endl;
    }

    return total_failures;
}
