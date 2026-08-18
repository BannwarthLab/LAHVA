#include "test_common.h"

using namespace lahva::cpu;
using lahva::Shape;
using lahva::CPURuntime;

// ============================================================================
// ComputeTrace Tests
// ============================================================================

template <typename T>
int test_compute_trace() {

    Shape s(3, 3);
    Matrix<T> A(s);
    A(0, 0) = (T)1.0;
    A(0, 1) = (T)2.0;
    A(0, 2) = (T)3.0;
    A(1, 0) = (T)4.0;
    A(1, 1) = (T)5.0;
    A(1, 2) = (T)6.0;
    A(2, 0) = (T)7.0;
    A(2, 1) = (T)8.0;
    A(2, 2) = (T)9.0;

    CPURuntime rt;
    T trace = ComputeTrace(rt, A);
    T expected = 1.0 + 5.0 + 9.0; // 15.0

    if (!check(trace, expected, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_compute_trace_vector() {

    Vector<T> diag({(T)1.0, (T)2.0, (T)3.0, (T)4.0, (T)5.0});

    CPURuntime rt;
    T trace = ComputeTrace(rt, diag);
    T expected = 15.0;

    if (!check(trace, expected, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// FrobeniusNorm Tests
// ============================================================================

template <typename T>
int test_frobenius_norm() {

    Shape s(2, 2);
    Matrix<T> A(s);
    A(0, 0) = (T)1.0;
    A(0, 1) = (T)2.0;
    A(1, 0) = (T)2.0;
    A(1, 1) = (T)1.0;

    T norm = FrobeniusNorm(A);
    T expected = std::sqrt(1.0 + 4.0 + 4.0 + 1.0); // sqrt(10)

    if (!check(norm, expected, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_frobenius_norm_difference() {

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
    T expected = std::sqrt(0.0 + 1.0 + 1.0 + 0.0); // sqrt(2)

    if (!check(norm, expected, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_frobenius_norm_with_runtime() {

    Shape s(2, 2);
    Matrix<T> A(s);
    A(0, 0) = (T)1.0;
    A(0, 1) = (T)2.0;
    A(1, 0) = (T)2.0;
    A(1, 1) = (T)1.0;

    CPURuntime rt;
    T norm = FrobeniusNorm(rt, A);
    T expected = std::sqrt(1.0 + 4.0 + 4.0 + 1.0);

    if (!check(norm, expected, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// FrobeniusInnerProduct Tests
// ============================================================================

template <typename T>
int test_frobenius_inner_product() {

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
    T expected = 1.0 * 2.0 + 2.0 * 1.0 + 3.0 * 1.0 + 4.0 * 2.0; // 15

    if (!check(prod, expected, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_frobenius_inner_product_with_runtime() {

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
    T expected = 15.0;

    if (!check(prod, expected, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// HadamardProduct Tests
// ============================================================================

template <typename T>
int test_hadamard_product_in_place() {

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

    if (!check(B(0, 0), (T)2.0, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(B(0, 1), (T)4.0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(B(1, 0), (T)6.0, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check(B(1, 1), (T)8.0, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_hadamard_product_output() {

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

    if (!check(C(0, 0), (T)2.0, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_hadamard_product_pointer_no_increment() {

    const int n = 4;
    T A[] = {(T)1.0, (T)2.0, (T)3.0, (T)4.0};
    T B[] = {(T)2.0, (T)2.0, (T)2.0, (T)2.0};
    T C[] = {(T)10.0, (T)20.0, (T)30.0, (T)40.0};

    HadamardProduct(n, A, B, C, false);

    if (!check(C[0], (T)2.0, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(C[1], (T)4.0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_hadamard_product_pointer_with_increment() {

    const int n = 4;
    T A[] = {(T)1.0, (T)2.0, (T)3.0, (T)4.0};
    T B[] = {(T)2.0, (T)2.0, (T)2.0, (T)2.0};
    T C[] = {(T)10.0, (T)20.0, (T)30.0, (T)40.0};

    HadamardProduct(n, A, B, C, true);

    if (!check(C[0], (T)12.0, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(C[1], (T)24.0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int total_failures = 0;

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

    if (total_failures > 0) {
        std::cerr << "cpu/blas/additional-level1 tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All cpu/blas/additional-level1 tests passed!" << std::endl;
    return TEST_PASS;
}
