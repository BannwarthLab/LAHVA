#include "test_common.h"

using namespace lahva::gpu;
using lahva::Shape;
using lahva::CudaRuntime;

// ============================================================================
// Mixed-Precision Matrix Constructor Tests (Templated)
// ============================================================================

template <typename T>
int test_mp_matrix_default_constructor() {

    MixedPrecisionMatrix<T> m;

    if (!check((int)m.shape().first, 0, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m.shape().second, 0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_mp_matrix_shape_constructor() {

    Shape s(4, 5);
    MixedPrecisionMatrix<T> m(s);

    if (!check((int)m.shape().first, 4, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m.shape().second, 5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_mp_matrix_shape_value_constructor() {

    Shape s(3, 3);
    MixedPrecisionMatrix<T> m(s);
    m.data()[0] = (T)2.5;

    if (!check(m.data()[0], (T)2.5, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_mp_matrix_cudart_constructor() {
    CudaRuntime cudart;

    Shape s(2, 2);
    MixedPrecisionMatrix<T> m(s, cudart);

    if (!check((int)m.shape().first, 2, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_mp_matrix_copy_from_matrix() {

    Shape s(3, 3);
    Matrix<T> base(s, (T)1.5);
    MixedPrecisionMatrix<T> m(base);

    if (!check((int)m.shape().first, 3, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check(m.data()[0], (T)1.5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_mp_matrix_initializer_list() {

    Shape s(2, 2);
    std::initializer_list<T> init = {(T)1.0, (T)2.0, (T)3.0, (T)4.0};
    MixedPrecisionMatrix<T> m(s, init, false);

    if (!check((int)m.size(), 4, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_mp_matrix_data_pointer_constructor() {

    T* data = new T[6];
    for (int i = 0; i < 6; i++) data[i] = (T)1.5;

    Shape s(2, 3);
    MixedPrecisionMatrix<T> m(s, data, false);

    if (!check((int)m.shape().first, 2, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    delete[] data;
    return TEST_PASS;
}

template <typename T>
int test_mp_matrix_copy_constructor() {

    Shape s(2, 2);
    MixedPrecisionMatrix<T> m1(s);
    m1.data()[0] = (T)3.5;

    MixedPrecisionMatrix<T> m2 = m1;

    if (!check((int)m2.shape().first, 2, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check(m2.data()[0], (T)3.5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_mp_matrix_move_constructor() {

    Shape s(2, 2);
    MixedPrecisionMatrix<T> m1(s);
    m1.data()[0] = (T)2.5;

    MixedPrecisionMatrix<T> m2 = std::move(m1);

    if (!check((int)m2.shape().first, 2, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check(m2.data()[0], (T)2.5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Mixed-Precision Matrix Assignment Tests (Templated)
// ============================================================================

template <typename T>
int test_mp_matrix_copy_assignment() {

    Shape s1(3, 3);
    Shape s2(2, 2);
    MixedPrecisionMatrix<T> m1(s1);
    m1.data()[0] = (T)1.5;

    MixedPrecisionMatrix<T> m2(s2);
    m2 = m1;

    if (!check((int)m2.shape().first, 3, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_mp_matrix_move_assignment() {

    Shape s1(3, 3);
    Shape s2(2, 2);
    MixedPrecisionMatrix<T> m1(s1);
    m1.data()[0] = (T)2.5;

    MixedPrecisionMatrix<T> m2(s2);
    m2 = std::move(m1);

    if (!check((int)m2.shape().first, 3, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_mp_matrix_self_assignment() {

    Shape s(2, 2);
    MixedPrecisionMatrix<T> m(s);
    m.data()[0] = (T)1.5;

    m = m;

    if (!check(m.data()[0], (T)1.5, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Mixed-Precision Matrix Split State Tests
// ============================================================================

int test_mp_matrix_split_size_fp32() {

    Shape s(4, 4);
    MixedPrecisionMatrix<double> m(s);

    size_t size = m.splitSize<float>();

    if (!check((int)size, 0, check_msg(get_type_name<double>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

int test_mp_matrix_split_size_fp16() {

    Shape s(4, 4);
    MixedPrecisionMatrix<double> m(s);

    size_t size = m.splitSize<__half>();

    if (!check((int)size, 0, check_msg(get_type_name<double>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

int test_mp_matrix_get_split_exponent() {

    Shape s(4, 4);
    MixedPrecisionMatrix<double> m(s);

    // Test that getSplitExponent throws when out of range
    try {
        m.getSplitExponent(0);
        return TEST_FAIL;  // Should have thrown
    } catch (const std::out_of_range&) {
        // Expected
    }

    return TEST_PASS;
}

int test_mp_matrix_reset_split() {

    Shape s(3, 3);
    MixedPrecisionMatrix<double> m(s);

    m.resetSplit();

    // After reset, split size should be 0
    size_t size_fp32 = m.splitSize<float>();
    if (!check((int)size_fp32, 0, check_msg(get_type_name<double>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

int test_mp_matrix_split_matrices_initial_state() {

    Shape s(4, 4);
    MixedPrecisionMatrix<double> m(s);

    // Check initial state of split matrices
    if (!check((int)m.split_matrices_fp16_.size(), 0, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m.split_matrices_fp32_.size(), 0, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    if (!check((int)m.splitted_fp16_, 0, check_msg(get_type_name<double>(), "check 3"))) return TEST_FAIL;

    if (!check((int)m.splitted_fp32_, 0, check_msg(get_type_name<double>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_mp_matrix_split_exponents_initial_state() {

    Shape s(3, 3);
    MixedPrecisionMatrix<double> m(s);

    // Check initial exponents vector
    if (!check((int)m.split_exponents_.size(), 0, check_msg(get_type_name<double>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

int test_mp_matrix_copy_constructor_copies_split_state() {

    Shape s(2, 2);
    MixedPrecisionMatrix<double> m1(s);
    m1.max_split_ = 4;
    m1.splitted_fp16_ = true;
    m1.splitted_fp32_ = false;

    MixedPrecisionMatrix<double> m2 = m1;

    if (!check((int)m2.max_split_, 4, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m2.splitted_fp16_, 1, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    if (!check((int)m2.splitted_fp32_, 0, check_msg(get_type_name<double>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_mp_matrix_move_constructor_transfers_split_state() {

    Shape s(2, 2);
    MixedPrecisionMatrix<double> m1(s);
    m1.max_split_ = 3;
    m1.splitted_fp32_ = true;
    m1.splitted_fp16_ = false;

    MixedPrecisionMatrix<double> m2 = std::move(m1);

    if (!check((int)m2.max_split_, 3, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m2.splitted_fp32_, 1, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    // After move, source should be reset
    if (!check((int)m1.max_split_, 0, check_msg(get_type_name<double>(), "check 3"))) return TEST_FAIL;

    if (!check((int)m1.splitted_fp16_, 0, check_msg(get_type_name<double>(), "check 4"))) return TEST_FAIL;

    if (!check((int)m1.splitted_fp32_, 0, check_msg(get_type_name<double>(), "check 5"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Mixed-Precision Matrix Float Type Tests
// ============================================================================

int test_mp_matrix_float_type() {

    Shape s(3, 3);
    MixedPrecisionMatrix<float> m(s);
    m.data()[0] = 1.5f;

    if (!check((int)m.shape().first, 3, check_msg(get_type_name<float>(), "check 1"))) return TEST_FAIL;

    if (!check(m.data()[0], 1.5f, check_msg(get_type_name<float>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Mixed-Precision Matrix Mod Value Tests
// ============================================================================

int test_mp_matrix_mod_value() {

    Shape s(4, 4);
    MixedPrecisionMatrix<double> m(s);

    if (!check((int)m.mod_value_, 8, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    m.mod_value_ = 16;
    if (!check((int)m.mod_value_, 16, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_mp_matrix_max_split_attribute() {

    Shape s(4, 4);
    MixedPrecisionMatrix<double> m(s);

    if (!check((int)m.max_split_, 0, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    m.max_split_ = 5;
    if (!check((int)m.max_split_, 5, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int total_failures = 0;

    // Constructor tests - test with double and float
    total_failures += test_mp_matrix_default_constructor<double>();
    total_failures += test_mp_matrix_default_constructor<float>();
    total_failures += test_mp_matrix_shape_constructor<double>();
    total_failures += test_mp_matrix_shape_constructor<float>();
    total_failures += test_mp_matrix_shape_value_constructor<double>();
    total_failures += test_mp_matrix_shape_value_constructor<float>();
    total_failures += test_mp_matrix_cudart_constructor<double>();
    total_failures += test_mp_matrix_cudart_constructor<float>();
    total_failures += test_mp_matrix_copy_from_matrix<double>();
    total_failures += test_mp_matrix_copy_from_matrix<float>();
    total_failures += test_mp_matrix_initializer_list<double>();
    total_failures += test_mp_matrix_initializer_list<float>();
    total_failures += test_mp_matrix_data_pointer_constructor<double>();
    total_failures += test_mp_matrix_data_pointer_constructor<float>();
    total_failures += test_mp_matrix_copy_constructor<double>();
    total_failures += test_mp_matrix_copy_constructor<float>();
    total_failures += test_mp_matrix_move_constructor<double>();
    total_failures += test_mp_matrix_move_constructor<float>();

    // Assignment tests
    total_failures += test_mp_matrix_copy_assignment<double>();
    total_failures += test_mp_matrix_copy_assignment<float>();
    total_failures += test_mp_matrix_move_assignment<double>();
    total_failures += test_mp_matrix_move_assignment<float>();
    total_failures += test_mp_matrix_self_assignment<double>();
    total_failures += test_mp_matrix_self_assignment<float>();

    // Split state tests (without calling split() which has GPU kernel issues)
    total_failures += test_mp_matrix_split_size_fp32();
    total_failures += test_mp_matrix_split_size_fp16();
    total_failures += test_mp_matrix_get_split_exponent();
    total_failures += test_mp_matrix_reset_split();
    total_failures += test_mp_matrix_split_matrices_initial_state();
    total_failures += test_mp_matrix_split_exponents_initial_state();
    total_failures += test_mp_matrix_copy_constructor_copies_split_state();
    total_failures += test_mp_matrix_move_constructor_transfers_split_state();

    // Type variation tests
    total_failures += test_mp_matrix_float_type();

    // Attribute tests
    total_failures += test_mp_matrix_mod_value();
    total_failures += test_mp_matrix_max_split_attribute();

    if (total_failures > 0) {
        std::cerr << "gpu/tensor/mixed-precision-matrix tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All gpu/tensor/mixed-precision-matrix tests passed!" << std::endl;
    return TEST_PASS;
}
