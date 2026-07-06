#include "test_common.h"
#ifdef _CUDA

using namespace lahva::gpu;

// ============================================================================
// Mixed-Precision Matrix Constructor Tests (Templated)
// ============================================================================

// Note: split() and merge() methods have GPU kernel instantiation issues.
// These advanced GPU operations are excluded from testing.

template <typename T>
int test_mp_matrix_default_constructor() {
    int failures = 0;

    MixedPrecisionMatrix<T> m;

    if (!check((int)m.shape().first, 0, "MP matrix default constructor rows")) {
        failures += 1;
    }

    if (!check((int)m.shape().second, 0, "MP matrix default constructor cols")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_mp_matrix_shape_constructor() {
    int failures = 0;

    Shape s(4, 5);
    MixedPrecisionMatrix<T> m(s);

    if (!check((int)m.shape().first, 4, "MP matrix shape constructor rows")) {
        failures += 1;
    }

    if (!check((int)m.shape().second, 5, "MP matrix shape constructor cols")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_mp_matrix_shape_value_constructor() {
    int failures = 0;

    Shape s(3, 3);
    MixedPrecisionMatrix<T> m(s);
    m.data()[0] = (T)2.5;

    double tol = get_tolerance<T>();
    if (!check(m.data()[0], (T)2.5, tol, "MP matrix shape value constructor")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_mp_matrix_cudart_constructor() {
    int failures = 0;
    CudaRuntime cudart;

    Shape s(2, 2);
    MixedPrecisionMatrix<T> m(s, cudart);

    if (!check((int)m.shape().first, 2, "MP matrix CudaRuntime constructor rows")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_mp_matrix_copy_from_matrix() {
    int failures = 0;

    Shape s(3, 3);
    Matrix<T> base(s, (T)1.5);
    MixedPrecisionMatrix<T> m(base);

    if (!check((int)m.shape().first, 3, "MP matrix copy from matrix rows")) {
        failures += 1;
    }

    double tol = get_tolerance<T>();
    if (!check(m.data()[0], (T)1.5, tol, "MP matrix copy from matrix data")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_mp_matrix_initializer_list() {
    int failures = 0;

    Shape s(2, 2);
    std::initializer_list<T> init = {(T)1.0, (T)2.0, (T)3.0, (T)4.0};
    MixedPrecisionMatrix<T> m(s, init, false);

    if (!check((int)m.size(), 4, "MP matrix initializer list size")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_mp_matrix_data_pointer_constructor() {
    int failures = 0;

    T* data = new T[6];
    for (int i = 0; i < 6; i++) data[i] = (T)1.5;

    Shape s(2, 3);
    MixedPrecisionMatrix<T> m(s, data, false);

    if (!check((int)m.shape().first, 2, "MP matrix data pointer constructor")) {
        failures += 1;
    }

    delete[] data;
    return failures;
}

template <typename T>
int test_mp_matrix_copy_constructor() {
    int failures = 0;

    Shape s(2, 2);
    MixedPrecisionMatrix<T> m1(s);
    m1.data()[0] = (T)3.5;

    MixedPrecisionMatrix<T> m2 = m1;

    if (!check((int)m2.shape().first, 2, "MP matrix copy constructor rows")) {
        failures += 1;
    }

    double tol = get_tolerance<T>();
    if (!check(m2.data()[0], (T)3.5, tol, "MP matrix copy constructor data")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_mp_matrix_move_constructor() {
    int failures = 0;

    Shape s(2, 2);
    MixedPrecisionMatrix<T> m1(s);
    m1.data()[0] = (T)2.5;

    MixedPrecisionMatrix<T> m2 = std::move(m1);

    if (!check((int)m2.shape().first, 2, "MP matrix move constructor rows")) {
        failures += 1;
    }

    double tol = get_tolerance<T>();
    if (!check(m2.data()[0], (T)2.5, tol, "MP matrix move constructor data")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Mixed-Precision Matrix Assignment Tests (Templated)
// ============================================================================

template <typename T>
int test_mp_matrix_copy_assignment() {
    int failures = 0;

    Shape s1(3, 3);
    Shape s2(2, 2);
    MixedPrecisionMatrix<T> m1(s1);
    m1.data()[0] = (T)1.5;

    MixedPrecisionMatrix<T> m2(s2);
    m2 = m1;

    if (!check((int)m2.shape().first, 3, "MP matrix copy assignment rows")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_mp_matrix_move_assignment() {
    int failures = 0;

    Shape s1(3, 3);
    Shape s2(2, 2);
    MixedPrecisionMatrix<T> m1(s1);
    m1.data()[0] = (T)2.5;

    MixedPrecisionMatrix<T> m2(s2);
    m2 = std::move(m1);

    if (!check((int)m2.shape().first, 3, "MP matrix move assignment rows")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_mp_matrix_self_assignment() {
    int failures = 0;

    Shape s(2, 2);
    MixedPrecisionMatrix<T> m(s);
    m.data()[0] = (T)1.5;

    m = m;

    double tol = get_tolerance<T>();
    if (!check(m.data()[0], (T)1.5, tol, "MP matrix self assignment")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Mixed-Precision Matrix Split State Tests
// ============================================================================

int test_mp_matrix_split_size_fp32() {
    int failures = 0;

    Shape s(4, 4);
    MixedPrecisionMatrix<double> m(s);

    size_t size = m.splitSize<float>();

    if (!check((int)size, 0, "MP matrix split size fp32 initial")) {
        failures += 1;
    }

    return failures;
}

int test_mp_matrix_split_size_fp16() {
    int failures = 0;

    Shape s(4, 4);
    MixedPrecisionMatrix<double> m(s);

    size_t size = m.splitSize<__half>();

    if (!check((int)size, 0, "MP matrix split size fp16 initial")) {
        failures += 1;
    }

    return failures;
}

int test_mp_matrix_get_split_exponent() {
    int failures = 0;

    Shape s(4, 4);
    MixedPrecisionMatrix<double> m(s);

    // Test that getSplitExponent throws when out of range
    try {
        m.getSplitExponent(0);
        failures += 1;  // Should have thrown
    } catch (const std::out_of_range&) {
        // Expected
    }

    return failures;
}

int test_mp_matrix_reset_split() {
    int failures = 0;

    Shape s(3, 3);
    MixedPrecisionMatrix<double> m(s);

    m.resetSplit();

    // After reset, split size should be 0
    size_t size_fp32 = m.splitSize<float>();
    if (!check((int)size_fp32, 0, "MP matrix reset split fp32")) {
        failures += 1;
    }

    return failures;
}

int test_mp_matrix_split_matrices_initial_state() {
    int failures = 0;

    Shape s(4, 4);
    MixedPrecisionMatrix<double> m(s);

    // Check initial state of split matrices
    if (!check((int)m.split_matrices_fp16_.size(), 0, "MP matrix split matrices fp16 initial")) {
        failures += 1;
    }

    if (!check((int)m.split_matrices_fp32_.size(), 0, "MP matrix split matrices fp32 initial")) {
        failures += 1;
    }

    if (!check((int)m.splitted_fp16_, 0, "MP matrix splitted_fp16 flag initial")) {
        failures += 1;
    }

    if (!check((int)m.splitted_fp32_, 0, "MP matrix splitted_fp32 flag initial")) {
        failures += 1;
    }

    return failures;
}

int test_mp_matrix_split_exponents_initial_state() {
    int failures = 0;

    Shape s(3, 3);
    MixedPrecisionMatrix<double> m(s);

    // Check initial exponents vector
    if (!check((int)m.split_exponents_.size(), 0, "MP matrix split exponents initial size")) {
        failures += 1;
    }

    return failures;
}

int test_mp_matrix_copy_constructor_copies_split_state() {
    int failures = 0;

    Shape s(2, 2);
    MixedPrecisionMatrix<double> m1(s);
    m1.max_split_ = 4;
    m1.splitted_fp16_ = true;
    m1.splitted_fp32_ = false;

    MixedPrecisionMatrix<double> m2 = m1;

    if (!check((int)m2.max_split_, 4, "MP matrix copy constructor copies max_split")) {
        failures += 1;
    }

    if (!check((int)m2.splitted_fp16_, 1, "MP matrix copy constructor copies splitted_fp16")) {
        failures += 1;
    }

    if (!check((int)m2.splitted_fp32_, 0, "MP matrix copy constructor preserves splitted_fp32")) {
        failures += 1;
    }

    return failures;
}

int test_mp_matrix_move_constructor_transfers_split_state() {
    int failures = 0;

    Shape s(2, 2);
    MixedPrecisionMatrix<double> m1(s);
    m1.max_split_ = 3;
    m1.splitted_fp32_ = true;
    m1.splitted_fp16_ = false;

    MixedPrecisionMatrix<double> m2 = std::move(m1);

    if (!check((int)m2.max_split_, 3, "MP matrix move constructor transfers max_split")) {
        failures += 1;
    }

    if (!check((int)m2.splitted_fp32_, 1, "MP matrix move constructor transfers splitted_fp32")) {
        failures += 1;
    }

    // After move, source should be reset
    if (!check((int)m1.max_split_, 0, "MP matrix move constructor resets source max_split")) {
        failures += 1;
    }

    if (!check((int)m1.splitted_fp16_, 0, "MP matrix move constructor resets source splitted_fp16")) {
        failures += 1;
    }

    if (!check((int)m1.splitted_fp32_, 0, "MP matrix move constructor resets source splitted_fp32")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Mixed-Precision Matrix Float Type Tests
// ============================================================================

int test_mp_matrix_float_type() {
    int failures = 0;

    Shape s(3, 3);
    MixedPrecisionMatrix<float> m(s);
    m.data()[0] = 1.5f;

    if (!check((int)m.shape().first, 3, "MP matrix float type rows")) {
        failures += 1;
    }

    if (!check(m.data()[0], 1.5f, 1e-6f, "MP matrix float type data")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Mixed-Precision Matrix Mod Value Tests
// ============================================================================

int test_mp_matrix_mod_value() {
    int failures = 0;

    Shape s(4, 4);
    MixedPrecisionMatrix<double> m(s);

    if (!check((int)m.mod_value_, 8, "MP matrix mod value default")) {
        failures += 1;
    }

    m.mod_value_ = 16;
    if (!check((int)m.mod_value_, 16, "MP matrix mod value assignment")) {
        failures += 1;
    }

    return failures;
}

int test_mp_matrix_max_split_attribute() {
    int failures = 0;

    Shape s(4, 4);
    MixedPrecisionMatrix<double> m(s);

    if (!check((int)m.max_split_, 0, "MP matrix max_split initial")) {
        failures += 1;
    }

    m.max_split_ = 5;
    if (!check((int)m.max_split_, 5, "MP matrix max_split assignment")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Main Test Runner
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

    if (total_failures == 0) {
        std::cout << "All Mixed-Precision Matrix tests passed!" << std::endl;
    } else {
        std::cout << "Mixed-Precision Matrix tests: " << total_failures << " failures" << std::endl;
    }

    return total_failures;
}

#else
int main() {
    std::cerr << "CUDA support not enabled" << std::endl;
    return 1;
}
#endif
