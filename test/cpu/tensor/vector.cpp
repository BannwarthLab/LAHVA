#include "test_common.h"

using namespace lahva::cpu;

// ============================================================================
// Vector Constructor Tests
// ============================================================================

template <typename T>
int test_vector_default_constructor() {
    // Test default constructed vector
    Vector<T> v;

    // Default constructor should create empty vector
    if (!check((int)v.size(), 0, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_vector_size_constructor() {
    // Test constructor with size
    Vector<T> v(10);

    if (!check((int)v.size(), 10, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    // Verify data is accessible
    if (v.data() == nullptr) {
        std::cerr << check_msg(get_type_name<T>(), "check 2") << std::endl;
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_vector_size_value_constructor() {
    // Test constructor with size and initial value
    Vector<T> v(5, (T)3.5);

    if (!check((int)v.size(), 5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    // Verify all elements are initialized
    for (int i = 0; i < 5; i++) {
        if (!check((double)v.data()[i], 3.5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_vector_initializer_list_constructor() {
    // Test constructor with initializer list
    Vector<T> v({(T)1.0, (T)2.0, (T)3.0, (T)4.0, (T)5.0});

    if (!check((int)v.size(), 5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    // Verify values
    T expected[] = {(T)1.0, (T)2.0, (T)3.0, (T)4.0, (T)5.0};
    for (int i = 0; i < 5; i++) {
        if (!check((double)v.data()[i], (double)expected[i], check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_vector_copy_constructor() {
    Vector<T> v1({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> v2 = v1;  // Copy constructor

    if (!check((int)v2.size(), (int)v1.size(), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    // Verify data is copied, not referenced
    T expected[] = {(T)1.0, (T)2.0, (T)3.0};
    for (int i = 0; i < 3; i++) {
        if (!check((double)v2.data()[i], (double)expected[i], check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    }

    // Modify original - copy should not change
    v1.data()[0] = (T)99.0;
    if (!check((double)v2.data()[0], 1.0, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_vector_move_constructor() {
    Vector<T> v1({(T)1.0, (T)2.0, (T)3.0});
    T* original_data = v1.data();

    Vector<T> v2 = std::move(v1);  // Move constructor

    if (!check((int)v2.size(), 3, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    // Check if pointer was transferred
    if (v2.data() != original_data) {
        std::cerr << check_msg(get_type_name<T>(), "check 2") << std::endl;
        return TEST_FAIL;
    }

    // Check if source was cleared
    if (v1.data() != nullptr) {
        std::cerr << check_msg(get_type_name<T>(), "check 3") << std::endl;
        return TEST_FAIL;
    }

    if (v1.size() != 0) {
        std::cerr << check_msg(get_type_name<T>(), "check 4") << std::endl;
        return TEST_FAIL;
    }

    return TEST_PASS;
}

// ============================================================================
// Vector Attribute Tests
// ============================================================================

template <typename T>
int test_vector_size_attribute() {
    Vector<T> v(25);

    if (!check((int)v.size(), 25, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_vector_data_access() {
    Vector<T> v({(T)5.0, (T)10.0, (T)15.0});

    // Test data() method
    if (v.data() == nullptr) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    }

    // Verify we can read/write through pointer
    if (!check((double)v.data()[0], 5.0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    v.data()[1] = (T)20.0;
    if (!check((double)v.data()[1], 20.0, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_vector_const_data_access() {
    const Vector<T> v({(T)1.0, (T)2.0, (T)3.0});

    // Test const data() method
    const T* data = v.data();
    if (data == nullptr) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    }

    if (!check((double)data[0], 1.0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_vector_empty_check() {
    Vector<T> empty;
    Vector<T> nonempty(5);

    // Note: May need to check actual implementation for empty() method
    if (!check((int)empty.size(), 0, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)nonempty.size(), 5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Vector Assignment Tests
// ============================================================================

template <typename T>
int test_vector_copy_assignment() {
    Vector<T> v1({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> v2;

    v2 = v1;  // Copy assignment

    if (!check((int)v2.size(), 3, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    T expected[] = {(T)1.0, (T)2.0, (T)3.0};
    for (int i = 0; i < 3; i++) {
        if (!check((double)v2.data()[i], (double)expected[i], check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    }

    // Self-assignment should be safe
    #if defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wself-assign-overloaded"
    #endif
    v1 = v1;
    #if defined(__clang__)
    #pragma GCC diagnostic pop
    #endif
    if (!check((int)v1.size(), 3, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_vector_move_assignment() {
    Vector<T> v1({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> v2;

    v2 = std::move(v1);  // Move assignment

    if (!check((int)v2.size(), 3, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Vector Memory Management Tests
// ============================================================================

template <typename T>
int test_vector_memory_allocation() {
    Vector<T> v(1000);

    // Should allocate memory
    if (v.data() == nullptr) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    }

    // Should be able to write entire array
    for (int i = 0; i < 1000; i++) {
        v.data()[i] = (T)(i * 1.5);
    }

    if (!check((double)v.data()[0], 0.0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((double)v.data()[999], 1498.5, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_vector_destructor() {
    {
        Vector<T> v(100);
        if (v.data() == nullptr) {
            std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

// ============================================================================
// Vector Type Tests
// ============================================================================

int test_vector_float_type() {
    Vector<float> v(5, 3.14f);

    if (!check((int)v.size(), 5, check_msg(get_type_name<float>(), "check 1"))) return TEST_FAIL;

    for (int i = 0; i < 5; i++) {
        if (!check(v.data()[i], 3.14f, check_msg(get_type_name<float>(), "check 2"))) return TEST_FAIL;
    }

    return TEST_PASS;
}

int test_vector_int_type() {
    Vector<int> v({1, 2, 3, 4, 5});

    if (!check((int)v.size(), 5, check_msg(get_type_name<int>(), "check 1"))) return TEST_FAIL;

    // Verify individual int values
    for (int i = 0; i < 5; i++) {
        if (!check(v.data()[i], i + 1, check_msg(get_type_name<int>(), "check 2"))) return TEST_FAIL;
    }

    return TEST_PASS;
}

int test_vector_complex_type() {
    Vector<std::complex<double>> v(3);
    v.data()[0] = std::complex<double>(1.0, 2.0);
    v.data()[1] = std::complex<double>(3.0, 4.0);
    v.data()[2] = std::complex<double>(5.0, 6.0);

    if (!check((int)v.size(), 3, check_msg(get_type_name<std::complex<double>>(), "check 1"))) return TEST_FAIL;

    if (!check(v.data()[0], std::complex<double>(1.0, 2.0), check_msg(get_type_name<std::complex<double>>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int total_failures = 0;

    // Constructors - double
    total_failures += test_vector_default_constructor<double>();
    total_failures += test_vector_size_constructor<double>();
    total_failures += test_vector_size_value_constructor<double>();
    total_failures += test_vector_initializer_list_constructor<double>();
    total_failures += test_vector_copy_constructor<double>();
    total_failures += test_vector_move_constructor<double>();

    // Constructors - float
    total_failures += test_vector_default_constructor<float>();
    total_failures += test_vector_size_constructor<float>();
    total_failures += test_vector_size_value_constructor<float>();
    total_failures += test_vector_initializer_list_constructor<float>();
    total_failures += test_vector_copy_constructor<float>();
    total_failures += test_vector_move_constructor<float>();

    // Attributes - double
    total_failures += test_vector_size_attribute<double>();
    total_failures += test_vector_data_access<double>();
    total_failures += test_vector_const_data_access<double>();
    total_failures += test_vector_empty_check<double>();

    // Attributes - float
    total_failures += test_vector_size_attribute<float>();
    total_failures += test_vector_data_access<float>();
    total_failures += test_vector_const_data_access<float>();
    total_failures += test_vector_empty_check<float>();

    // Assignment - double
    total_failures += test_vector_copy_assignment<double>();
    total_failures += test_vector_move_assignment<double>();

    // Assignment - float
    total_failures += test_vector_copy_assignment<float>();
    total_failures += test_vector_move_assignment<float>();

    // Memory - double
    total_failures += test_vector_memory_allocation<double>();
    total_failures += test_vector_destructor<double>();

    // Memory - float
    total_failures += test_vector_memory_allocation<float>();
    total_failures += test_vector_destructor<float>();

    // Types (specific type tests - not templated)
    total_failures += test_vector_float_type();
    total_failures += test_vector_int_type();
    total_failures += test_vector_complex_type();

    if (total_failures > 0) {
        std::cerr << "cpu/tensor/vector tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All cpu/tensor/vector tests passed!" << std::endl;
    return TEST_PASS;
}
