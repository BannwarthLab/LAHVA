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
    if (!check((int)v.size(), 0, make_check_msg(__func__, get_type_name<T>(), "Default constructor should create empty vector"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_vector_size_constructor() {
    // Test constructor with size
    Vector<T> v(10);

    if (!check((int)v.size(), 10, make_check_msg(__func__, get_type_name<T>(), "Size constructor should set correct size"))) return TEST_FAIL;

    // Verify data is accessible
    if (v.data() == nullptr) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_vector_size_value_constructor() {
    // Test constructor with size and initial value
    Vector<T> v(5, (T)3.5);

    if (!check((int)v.size(), 5, make_check_msg(__func__, get_type_name<T>(), "Size+value constructor should set correct size"))) return TEST_FAIL;

    // Verify all elements are initialized
    for (int i = 0; i < 5; i++) {
        if (!check((double)v.data()[i], 3.5, make_check_msg(__func__, get_type_name<T>(), "Elements should be initialized with value"))) return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_vector_initializer_list_constructor() {
    // Test constructor with initializer list
    Vector<T> v({(T)1.0, (T)2.0, (T)3.0, (T)4.0, (T)5.0});

    if (!check((int)v.size(), 5, make_check_msg(__func__, get_type_name<T>(), "Initializer list constructor should set correct size"))) return TEST_FAIL;

    // Verify values
    T expected[] = {(T)1.0, (T)2.0, (T)3.0, (T)4.0, (T)5.0};
    for (int i = 0; i < 5; i++) {
        if (!check((double)v.data()[i], (double)expected[i], make_check_msg(__func__, get_type_name<T>(), "Elements should match initializer list"))) return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_vector_copy_constructor() {
    Vector<T> v1({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> v2 = v1;  // Copy constructor

    if (!check((int)v2.size(), (int)v1.size(), make_check_msg(__func__, get_type_name<T>(), "Copy constructor should copy size"))) return TEST_FAIL;

    // Verify data is copied, not referenced
    T expected[] = {(T)1.0, (T)2.0, (T)3.0};
    for (int i = 0; i < 3; i++) {
        if (!check((double)v2.data()[i], (double)expected[i], make_check_msg(__func__, get_type_name<T>(), "Copy constructor should copy data"))) return TEST_FAIL;
    }

    // Modify original - copy should not change
    v1.data()[0] = (T)99.0;
    if (!check((double)v2.data()[0], 1.0, make_check_msg(__func__, get_type_name<T>(), "Copy should be independent"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_vector_move_constructor() {
    Vector<T> v1({(T)1.0, (T)2.0, (T)3.0});
    T* original_data = v1.data();

    Vector<T> v2 = std::move(v1);  // Move constructor

    if (!check((int)v2.size(), 3, make_check_msg(__func__, get_type_name<T>(), "Move constructor should transfer size"))) return TEST_FAIL;

    // Check if pointer was transferred
    if (v2.data() != original_data) {
        std::cerr << "[Fatal] Move constructor did NOT transfer pointer\n";
        std::cerr << "  Expected: " << (void*)original_data << "\n";
        std::cerr << "  Got:      " << (void*)v2.data() << "\n";
        return TEST_FAIL;
    }

    // Check if source was cleared
    if (v1.data() != nullptr) {
        std::cerr << "[Fatal] Move constructor did NOT clear v1.data(): " << (void*)v1.data() << "\n";
        return TEST_FAIL;
    }

    if (v1.size() != 0) {
        std::cerr << "[Fatal] Move constructor did NOT clear v1.size(): " << v1.size() << "\n";
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

    if (!check((int)v.size(), 25, make_check_msg(__func__, get_type_name<T>(), "Size attribute should return correct size"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_vector_data_access() {
    Vector<T> v({(T)5.0, (T)10.0, (T)15.0});

    // Test data() method
    if (v.data() == nullptr) return TEST_FAIL;

    // Verify we can read/write through pointer
    if (!check((double)v.data()[0], 5.0, make_check_msg(__func__, get_type_name<T>(), "Data access should work"))) return TEST_FAIL;

    v.data()[1] = (T)20.0;
    if (!check((double)v.data()[1], 20.0, make_check_msg(__func__, get_type_name<T>(), "Data modification should work"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_vector_const_data_access() {
    const Vector<T> v({(T)1.0, (T)2.0, (T)3.0});

    // Test const data() method
    const T* data = v.data();
    if (data == nullptr) return TEST_FAIL;

    if (!check((double)data[0], 1.0, make_check_msg(__func__, get_type_name<T>(), "Const data access should work"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_vector_empty_check() {
    Vector<T> empty;
    Vector<T> nonempty(5);

    // Note: May need to check actual implementation for empty() method
    if (!check((int)empty.size(), 0, make_check_msg(__func__, get_type_name<T>(), "Empty vector should have size 0"))) return TEST_FAIL;

    if (!check((int)nonempty.size(), 5, make_check_msg(__func__, get_type_name<T>(), "Non-empty vector should have non-zero size"))) return TEST_FAIL;

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

    if (!check((int)v2.size(), 3, make_check_msg(__func__, get_type_name<T>(), "Copy assignment should copy size"))) return TEST_FAIL;

    T expected[] = {(T)1.0, (T)2.0, (T)3.0};
    for (int i = 0; i < 3; i++) {
        if (!check((double)v2.data()[i], (double)expected[i], make_check_msg(__func__, get_type_name<T>(), "Copy assignment should copy data"))) return TEST_FAIL;
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
    if (!check((int)v1.size(), 3, make_check_msg(__func__, get_type_name<T>(), "Self-assignment should not break vector"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_vector_move_assignment() {
    Vector<T> v1({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> v2;

    v2 = std::move(v1);  // Move assignment

    if (!check((int)v2.size(), 3, make_check_msg(__func__, get_type_name<T>(), "Move assignment should transfer size"))) return TEST_FAIL;

    // Note: Move assignment doesn't clear source in this implementation

    return TEST_PASS;
}

// ============================================================================
// Vector Memory Management Tests
// ============================================================================

template <typename T>
int test_vector_memory_allocation() {
    Vector<T> v(1000);

    // Should allocate memory
    if (v.data() == nullptr) return TEST_FAIL;

    // Should be able to write entire array
    for (int i = 0; i < 1000; i++) {
        v.data()[i] = (T)(i * 1.5);
    }

    // Verify some values
    if (!check((double)v.data()[0], 0.0, make_check_msg(__func__, get_type_name<T>(), "First element should be 0"))) return TEST_FAIL;

    if (!check((double)v.data()[999], 1498.5, make_check_msg(__func__, get_type_name<T>(), "Last element should be correct"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_vector_destructor() {
    {
        Vector<T> v(100);
        if (v.data() == nullptr) return TEST_FAIL;
    }
    // Destructor should deallocate - we're just checking it doesn't crash

    return TEST_PASS;
}

// ============================================================================
// Vector Type Tests
// ============================================================================

int test_vector_float_type() {
    Vector<float> v(5, 3.14f);

    if (!check((int)v.size(), 5, make_check_msg(__func__, get_type_name<float>(), "Float vector should have correct size"))) return TEST_FAIL;

    for (int i = 0; i < 5; i++) {
        if (!check(v.data()[i], 3.14f, make_check_msg(__func__, get_type_name<float>(), "Float vector should store float values"))) return TEST_FAIL;
    }

    return TEST_PASS;
}

int test_vector_int_type() {
    Vector<int> v({1, 2, 3, 4, 5});

    if (!check((int)v.size(), 5, make_check_msg(__func__, get_type_name<int>(), "Int vector should have correct size"))) return TEST_FAIL;

    // Verify individual int values
    for (int i = 0; i < 5; i++) {
        if (!check(v.data()[i], i + 1, make_check_msg(__func__, get_type_name<int>(), "Int vector should store correct values"))) return TEST_FAIL;
    }

    return TEST_PASS;
}

int test_vector_complex_type() {
    Vector<std::complex<double>> v(3);
    v.data()[0] = std::complex<double>(1.0, 2.0);
    v.data()[1] = std::complex<double>(3.0, 4.0);
    v.data()[2] = std::complex<double>(5.0, 6.0);

    if (!check((int)v.size(), 3, make_check_msg(__func__, get_type_name<std::complex<double>>(), "Complex vector should have correct size"))) return TEST_FAIL;

    if (!check(v.data()[0], std::complex<double>(1.0, 2.0), make_check_msg(__func__, get_type_name<std::complex<double>>(), "Complex vector should store complex values"))) return TEST_FAIL;

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
