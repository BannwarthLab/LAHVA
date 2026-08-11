#include "test_common.h"

using namespace lahva::cpu;

// ============================================================================
// LowTriMatrix Constructor Tests
// ============================================================================

template <typename T>
int test_lowtrimatrix_size_constructor() {
    LowTriMatrix<T> m(5);

    if (!check((int)m.size(), 15, check_msg(get_type_name<T>(), "Size constructor packed size n*(n+1)/2 = 5*6/2 = 15"))) return TEST_FAIL;

    if (m.data() == nullptr) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_lowtrimatrix_size_value_constructor() {
    LowTriMatrix<T> m(4, (T)3.5);

    if (!check((int)m.size(), 10, check_msg(get_type_name<T>(), "Size+value constructor packed size n*(n+1)/2 = 4*5/2 = 10"))) return TEST_FAIL;

    // For packed lower triangular format with n=4:
    // Storage: n*(n+1)/2 = 4*5/2 = 10 elements
    int packed_size = 4 * 5 / 2;

    // Verify all elements are initialized
    for (int i = 0; i < packed_size; i++) {
        if (!check((double)m.data()[i], 3.5, check_msg(get_type_name<T>(), "All elements should be initialized"))) return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_lowtrimatrix_copy_constructor() {
    LowTriMatrix<T> m1(3, (T)2.5);
    LowTriMatrix<T> m2 = m1;  // Copy constructor

    if (!check((int)m2.size(), (int)m1.size(), check_msg(get_type_name<T>(), "Copy constructor should copy packed size"))) return TEST_FAIL;

    // Packed size: 3*4/2 = 6
    int packed_size = 3 * 4 / 2;

    // Verify data is copied
    for (int i = 0; i < packed_size; i++) {
        if (!check((double)m2.data()[i], 2.5, check_msg(get_type_name<T>(), "Copy constructor should copy data"))) return TEST_FAIL;
    }

    // Modify original - copy should not change
    m1.data()[0] = (T)99.0;
    if (!check((double)m2.data()[0], 2.5, check_msg(get_type_name<T>(), "Copy should be independent"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_lowtrimatrix_move_constructor() {
    LowTriMatrix<T> m1(3, (T)4.0);
    T* original_data = m1.data();

    LowTriMatrix<T> m2 = std::move(m1);  // Move constructor

    if (!check((int)m2.size(), 6, check_msg(get_type_name<T>(), "Move constructor packed size n*(n+1)/2 = 3*4/2 = 6"))) return TEST_FAIL;

    // Check if pointer was transferred
    if (m2.data() != original_data) {
        std::cerr << "[Fatal] Move constructor did NOT transfer pointer\n";
        std::cerr << "  Expected: " << (void*)original_data << "\n";
        std::cerr << "  Got:      " << (void*)m2.data() << "\n";
        return TEST_FAIL;
    }

    // Check if source was cleared
    if (m1.data() != nullptr) {
        std::cerr << "[Fatal] Move constructor did NOT clear m1.data(): " << (void*)m1.data() << "\n";
        return TEST_FAIL;
    }

    if (m1.size() != 0) {
        std::cerr << "[Fatal] Move constructor did NOT clear m1.size(): " << m1.size() << "\n";
        return TEST_FAIL;
    }

    return TEST_PASS;
}

// ============================================================================
// LowTriMatrix Attribute Tests
// ============================================================================

template <typename T>
int test_lowtrimatrix_size_attribute() {
    LowTriMatrix<T> m(7);

    if (!check((int)m.size(), 28, check_msg(get_type_name<T>(), "size() returns packed size n*(n+1)/2 = 7*8/2 = 28"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_lowtrimatrix_packed_size() {
    LowTriMatrix<T> m(5);

    if (m.data() == nullptr) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_lowtrimatrix_data_access() {
    LowTriMatrix<T> m(3);

    if (m.data() == nullptr) return TEST_FAIL;

    m.data()[0] = (T)5.5;
    if (!check((double)m.data()[0], 5.5, check_msg(get_type_name<T>(), "Data access should work"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// LowTriMatrix Storage Format Tests
// ============================================================================

template <typename T>
int test_lowtrimatrix_packed_format() {
    LowTriMatrix<T> m(3);

    m.data()[0] = (T)1.0;  // m[0,0]
    m.data()[1] = (T)2.0;  // m[1,0]
    m.data()[2] = (T)3.0;  // m[1,1]
    m.data()[3] = (T)4.0;  // m[2,0]
    m.data()[4] = (T)5.0;  // m[2,1]
    m.data()[5] = (T)6.0;  // m[2,2]

    if (!check((double)m.data()[0], 1.0, check_msg(get_type_name<T>(), "Packed format should store correctly"))) return TEST_FAIL;

    if (!check((double)m.data()[5], 6.0, check_msg(get_type_name<T>(), "Diagonal element should be stored"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_lowtrimatrix_diagonal_elements() {
    LowTriMatrix<T> m(4, (T)1.0);

    if (!check((double)m.data()[0], 1.0, check_msg(get_type_name<T>(), "First diagonal element"))) return TEST_FAIL;

    if (!check((double)m.data()[9], 1.0, check_msg(get_type_name<T>(), "Last diagonal element"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// LowTriMatrix Assignment Tests
// ============================================================================

template <typename T>
int test_lowtrimatrix_copy_assignment() {
    LowTriMatrix<T> m1(3, (T)2.5);
    LowTriMatrix<T> m2(2, (T)1.0);  // Create with size first

    m2 = m1;  // Copy assignment

    if (!check((int)m2.size(), 6, check_msg(get_type_name<T>(), "Copy assignment packed size n*(n+1)/2 = 3*4/2 = 6"))) return TEST_FAIL;

    // Self-assignment should be safe
    #if defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wself-assign-overloaded"
    #endif
    m1 = m1;
    #if defined(__clang__)
    #pragma GCC diagnostic pop
    #endif
    if (!check((int)m1.size(), 6, check_msg(get_type_name<T>(), "Self-assignment packed size n*(n+1)/2 = 3*4/2 = 6"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_lowtrimatrix_move_assignment() {
    LowTriMatrix<T> m1(4, (T)2.0);
    LowTriMatrix<T> m2(2, (T)1.0);  // Create with size first

    m2 = std::move(m1);  // Move assignment

    if (!check((int)m2.size(), 10, check_msg(get_type_name<T>(), "Move assignment packed size n*(n+1)/2 = 4*5/2 = 10"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// LowTriMatrix Memory Tests
// ============================================================================

template <typename T>
int test_lowtrimatrix_memory_allocation() {
    LowTriMatrix<T> m(10);

    if (m.data() == nullptr) return TEST_FAIL;

    // For n=10, packed size is 10*11/2 = 55
    int packed_size = 10 * 11 / 2;

    // Write to first and last elements
    m.data()[0] = (T)1.5;
    m.data()[packed_size - 1] = (T)55.5;

    if (!check((double)m.data()[0], 1.5, check_msg(get_type_name<T>(), "First element should be accessible"))) return TEST_FAIL;

    if (!check((double)m.data()[packed_size - 1], 55.5, check_msg(get_type_name<T>(), "Last element should be accessible"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_lowtrimatrix_destructor() {
    {
        LowTriMatrix<T> m(50);
        if (m.data() == nullptr) return TEST_FAIL;
    }

    return TEST_PASS;
}

// ============================================================================
// LowTriMatrix Type Tests
// ============================================================================

int test_lowtrimatrix_float_type() {
    LowTriMatrix<float> m(3, 2.5f);

    if (!check((int)m.size(), 6, check_msg(get_type_name<float>(), "Float LowTriMatrix packed size n*(n+1)/2 = 3*4/2 = 6"))) return TEST_FAIL;

    if (!check(m.data()[0], 2.5f, check_msg(get_type_name<float>(), "Float LowTriMatrix should store float values"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_lowtrimatrix_int_type() {
    LowTriMatrix<int> m(2);
    m.data()[0] = 10;
    m.data()[1] = 20;
    m.data()[2] = 30;

    if (!check((int)m.size(), 3, check_msg(get_type_name<int>(), "Int LowTriMatrix packed size n*(n+1)/2 = 2*3/2 = 3"))) return TEST_FAIL;

    if (!check(m.data()[0], 10, check_msg(get_type_name<int>(), "Int LowTriMatrix should store int values"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int total_failures = 0;

    // Constructors - double
    total_failures += test_lowtrimatrix_size_constructor<double>();
    total_failures += test_lowtrimatrix_size_value_constructor<double>();
    total_failures += test_lowtrimatrix_copy_constructor<double>();
    total_failures += test_lowtrimatrix_move_constructor<double>();

    // Constructors - float
    total_failures += test_lowtrimatrix_size_constructor<float>();
    total_failures += test_lowtrimatrix_size_value_constructor<float>();
    total_failures += test_lowtrimatrix_copy_constructor<float>();
    total_failures += test_lowtrimatrix_move_constructor<float>();

    // Attributes - double
    total_failures += test_lowtrimatrix_size_attribute<double>();
    total_failures += test_lowtrimatrix_packed_size<double>();
    total_failures += test_lowtrimatrix_data_access<double>();

    // Attributes - float
    total_failures += test_lowtrimatrix_size_attribute<float>();
    total_failures += test_lowtrimatrix_packed_size<float>();
    total_failures += test_lowtrimatrix_data_access<float>();

    // Storage format - double
    total_failures += test_lowtrimatrix_packed_format<double>();
    total_failures += test_lowtrimatrix_diagonal_elements<double>();

    // Storage format - float
    total_failures += test_lowtrimatrix_packed_format<float>();
    total_failures += test_lowtrimatrix_diagonal_elements<float>();

    // Assignment - double
    total_failures += test_lowtrimatrix_copy_assignment<double>();
    total_failures += test_lowtrimatrix_move_assignment<double>();

    // Assignment - float
    total_failures += test_lowtrimatrix_copy_assignment<float>();
    total_failures += test_lowtrimatrix_move_assignment<float>();

    // Memory - double
    total_failures += test_lowtrimatrix_memory_allocation<double>();
    total_failures += test_lowtrimatrix_destructor<double>();

    // Memory - float
    total_failures += test_lowtrimatrix_memory_allocation<float>();
    total_failures += test_lowtrimatrix_destructor<float>();

    // Types (specific type tests - not templated)
    total_failures += test_lowtrimatrix_float_type();
    total_failures += test_lowtrimatrix_int_type();

    if (total_failures > 0) {
        std::cerr << "cpu/tensor/lowtrimatrix tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All cpu/tensor/lowtrimatrix tests passed!" << std::endl;
    return TEST_PASS;
}
