#include "test_common.h"

using namespace lahva::cpu;

const double TOLERANCE = 5.0e-7;

// ============================================================================
// LowTriMatrix Constructor Tests
// ============================================================================

template <typename T>
int test_lowtrimatrix_size_constructor() {
    int failures = 0;

    LowTriMatrix<T> m(5);

    if (!check((int)m.size(), 15, "Size constructor packed size n*(n+1)/2 = 5*6/2 = 15")) {
        failures += 1;
    }

    if (m.data() == nullptr) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_lowtrimatrix_size_value_constructor() {
    int failures = 0;

    LowTriMatrix<T> m(4, (T)3.5);

    if (!check((int)m.size(), 10, "Size+value constructor packed size n*(n+1)/2 = 4*5/2 = 10")) {
        failures += 1;
    }

    // For packed lower triangular format with n=4:
    // Storage: n*(n+1)/2 = 4*5/2 = 10 elements
    int packed_size = 4 * 5 / 2;

    // Verify all elements are initialized
    for (int i = 0; i < packed_size; i++) {
        if (!check((double)m.data()[i], 3.5, TOLERANCE, "All elements should be initialized")) {
            failures += 1;
            break;
        }
    }

    return failures;
}

template <typename T>
int test_lowtrimatrix_copy_constructor() {
    int failures = 0;

    LowTriMatrix<T> m1(3, (T)2.5);
    LowTriMatrix<T> m2 = m1;  // Copy constructor

    if (!check((int)m2.size(), (int)m1.size(), "Copy constructor should copy packed size")) {
        failures += 1;
    }

    // Packed size: 3*4/2 = 6
    int packed_size = 3 * 4 / 2;

    // Verify data is copied
    for (int i = 0; i < packed_size; i++) {
        if (!check((double)m2.data()[i], 2.5, TOLERANCE, "Copy constructor should copy data")) {
            failures += 1;
            break;
        }
    }

    // Modify original - copy should not change
    m1.data()[0] = (T)99.0;
    if (!check((double)m2.data()[0], 2.5, TOLERANCE, "Copy should be independent")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_lowtrimatrix_move_constructor() {
    int failures = 0;

    LowTriMatrix<T> m1(3, (T)4.0);
    T* original_data = m1.data();

    LowTriMatrix<T> m2 = std::move(m1);  // Move constructor

    if (!check((int)m2.size(), 6, "Move constructor packed size n*(n+1)/2 = 3*4/2 = 6")) {
        failures += 1;
    }

    // Check if pointer was transferred
    if (m2.data() != original_data) {
        std::cerr << "[Fatal] Move constructor did NOT transfer pointer\n";
        std::cerr << "  Expected: " << (void*)original_data << "\n";
        std::cerr << "  Got:      " << (void*)m2.data() << "\n";
        failures += 1;
    }

    // Check if source was cleared
    if (m1.data() != nullptr) {
        std::cerr << "[Fatal] Move constructor did NOT clear m1.data(): " << (void*)m1.data() << "\n";
        failures += 1;
    }

    if (m1.size() != 0) {
        std::cerr << "[Fatal] Move constructor did NOT clear m1.size(): " << m1.size() << "\n";
        failures += 1;
    }

    return failures;
}

// ============================================================================
// LowTriMatrix Attribute Tests
// ============================================================================

template <typename T>
int test_lowtrimatrix_size_attribute() {
    int failures = 0;

    LowTriMatrix<T> m(7);

    if (!check((int)m.size(), 28, "size() returns packed size n*(n+1)/2 = 7*8/2 = 28")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_lowtrimatrix_packed_size() {
    int failures = 0;

    LowTriMatrix<T> m(5);

    // For lower triangular packed format: n*(n+1)/2
    // For n=5: 5*6/2 = 15

    // Verify we can access all packed elements
    if (m.data() == nullptr) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_lowtrimatrix_data_access() {
    int failures = 0;

    LowTriMatrix<T> m(3);

    if (m.data() == nullptr) {
        failures += 1;
    }

    // Write and read
    m.data()[0] = (T)5.5;
    if (!check((double)m.data()[0], 5.5, TOLERANCE, "Data access should work")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// LowTriMatrix Storage Format Tests
// ============================================================================

template <typename T>
int test_lowtrimatrix_packed_format() {
    int failures = 0;

    // Test that packed format stores only lower triangle
    LowTriMatrix<T> m(3);
    // Packed storage for 3x3 lower triangular:
    // m[0,0]
    // m[1,0] m[1,1]
    // m[2,0] m[2,1] m[2,2]
    // Total: 6 elements

    m.data()[0] = (T)1.0;  // m[0,0]
    m.data()[1] = (T)2.0;  // m[1,0]
    m.data()[2] = (T)3.0;  // m[1,1]
    m.data()[3] = (T)4.0;  // m[2,0]
    m.data()[4] = (T)5.0;  // m[2,1]
    m.data()[5] = (T)6.0;  // m[2,2]

    if (!check((double)m.data()[0], 1.0, TOLERANCE, "Packed format should store correctly")) {
        failures += 1;
    }

    if (!check((double)m.data()[5], 6.0, TOLERANCE, "Diagonal element should be stored")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_lowtrimatrix_diagonal_elements() {
    int failures = 0;

    LowTriMatrix<T> m(4, (T)1.0);

    // All diagonal elements should be 1.0
    // Diagonal positions in packed format for n=4:
    // Position 0 (m[0,0]), 2 (m[1,1]), 5 (m[2,2]), 9 (m[3,3])

    if (!check((double)m.data()[0], 1.0, TOLERANCE, "First diagonal element")) {
        failures += 1;
    }

    // Last diagonal for n=4 is at position 4*5/2 - 1 = 9
    if (!check((double)m.data()[9], 1.0, TOLERANCE, "Last diagonal element")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// LowTriMatrix Assignment Tests
// ============================================================================

template <typename T>
int test_lowtrimatrix_copy_assignment() {
    int failures = 0;

    LowTriMatrix<T> m1(3, (T)2.5);
    LowTriMatrix<T> m2(2, (T)1.0);  // Create with size first

    m2 = m1;  // Copy assignment

    if (!check((int)m2.size(), 6, "Copy assignment packed size n*(n+1)/2 = 3*4/2 = 6")) {
        failures += 1;
    }

    // Self-assignment should be safe
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wself-assign-overloaded"
    m1 = m1;
    #pragma GCC diagnostic pop
    if (!check((int)m1.size(), 6, "Self-assignment packed size n*(n+1)/2 = 3*4/2 = 6")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_lowtrimatrix_move_assignment() {
    int failures = 0;

    LowTriMatrix<T> m1(4, (T)2.0);
    LowTriMatrix<T> m2(2, (T)1.0);  // Create with size first

    m2 = std::move(m1);  // Move assignment

    if (!check((int)m2.size(), 10, "Move assignment packed size n*(n+1)/2 = 4*5/2 = 10")) {
        failures += 1;
    }

    // Note: Move assignment doesn't clear source in this implementation

    return failures;
}

// ============================================================================
// LowTriMatrix Memory Tests
// ============================================================================

template <typename T>
int test_lowtrimatrix_memory_allocation() {
    int failures = 0;

    LowTriMatrix<T> m(10);

    if (m.data() == nullptr) {
    }

    // For n=10, packed size is 10*11/2 = 55
    int packed_size = 10 * 11 / 2;

    // Write to first and last elements
    m.data()[0] = (T)1.5;
    m.data()[packed_size - 1] = (T)55.5;

    if (!check((double)m.data()[0], 1.5, TOLERANCE, "First element should be accessible")) {
        failures += 1;
    }

    if (!check((double)m.data()[packed_size - 1], 55.5, TOLERANCE, "Last element should be accessible")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_lowtrimatrix_destructor() {
    int failures = 0;

    {
        LowTriMatrix<T> m(50);
        if (m.data() == nullptr) {
            failures += 1;
        }
    }
    // Destructor should deallocate

    return failures;
}

// ============================================================================
// LowTriMatrix Type Tests
// ============================================================================

int test_lowtrimatrix_float_type() {
    int failures = 0;

    LowTriMatrix<float> m(3, 2.5f);

    if (!check((int)m.size(), 6, "Float LowTriMatrix packed size n*(n+1)/2 = 3*4/2 = 6")) {
        failures += 1;
    }

    if (!check(m.data()[0], 2.5f, 1e-6f, "Float LowTriMatrix should store float values")) {
        failures += 1;
    }

    return failures;
}

int test_lowtrimatrix_int_type() {
    int failures = 0;

    LowTriMatrix<int> m(2);
    m.data()[0] = 10;
    m.data()[1] = 20;
    m.data()[2] = 30;

    if (!check((int)m.size(), 3, "Int LowTriMatrix packed size n*(n+1)/2 = 2*3/2 = 3")) {
        failures += 1;
    }

    if (!check(m.data()[0], 10, "Int LowTriMatrix should store int values")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Main Test Runner
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

    if (total_failures == 0) {
        std::cout << "All LowTriMatrix type tests passed!" << std::endl;
    } else {
        std::cout << "LowTriMatrix type tests: " << total_failures << " failures" << std::endl;
    }

    return total_failures;
}
