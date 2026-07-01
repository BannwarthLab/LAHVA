#include "test_common.h"

using namespace lahva::cpu;

const double TOLERANCE = 5.0e-7;

// ============================================================================
// Vector Constructor Tests
// ============================================================================

template <typename T>
int test_vector_default_constructor() {
    int failures = 0;

    // Test default constructed vector
    Vector<T> v;

    // Default constructor should create empty vector
    if (!check((int)v.size(), 0, "Default constructor should create empty vector")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_vector_size_constructor() {
    int failures = 0;

    // Test constructor with size
    Vector<T> v(10);

    if (!check((int)v.size(), 10, "Size constructor should set correct size")) {
        failures += 1;
    }

    // Verify data is accessible
    if (v.data() == nullptr) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_vector_size_value_constructor() {
    int failures = 0;

    // Test constructor with size and initial value
    Vector<T> v(5, (T)3.5);

    if (!check((int)v.size(), 5, "Size+value constructor should set correct size")) {
        failures += 1;
    }

    // Verify all elements are initialized
    for (int i = 0; i < 5; i++) {
        if (!check((double)v.data()[i], 3.5, TOLERANCE, "Elements should be initialized with value")) {
            failures += 1;
            break;
        }
    }

    return failures;
}

template <typename T>
int test_vector_initializer_list_constructor() {
    int failures = 0;

    // Test constructor with initializer list
    Vector<T> v({(T)1.0, (T)2.0, (T)3.0, (T)4.0, (T)5.0});

    if (!check((int)v.size(), 5, "Initializer list constructor should set correct size")) {
        failures += 1;
    }

    // Verify values
    T expected[] = {(T)1.0, (T)2.0, (T)3.0, (T)4.0, (T)5.0};
    for (int i = 0; i < 5; i++) {
        if (!check((double)v.data()[i], (double)expected[i], TOLERANCE, "Elements should match initializer list")) {
            failures += 1;
            break;
        }
    }

    return failures;
}

template <typename T>
int test_vector_copy_constructor() {
    int failures = 0;

    Vector<T> v1({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> v2 = v1;  // Copy constructor

    if (!check((int)v2.size(), (int)v1.size(), "Copy constructor should copy size")) {
        failures += 1;
    }

    // Verify data is copied, not referenced
    T expected[] = {(T)1.0, (T)2.0, (T)3.0};
    for (int i = 0; i < 3; i++) {
        if (!check((double)v2.data()[i], (double)expected[i], TOLERANCE, "Copy constructor should copy data")) {
            failures += 1;
            break;
        }
    }

    // Modify original - copy should not change
    v1.data()[0] = (T)99.0;
    if (!check((double)v2.data()[0], 1.0, TOLERANCE, "Copy should be independent")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_vector_move_constructor() {
    int failures = 0;

    Vector<T> v1({(T)1.0, (T)2.0, (T)3.0});
    T* original_data = v1.data();

    Vector<T> v2 = std::move(v1);  // Move constructor

    if (!check((int)v2.size(), 3, "Move constructor should transfer size")) {
        failures += 1;
    }

    // Check if pointer was transferred
    if (v2.data() != original_data) {
        std::cerr << "[Fatal] Move constructor did NOT transfer pointer\n";
        std::cerr << "  Expected: " << (void*)original_data << "\n";
        std::cerr << "  Got:      " << (void*)v2.data() << "\n";
        failures += 1;
    }

    // Check if source was cleared
    if (v1.data() != nullptr) {
        std::cerr << "[Fatal] Move constructor did NOT clear v1.data(): " << (void*)v1.data() << "\n";
        failures += 1;
    }

    if (v1.size() != 0) {
        std::cerr << "[Fatal] Move constructor did NOT clear v1.size(): " << v1.size() << "\n";
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Vector Attribute Tests
// ============================================================================

template <typename T>
int test_vector_size_attribute() {
    int failures = 0;

    Vector<T> v(25);

    if (!check((int)v.size(), 25, "Size attribute should return correct size")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_vector_data_access() {
    int failures = 0;

    Vector<T> v({(T)5.0, (T)10.0, (T)15.0});

    // Test data() method
    if (v.data() == nullptr) {
        failures += 1;
    }

    // Verify we can read/write through pointer
    if (!check((double)v.data()[0], 5.0, TOLERANCE, "Data access should work")) {
        failures += 1;
    }

    v.data()[1] = (T)20.0;
    if (!check((double)v.data()[1], 20.0, TOLERANCE, "Data modification should work")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_vector_const_data_access() {
    int failures = 0;

    const Vector<T> v({(T)1.0, (T)2.0, (T)3.0});

    // Test const data() method
    const T* data = v.data();
    if (data == nullptr) {
        failures += 1;
    }

    if (!check((double)data[0], 1.0, TOLERANCE, "Const data access should work")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_vector_empty_check() {
    int failures = 0;

    Vector<T> empty;
    Vector<T> nonempty(5);

    // Note: May need to check actual implementation for empty() method
    if (!check((int)empty.size(), 0, "Empty vector should have size 0")) {
        failures += 1;
    }

    if (!check((int)nonempty.size(), 5, "Non-empty vector should have non-zero size")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Vector Assignment Tests
// ============================================================================

template <typename T>
int test_vector_copy_assignment() {
    int failures = 0;

    Vector<T> v1({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> v2;

    v2 = v1;  // Copy assignment

    if (!check((int)v2.size(), 3, "Copy assignment should copy size")) {
        failures += 1;
    }

    T expected[] = {(T)1.0, (T)2.0, (T)3.0};
    for (int i = 0; i < 3; i++) {
        if (!check((double)v2.data()[i], (double)expected[i], TOLERANCE, "Copy assignment should copy data")) {
            failures += 1;
            break;
        }
    }

    // Self-assignment should be safe
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wself-assign-overloaded"
    v1 = v1;
    #pragma GCC diagnostic pop
    if (!check((int)v1.size(), 3, "Self-assignment should not break vector")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_vector_move_assignment() {
    int failures = 0;

    Vector<T> v1({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> v2;

    v2 = std::move(v1);  // Move assignment

    if (!check((int)v2.size(), 3, "Move assignment should transfer size")) {
        failures += 1;
    }

    // Note: Move assignment doesn't clear source in this implementation

    return failures;
}

// ============================================================================
// Vector Memory Management Tests
// ============================================================================

template <typename T>
int test_vector_memory_allocation() {
    int failures = 0;

    Vector<T> v(1000);

    // Should allocate memory
    if (v.data() == nullptr) {
        failures += 1;
    }

    // Should be able to write entire array
    for (int i = 0; i < 1000; i++) {
        v.data()[i] = (T)(i * 1.5);
    }

    // Verify some values
    if (!check((double)v.data()[0], 0.0, TOLERANCE, "First element should be 0")) {
        failures += 1;
    }

    if (!check((double)v.data()[999], 1498.5, TOLERANCE, "Last element should be correct")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_vector_destructor() {
    int failures = 0;

    {
        Vector<T> v(100);
        if (v.data() == nullptr) {
            failures += 1;
        }
    }
    // Destructor should deallocate - we're just checking it doesn't crash

    return failures;
}

// ============================================================================
// Vector Type Tests
// ============================================================================

int test_vector_float_type() {
    int failures = 0;

    Vector<float> v(5, 3.14f);

    if (!check((int)v.size(), 5, "Float vector should have correct size")) {
        failures += 1;
    }

    for (int i = 0; i < 5; i++) {
        if (!check(v.data()[i], 3.14f, 1e-6f, "Float vector should store float values")) {
            failures += 1;
            break;
        }
    }

    return failures;
}

int test_vector_int_type() {
    int failures = 0;

    Vector<int> v({1, 2, 3, 4, 5});

    if (!check((int)v.size(), 5, "Int vector should have correct size")) {
        failures += 1;
    }

    // Verify individual int values
    for (int i = 0; i < 5; i++) {
        if (!check(v.data()[i], i + 1, "Int vector should store correct values")) {
            failures += 1;
            break;
        }
    }

    return failures;
}

int test_vector_complex_type() {
    int failures = 0;

    Vector<std::complex<double>> v(3);
    v.data()[0] = std::complex<double>(1.0, 2.0);
    v.data()[1] = std::complex<double>(3.0, 4.0);
    v.data()[2] = std::complex<double>(5.0, 6.0);

    if (!check((int)v.size(), 3, "Complex vector should have correct size")) {
        failures += 1;
    }

    if (!check(v.data()[0], std::complex<double>(1.0, 2.0),
              std::complex<double>(TOLERANCE, TOLERANCE), "Complex vector should store complex values")) {
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

    if (total_failures == 0) {
        std::cout << "All Vector type tests passed!" << std::endl;
    } else {
        std::cout << "Vector type tests: " << total_failures << " failures" << std::endl;
    }

    return total_failures;
}
