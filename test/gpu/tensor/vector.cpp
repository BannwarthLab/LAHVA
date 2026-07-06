#include "test_common.h"
#ifdef _CUDA

using namespace lahva::gpu;

// ============================================================================
// GPU Vector Constructor Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_vector_default_constructor() {
    int failures = 0;

    Vector<T> v;

    if (!check((int)v.size(), 0, "GPU default constructor should create empty vector")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_vector_size_constructor() {
    int failures = 0;

    Vector<T> v(10);

    if (!check((int)v.size(), 10, "GPU size constructor should set correct size")) {
        failures += 1;
    }

    // Should allocate both host and device memory
    if (v.data() == nullptr) {  // Host data
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_vector_size_value_constructor() {
    int failures = 0;

    Vector<T> v(5, (T)3.5);

    if (!check((int)v.size(), 5, "GPU size+value constructor should set correct size")) {
        failures += 1;
    }

    // Verify all elements are initialized on host
    double tol = get_tolerance<T>();
    for (int i = 0; i < 5; i++) {
        if (!check(v.data()[i], (T)3.5, tol, "GPU elements should be initialized")) {
            failures += 1;
            break;
        }
    }

    return failures;
}

template <typename T>
int test_gpu_vector_copy_constructor() {
    int failures = 0;

    Vector<T> v1({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> v2 = v1;  // Copy constructor

    if (!check((int)v2.size(), (int)v1.size(), "GPU copy constructor should copy size")) {
        failures += 1;
    }

    // Verify data is copied
    T expected[] = {(T)1.0, (T)2.0, (T)3.0};
    double tol = get_tolerance<T>();
    if (!check(v2.data(), expected, tol, 3, "GPU copy constructor should copy data")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_vector_move_constructor() {
    int failures = 0;

    Vector<T> v1({(T)1.0, (T)2.0, (T)3.0});

    Vector<T> v2 = std::move(v1);  // Move constructor

    if (!check((int)v2.size(), 3, "GPU move constructor should transfer size")) {
        failures += 1;
    }

    // Original should be empty after move
    if (!check((int)v1.size(), 0, "Original should be empty after move")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// GPU Vector Memory Management Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_vector_host_device_memory() {
    int failures = 0;

    Vector<T> v(10, (T)2.5);

    // Should have both host and device memory
    if (v.data() == nullptr) {  // Host pointer
        failures += 1;
    }

    // Verify host data
    double tol = get_tolerance<T>();
    for (int i = 0; i < 10; i++) {
        if (!check(v.data()[i], (T)2.5, tol, "Host memory should be initialized")) {
            failures += 1;
            break;
        }
    }

    return failures;
}

template <typename T>
int test_gpu_vector_data_host_sync() {
    int failures = 0;

    lahva::CudaRuntime runtime;
    Vector<T> v(5, (T)1.0);

    // Modify host data
    v.data()[0] = (T)5.0;

    // Access host data pointer
    T* host_data = v.data();
    if (host_data == nullptr) {
        failures += 1;
    }

    double tol = get_tolerance<T>();
    if (!check(host_data[0], (T)5.0, tol, "data_host() should return current host data")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_vector_synchronization() {
    int failures = 0;

    Vector<T> v(5, (T)3.0);

    // After operations on GPU, sync() should synchronize
    // (assuming sync() method exists)
    // This is a placeholder test

    if (!check((int)v.size(), 5, "GPU vector should maintain size after operations")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// GPU Vector Attribute Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_vector_size_attribute() {
    int failures = 0;

    Vector<T> v(25);

    if (!check((int)v.size(), 25, "GPU vector size should return correct value")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_vector_data_access() {
    int failures = 0;

    Vector<T> v({(T)5.0, (T)10.0, (T)15.0});

    T* data = v.data();
    if (data == nullptr) {
        failures += 1;
    }

    double tol = get_tolerance<T>();
    if (!check(data[0], (T)5.0, tol, "GPU vector data access should work")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_vector_type_float() {
    int failures = 0;

    Vector<float> v(5, 3.14f);

    if (!check((int)v.size(), 5, "GPU float vector should have correct size")) {
        failures += 1;
    }

    for (int i = 0; i < 5; i++) {
        if (!check(v.data()[i], 3.14f, 1e-6f, "GPU float vector should store float values")) {
            failures += 1;
            break;
        }
    }

    return failures;
}

int test_gpu_vector_type_int() {
    int failures = 0;

    Vector<int> v({1, 2, 3, 4, 5});

    if (!check((int)v.size(), 5, "GPU int vector should have correct size")) {
        failures += 1;
    }

    int expected[] = {1, 2, 3, 4, 5};
    // Verify individual int values due to ambiguous check overload
    for (int i = 0; i < 5; i++) {
        if (!check(v.data()[i], expected[i], "GPU int vector should store int values")) {
            failures += 1;
            break;
        }
    }

    return failures;
}

// ============================================================================
// GPU Vector Assignment Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_vector_copy_assignment() {
    int failures = 0;

    Vector<T> v1({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> v2;

    v2 = v1;  // Copy assignment

    if (!check((int)v2.size(), 3, "GPU copy assignment should copy size")) {
        failures += 1;
    }

    T expected[] = {(T)1.0, (T)2.0, (T)3.0};
    double tol = get_tolerance<T>();
    if (!check(v2.data(), expected, tol, 3, "GPU copy assignment should copy data")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_vector_move_assignment() {
    int failures = 0;

    Vector<T> v1({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> v2;

    v2 = std::move(v1);

    if (!check((int)v2.size(), 3, "GPU move assignment should transfer size")) {
        failures += 1;
    }

    if (!check((int)v1.size(), 0, "GPU original should be empty after move assignment")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// GPU Vector Destructor Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_vector_destructor() {
    int failures = 0;

    {
        Vector<T> v(100);
        if (v.data() == nullptr) {
            failures += 1;
        }
    }
    // Destructor should deallocate both host and device memory

    return failures;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    int total_failures = 0;

    // Constructors - test with double and float
    total_failures += test_gpu_vector_default_constructor<double>();
    total_failures += test_gpu_vector_default_constructor<float>();
    total_failures += test_gpu_vector_size_constructor<double>();
    total_failures += test_gpu_vector_size_constructor<float>();
    total_failures += test_gpu_vector_size_value_constructor<double>();
    total_failures += test_gpu_vector_size_value_constructor<float>();
    total_failures += test_gpu_vector_copy_constructor<double>();
    total_failures += test_gpu_vector_copy_constructor<float>();
    total_failures += test_gpu_vector_move_constructor<double>();
    total_failures += test_gpu_vector_move_constructor<float>();

    // Memory management
    total_failures += test_gpu_vector_host_device_memory<double>();
    total_failures += test_gpu_vector_host_device_memory<float>();
    total_failures += test_gpu_vector_data_host_sync<double>();
    total_failures += test_gpu_vector_data_host_sync<float>();
    total_failures += test_gpu_vector_synchronization<double>();
    total_failures += test_gpu_vector_synchronization<float>();

    // Attributes
    total_failures += test_gpu_vector_size_attribute<double>();
    total_failures += test_gpu_vector_size_attribute<float>();
    total_failures += test_gpu_vector_data_access<double>();
    total_failures += test_gpu_vector_data_access<float>();

    // Types (type-specific tests)
    total_failures += test_gpu_vector_type_float();
    total_failures += test_gpu_vector_type_int();

    // Assignment
    total_failures += test_gpu_vector_copy_assignment<double>();
    total_failures += test_gpu_vector_copy_assignment<float>();
    total_failures += test_gpu_vector_move_assignment<double>();
    total_failures += test_gpu_vector_move_assignment<float>();

    // Destructor
    total_failures += test_gpu_vector_destructor<double>();
    total_failures += test_gpu_vector_destructor<float>();

    if (total_failures == 0) {
        std::cout << "All GPU Vector type tests passed!" << std::endl;
    } else {
        std::cout << "GPU Vector type tests: " << total_failures << " failures" << std::endl;
    }

    return total_failures;
}

#else
int main() {
    std::cerr << "CUDA support not enabled" << std::endl;
    return 1;
}
#endif
