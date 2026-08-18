#include "test_common.h"

using namespace lahva::gpu;

// ============================================================================
// GPU Vector Constructor Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_vector_default_constructor() {

    Vector<T> v;

    if (!check((int)v.size(), 0, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_vector_size_constructor() {

    Vector<T> v(10);

    if (!check((int)v.size(), 10, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    // Should allocate both host and device memory
    if (v.data() == nullptr) {  // Host data
        std::cerr << check_msg(get_type_name<T>(), "check 2") << std::endl;
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_gpu_vector_size_value_constructor() {

    Vector<T> v(5, (T)3.5);

    if (!check((int)v.size(), 5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    for (int i = 0; i < 5; i++) {
        if (!check(v.data()[i], (T)3.5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_gpu_vector_copy_constructor() {

    Vector<T> v1({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> v2 = v1;  // Copy constructor

    if (!check((int)v2.size(), (int)v1.size(), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    // Verify data is copied
    T expected[] = {(T)1.0, (T)2.0, (T)3.0};
    if (!check(v2.data(), expected, 3, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_vector_move_constructor() {

    Vector<T> v1({(T)1.0, (T)2.0, (T)3.0});

    Vector<T> v2 = std::move(v1);  // Move constructor

    if (!check((int)v2.size(), 3, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    // Original should be empty after move
    if (!check((int)v1.size(), 0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// GPU Vector Memory Management Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_vector_host_device_memory() {

    Vector<T> v(10, (T)2.5);

    // Should have both host and device memory
    if (v.data() == nullptr) {  // Host pointer
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    }

    // Verify host data
    for (int i = 0; i < 10; i++) {
        if (!check(v.data()[i], (T)2.5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_gpu_vector_data_host_sync() {

    lahva::CudaRuntime runtime;
    Vector<T> v(5, (T)1.0);

    // Modify host data
    v.data()[0] = (T)5.0;

    // Access host data pointer
    T* host_data = v.data();
    if (host_data == nullptr) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    }

    if (!check(host_data[0], (T)5.0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// GPU Vector Attribute Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_vector_size_attribute() {

    Vector<T> v(25);

    if (!check((int)v.size(), 25, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_vector_data_access() {

    Vector<T> v({(T)5.0, (T)10.0, (T)15.0});

    T* data = v.data();
    if (data == nullptr) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    }

    if (!check(data[0], (T)5.0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_vector_type_float() {

    Vector<float> v(5, 3.14f);

    if (!check((int)v.size(), 5, check_msg(get_type_name<float>(), "check 1"))) return TEST_FAIL;

    for (int i = 0; i < 5; i++) {
        if (!check(v.data()[i], 3.14f, check_msg(get_type_name<float>(), "check 2"))) return TEST_FAIL;
    }

    return TEST_PASS;
}

int test_gpu_vector_type_int() {

    Vector<int> v({1, 2, 3, 4, 5});

    if (!check((int)v.size(), 5, check_msg(get_type_name<int>(), "check 1"))) return TEST_FAIL;

    int expected[] = {1, 2, 3, 4, 5};


    for (int i = 0; i < 5; i++) {
        if (!check(v.data()[i], expected[i], check_msg(get_type_name<int>(), "check 2"))) return TEST_FAIL;
    }

    return TEST_PASS;
}

// ============================================================================
// GPU Vector Assignment Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_vector_copy_assignment() {

    Vector<T> v1({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> v2;

    v2 = v1;  // Copy assignment

    if (!check((int)v2.size(), 3, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    T expected[] = {(T)1.0, (T)2.0, (T)3.0};
    if (!check(v2.data(), expected, 3, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_vector_move_assignment() {

    Vector<T> v1({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> v2;

    v2 = std::move(v1);

    if (!check((int)v2.size(), 3, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((int)v1.size(), 0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// GPU Vector Destructor Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_vector_destructor() {

    {
        Vector<T> v(100);
        if (v.data() == nullptr) {
            std::cerr << check_msg(get_type_name<T>(), "") << std::endl;
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

// ============================================================================
// Main
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

    if (total_failures > 0) {
        std::cerr << "gpu/tensor/vector tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All gpu/tensor/vector tests passed!" << std::endl;
    return TEST_PASS;
}
