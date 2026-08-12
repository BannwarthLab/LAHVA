#include "test_common.h"

using namespace lahva::gpu;
using lahva::Shape;
using lahva::CudaRuntime;

// ============================================================================
// GPU LowTriMatrix Constructor Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_lowtrimatrix_size_constructor() {
    CudaRuntime cudart;

    LowTriMatrix<T> m(5);

    if (!check((int)m.size(), 15, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    if (m.data() == nullptr) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_gpu_lowtrimatrix_size_value_constructor() {
    CudaRuntime cudart;

    LowTriMatrix<T> m(4, (T)3.5);

    if (!check((int)m.size(), 10, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    int packed_size = 4 * 5 / 2;

    for (int i = 0; i < packed_size; i++) {
        if (!check(m.data()[i], (T)3.5, check_msg(get_type_name<T>(), "check 2"))) {
            return TEST_FAIL;
            break;
        }
    }

    return TEST_PASS;
}

template <typename T>
int test_gpu_lowtrimatrix_copy_constructor() {
    CudaRuntime cudart;

    LowTriMatrix<T> m1(3, (T)2.5);
    LowTriMatrix<T> m2 = m1;  // Copy constructor

    if (!check((int)m2.size(), (int)m1.size(), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    int packed_size = 3 * 4 / 2;

    for (int i = 0; i < packed_size; i++) {
        if (!check(m2.data()[i], (T)2.5, check_msg(get_type_name<T>(), "check 2"))) {
            return TEST_FAIL;
            break;
        }
    }

    m1.data()[0] = (T)99.0;
    if (!check(m2.data()[0], (T)2.5, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_lowtrimatrix_move_constructor() {
    CudaRuntime cudart;

    LowTriMatrix<T> m1(3, (T)4.0);
    T* original_data = m1.data();

    LowTriMatrix<T> m2 = std::move(m1);  // Move constructor

    if (!check((int)m2.size(), 6, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    // Check if pointer was transferred
    if (m2.data() != original_data) {
        std::cerr << "[Fatal] GPU move constructor did NOT transfer pointer\n";
        std::cerr << "  Expected: " << (void*)original_data << "\n";
        std::cerr << "  Got:      " << (void*)m2.data() << "\n";
        return TEST_FAIL;
    }

    // Note: Move constructor doesn't clear source in this implementation
    // and may not transfer the pointer either - allocator determines behavior

    return TEST_PASS;
}

// ============================================================================
// GPU LowTriMatrix Attribute Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_lowtrimatrix_size_attribute() {
    CudaRuntime cudart;

    LowTriMatrix<T> m(7);

    if (!check((int)m.size(), 28, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_lowtrimatrix_data_access() {
    CudaRuntime cudart;

    LowTriMatrix<T> m(3);

    if (m.data() == nullptr) {
        return TEST_FAIL;
    }

    m.data()[0] = (T)5.5;
    if (!check(m.data()[0], (T)5.5, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// GPU LowTriMatrix Type Tests
// ============================================================================

int test_gpu_lowtrimatrix_float_type() {
    CudaRuntime cudart;

    LowTriMatrix<float> m(3, 2.5f);

    if (!check((int)m.size(), 6, check_msg(get_type_name<float>(), "check 1"))) return TEST_FAIL;

    if (!check(m.data()[0], 2.5f, check_msg(get_type_name<float>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_lowtrimatrix_int_type() {
    CudaRuntime cudart;

    LowTriMatrix<int> m(2);
    m.data()[0] = 10;
    m.data()[1] = 20;
    m.data()[2] = 30;

    if (!check((int)m.size(), 3, check_msg(get_type_name<int>(), "check 1"))) return TEST_FAIL;

    if (!check(m.data()[0], 10, check_msg(get_type_name<int>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// GPU LowTriMatrix Assignment Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_lowtrimatrix_copy_assignment() {
    CudaRuntime cudart;

    LowTriMatrix<T> m1(3, (T)2.5);
    LowTriMatrix<T> m2(2, (T)1.0);

    m2 = m1;  // Copy assignment

    if (!check((int)m2.size(), 6, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    // Self-assignment should be safe
    m1 = m1;
    if (!check((int)m1.size(), 6, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_lowtrimatrix_move_assignment() {
    CudaRuntime cudart;

    LowTriMatrix<T> m1(4, (T)2.0);
    LowTriMatrix<T> m2(2, (T)1.0);

    m2 = std::move(m1);  // Move assignment

    if (!check((int)m2.size(), 10, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    // Note: Move assignment doesn't clear source in this implementation

    return TEST_PASS;
}

// ============================================================================
// GPU LowTriMatrix Memory Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_lowtrimatrix_memory_allocation() {
    CudaRuntime cudart;

    LowTriMatrix<T> m(10);

    if (m.data() == nullptr) {
        return TEST_FAIL;
    }

    int packed_size = 10 * 11 / 2;

    m.data()[0] = (T)1.5;
    m.data()[packed_size - 1] = (T)55.5;

    if (!check(m.data()[0], (T)1.5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check(m.data()[packed_size - 1], (T)55.5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_lowtrimatrix_destructor() {
    CudaRuntime cudart;

    {
        LowTriMatrix<T> m(50);
        if (m.data() == nullptr) {
            return TEST_FAIL;
        }
    }
    // Destructor should deallocate

    return TEST_PASS;
}

// ============================================================================
// Extended GPU LowTriMatrix Tests
// ============================================================================

int test_gpu_lowtrimatrix_shape_constructor() {

    Shape s(5, 5);
    LowTriMatrix<double> m(s);

    if (!check((int)m.shape().first, 5, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m.shape().second, 5, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_lowtrimatrix_shape_value_constructor() {

    Shape s(4, 4);
    LowTriMatrix<double> m(s, 2.75);

    if (!check((int)m.size(), 10, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    for (int i = 0; i < 10; i++) {
        if (!check(m.data()[i], 2.75, check_msg(get_type_name<double>(), "check 2"))) {
            return TEST_FAIL;
            break;
        }
    }

    return TEST_PASS;
}

int test_gpu_lowtrimatrix_assign_after_construct() {

    // Test assignment after construction
    LowTriMatrix<double> m1(3, 1.5);
    LowTriMatrix<double> m2(2, 0.0);

    m2 = m1;

    if (!check((int)m2.size(), 6, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check(m2.data()[0], 1.5, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_lowtrimatrix_element_access() {

    LowTriMatrix<double> m(3, 0.0);

    // Set diagonal elements
    m(0, 0) = 1.5;
    m(1, 1) = 2.5;
    m(2, 2) = 3.5;

    // Set lower triangular elements
    m(1, 0) = 1.0;
    m(2, 0) = 2.0;
    m(2, 1) = 2.5;

    if (!check(m(0, 0), 1.5, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check(m(2, 1), 2.5, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_lowtrimatrix_const_element_access() {

    LowTriMatrix<double> m(3, 1.5);

    const LowTriMatrix<double>& const_m = m;

    if (!check(const_m(0, 0), 1.5, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check(const_m(2, 1), 1.5, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_lowtrimatrix_complex_double() {

    LowTriMatrix<complex_double> m(3, complex_double(1.0, 2.0));

    if (!check((int)m.size(), 6, check_msg(get_type_name<complex_double>(), "check 1"))) return TEST_FAIL;

    if (!check(m.data()[0].real(), 1.0, check_msg(get_type_name<complex_double>(), "check 2"))) return TEST_FAIL;

    if (!check(m.data()[0].imag(), 2.0, check_msg(get_type_name<complex_double>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_lowtrimatrix_gpu_operations() {
    CudaRuntime cudart;

    LowTriMatrix<double> m(4, 3.0);

    // Copy to device
    m.copy2device(cudart);
    cudart.synchronize();

    // Modify host data
    m.data()[0] = 1.0;

    // Copy back to host
    m.copy2host(cudart);
    cudart.synchronize();

    // Check that original device value was restored
    if (!check(m.data()[0], 3.0, check_msg(get_type_name<double>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_lowtrimatrix_large_matrix() {

    LowTriMatrix<double> m(50);

    int expected_size = 50 * 51 / 2;
    if (!check((int)m.size(), expected_size, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m.shape().first, 50, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_lowtrimatrix_float_element_access() {

    LowTriMatrix<float> m(3, 0.0f);

    m(0, 0) = 1.5f;
    m(1, 1) = 2.5f;
    m(2, 2) = 3.5f;

    if (!check(m(0, 0), 1.5f, check_msg(get_type_name<float>(), "check 1"))) return TEST_FAIL;

    if (!check(m(2, 2), 3.5f, check_msg(get_type_name<float>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_lowtrimatrix_full_lower_access() {

    LowTriMatrix<double> m(4, 0.0);

    // Fill entire lower triangular matrix
    for (int j = 0; j < 4; j++) {
        for (int i = j; i < 4; i++) {
            m(i, j) = i + j * 0.1;
        }
    }

    // Verify some elements
    if (!check(m(0, 0), 0.0, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check(m(2, 1), 2.1, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    if (!check(m(3, 2), 3.2, check_msg(get_type_name<double>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_lowtrimatrix_shape_accessors() {

    LowTriMatrix<double> m(6);

    Shape s = m.shape();
    if (!check((int)s.first, 6, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check((int)s.second, 6, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_lowtrimatrix_int_type_access() {

    LowTriMatrix<int> m(3, 0);

    m(0, 0) = 10;
    m(1, 0) = 20;
    m(1, 1) = 30;
    m(2, 0) = 40;
    m(2, 1) = 50;
    m(2, 2) = 60;

    if (!check(m(1, 0), 20, check_msg(get_type_name<int>(), "check 1"))) return TEST_FAIL;

    if (!check(m(2, 2), 60, check_msg(get_type_name<int>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_lowtrimatrix_const_access_full() {

    LowTriMatrix<double> m(3, 5.5);

    const LowTriMatrix<double>& const_m = m;

    if (!check(const_m(0, 0), 5.5, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check(const_m(2, 1), 5.5, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int total_failures = 0;

    // Constructors - test with double and float
    total_failures += test_gpu_lowtrimatrix_size_value_constructor<double>();
    total_failures += test_gpu_lowtrimatrix_size_value_constructor<float>();
    total_failures += test_gpu_lowtrimatrix_copy_constructor<double>();
    total_failures += test_gpu_lowtrimatrix_copy_constructor<float>();
    total_failures += test_gpu_lowtrimatrix_move_constructor<double>();
    total_failures += test_gpu_lowtrimatrix_move_constructor<float>();

    // Attributes
    total_failures += test_gpu_lowtrimatrix_size_attribute<double>();
    total_failures += test_gpu_lowtrimatrix_size_attribute<float>();
    total_failures += test_gpu_lowtrimatrix_data_access<double>();
    total_failures += test_gpu_lowtrimatrix_data_access<float>();

    // Types (type-specific tests)
    total_failures += test_gpu_lowtrimatrix_float_type();
    total_failures += test_gpu_lowtrimatrix_int_type();

    // Assignment
    total_failures += test_gpu_lowtrimatrix_copy_assignment<double>();
    total_failures += test_gpu_lowtrimatrix_copy_assignment<float>();
    total_failures += test_gpu_lowtrimatrix_move_assignment<double>();
    total_failures += test_gpu_lowtrimatrix_move_assignment<float>();

    // Memory
    total_failures += test_gpu_lowtrimatrix_memory_allocation<double>();
    total_failures += test_gpu_lowtrimatrix_memory_allocation<float>();
    total_failures += test_gpu_lowtrimatrix_destructor<double>();
    total_failures += test_gpu_lowtrimatrix_destructor<float>();

    // Extended Tests
    total_failures += test_gpu_lowtrimatrix_shape_constructor();
    total_failures += test_gpu_lowtrimatrix_shape_value_constructor();
    total_failures += test_gpu_lowtrimatrix_assign_after_construct();
    total_failures += test_gpu_lowtrimatrix_element_access();
    total_failures += test_gpu_lowtrimatrix_const_element_access();
    total_failures += test_gpu_lowtrimatrix_complex_double();
    total_failures += test_gpu_lowtrimatrix_gpu_operations();
    total_failures += test_gpu_lowtrimatrix_large_matrix();
    total_failures += test_gpu_lowtrimatrix_float_element_access();
    total_failures += test_gpu_lowtrimatrix_full_lower_access();
    total_failures += test_gpu_lowtrimatrix_shape_accessors();
    total_failures += test_gpu_lowtrimatrix_int_type_access();
    total_failures += test_gpu_lowtrimatrix_const_access_full();

    if (total_failures > 0) {
        std::cerr << "gpu/tensor/lowtrimatrix tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All gpu/tensor/lowtrimatrix tests passed!" << std::endl;
    return TEST_PASS;
}
