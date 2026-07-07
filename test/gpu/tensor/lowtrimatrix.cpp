#include "test_common.h"
#ifdef _CUDA

using namespace lahva::gpu;

// ============================================================================
// GPU LowTriMatrix Constructor Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_lowtrimatrix_size_constructor() {
    int failures = 0;
    CudaRuntime cudart;

    LowTriMatrix<T> m(5);

    if (!check((int)m.size(), 15, "GPU size constructor packed size n*(n+1)/2 = 5*6/2 = 15")) {
        failures += 1;
    }

    if (m.data() == nullptr) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_lowtrimatrix_size_value_constructor() {
    int failures = 0;
    CudaRuntime cudart;

    LowTriMatrix<T> m(4, (T)3.5);

    if (!check((int)m.size(), 10, "GPU size+value constructor packed size n*(n+1)/2 = 4*5/2 = 10")) {
        failures += 1;
    }

    int packed_size = 4 * 5 / 2;
    double tol = get_tolerance<T>();

    for (int i = 0; i < packed_size; i++) {
        if (!check(m.data()[i], (T)3.5, tol, "All elements should be initialized")) {
            failures += 1;
            break;
        }
    }

    return failures;
}

template <typename T>
int test_gpu_lowtrimatrix_copy_constructor() {
    int failures = 0;
    CudaRuntime cudart;

    LowTriMatrix<T> m1(3, (T)2.5);
    LowTriMatrix<T> m2 = m1;  // Copy constructor

    if (!check((int)m2.size(), (int)m1.size(), "GPU copy constructor should copy packed size")) {
        failures += 1;
    }

    int packed_size = 3 * 4 / 2;
    double tol = get_tolerance<T>();

    for (int i = 0; i < packed_size; i++) {
        if (!check(m2.data()[i], (T)2.5, tol, "Copy constructor should copy data")) {
            failures += 1;
            break;
        }
    }

    m1.data()[0] = (T)99.0;
    if (!check(m2.data()[0], (T)2.5, tol, "Copy should be independent")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_lowtrimatrix_move_constructor() {
    int failures = 0;
    CudaRuntime cudart;

    LowTriMatrix<T> m1(3, (T)4.0);
    T* original_data = m1.data();

    LowTriMatrix<T> m2 = std::move(m1);  // Move constructor

    if (!check((int)m2.size(), 6, "GPU move constructor packed size n*(n+1)/2 = 3*4/2 = 6")) {
        failures += 1;
    }

    // Check if pointer was transferred
    if (m2.data() != original_data) {
        std::cerr << "[Fatal] GPU move constructor did NOT transfer pointer\n";
        std::cerr << "  Expected: " << (void*)original_data << "\n";
        std::cerr << "  Got:      " << (void*)m2.data() << "\n";
        failures += 1;
    }

    // Note: Move constructor doesn't clear source in this implementation
    // and may not transfer the pointer either - allocator determines behavior

    return failures;
}

// ============================================================================
// GPU LowTriMatrix Attribute Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_lowtrimatrix_size_attribute() {
    int failures = 0;
    CudaRuntime cudart;

    LowTriMatrix<T> m(7);

    if (!check((int)m.size(), 28, "GPU size() returns packed size n*(n+1)/2 = 7*8/2 = 28")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_lowtrimatrix_data_access() {
    int failures = 0;
    CudaRuntime cudart;

    LowTriMatrix<T> m(3);

    if (m.data() == nullptr) {
        failures += 1;
    }

    m.data()[0] = (T)5.5;
    double tol = get_tolerance<T>();
    if (!check(m.data()[0], (T)5.5, tol, "Data access should work")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// GPU LowTriMatrix Type Tests
// ============================================================================

int test_gpu_lowtrimatrix_float_type() {
    int failures = 0;
    CudaRuntime cudart;

    LowTriMatrix<float> m(3, 2.5f);

    if (!check((int)m.size(), 6, "GPU float LowTriMatrix packed size n*(n+1)/2 = 3*4/2 = 6")) {
        failures += 1;
    }

    if (!check(m.data()[0], 2.5f, 1e-6f, "GPU float LowTriMatrix should store float values")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_lowtrimatrix_int_type() {
    int failures = 0;
    CudaRuntime cudart;

    LowTriMatrix<int> m(2);
    m.data()[0] = 10;
    m.data()[1] = 20;
    m.data()[2] = 30;

    if (!check((int)m.size(), 3, "GPU int LowTriMatrix packed size n*(n+1)/2 = 2*3/2 = 3")) {
        failures += 1;
    }

    if (!check(m.data()[0], 10, "GPU int LowTriMatrix should store int values")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// GPU LowTriMatrix Assignment Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_lowtrimatrix_copy_assignment() {
    int failures = 0;
    CudaRuntime cudart;

    LowTriMatrix<T> m1(3, (T)2.5);
    LowTriMatrix<T> m2(2, (T)1.0);

    m2 = m1;  // Copy assignment

    if (!check((int)m2.size(), 6, "GPU copy assignment packed size n*(n+1)/2 = 3*4/2 = 6")) {
        failures += 1;
    }

    // Self-assignment should be safe
    #if defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wself-assign-overloaded"
    #endif
    m1 = m1;
    #if defined(__clang__)
    #pragma GCC diagnostic pop
    #endif
    if (!check((int)m1.size(), 6, "GPU self-assignment packed size n*(n+1)/2 = 3*4/2 = 6")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_lowtrimatrix_move_assignment() {
    int failures = 0;
    CudaRuntime cudart;

    LowTriMatrix<T> m1(4, (T)2.0);
    LowTriMatrix<T> m2(2, (T)1.0);

    m2 = std::move(m1);  // Move assignment

    if (!check((int)m2.size(), 10, "GPU move assignment packed size n*(n+1)/2 = 4*5/2 = 10")) {
        failures += 1;
    }

    // Note: Move assignment doesn't clear source in this implementation

    return failures;
}

// ============================================================================
// GPU LowTriMatrix Memory Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_lowtrimatrix_memory_allocation() {
    int failures = 0;
    CudaRuntime cudart;

    LowTriMatrix<T> m(10);

    if (m.data() == nullptr) {
        failures += 1;
    }

    int packed_size = 10 * 11 / 2;
    double tol = get_tolerance<T>();

    m.data()[0] = (T)1.5;
    m.data()[packed_size - 1] = (T)55.5;

    if (!check(m.data()[0], (T)1.5, tol, "First element should be accessible")) {
        failures += 1;
    }

    if (!check(m.data()[packed_size - 1], (T)55.5, tol, "Last element should be accessible")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_lowtrimatrix_destructor() {
    int failures = 0;
    CudaRuntime cudart;

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
// Extended GPU LowTriMatrix Tests
// ============================================================================

int test_gpu_lowtrimatrix_shape_constructor() {
    int failures = 0;

    Shape s(5, 5);
    LowTriMatrix<double> m(s);

    if (!check((int)m.shape().first, 5, "Shape constructor rows")) {
        failures += 1;
    }

    if (!check((int)m.shape().second, 5, "Shape constructor cols")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_lowtrimatrix_shape_value_constructor() {
    int failures = 0;

    Shape s(4, 4);
    LowTriMatrix<double> m(s, 2.75);

    if (!check((int)m.size(), 10, "Shape+value constructor packed size")) {
        failures += 1;
    }

    double tol = get_tolerance<double>();
    for (int i = 0; i < 10; i++) {
        if (!check(m.data()[i], 2.75, tol, "Shape+value init")) {
            failures += 1;
            break;
        }
    }

    return failures;
}

int test_gpu_lowtrimatrix_assign_after_construct() {
    int failures = 0;

    // Test assignment after construction
    LowTriMatrix<double> m1(3, 1.5);
    LowTriMatrix<double> m2(2, 0.0);

    m2 = m1;

    if (!check((int)m2.size(), 6, "Assignment constructor size")) {
        failures += 1;
    }

    double tol = get_tolerance<double>();
    if (!check(m2.data()[0], 1.5, tol, "Assignment constructor data")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_lowtrimatrix_element_access() {
    int failures = 0;

    LowTriMatrix<double> m(3, 0.0);

    // Set diagonal elements
    m(0, 0) = 1.5;
    m(1, 1) = 2.5;
    m(2, 2) = 3.5;

    // Set lower triangular elements
    m(1, 0) = 1.0;
    m(2, 0) = 2.0;
    m(2, 1) = 2.5;

    double tol = get_tolerance<double>();
    if (!check(m(0, 0), 1.5, tol, "Element (0,0)")) {
        failures += 1;
    }

    if (!check(m(2, 1), 2.5, tol, "Element (2,1)")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_lowtrimatrix_const_element_access() {
    int failures = 0;

    LowTriMatrix<double> m(3, 1.5);

    const LowTriMatrix<double>& const_m = m;

    double tol = get_tolerance<double>();
    if (!check(const_m(0, 0), 1.5, tol, "Const element access")) {
        failures += 1;
    }

    if (!check(const_m(2, 1), 1.5, tol, "Const lower triangular access")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_lowtrimatrix_complex_double() {
    int failures = 0;

    LowTriMatrix<complex_double> m(3, complex_double(1.0, 2.0));

    if (!check((int)m.size(), 6, "Complex double size")) {
        failures += 1;
    }

    double tol = get_tolerance<double>();
    if (!check(m.data()[0].real(), 1.0, tol, "Complex double real part")) {
        failures += 1;
    }

    if (!check(m.data()[0].imag(), 2.0, tol, "Complex double imag part")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_lowtrimatrix_gpu_operations() {
    int failures = 0;
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
    double tol = get_tolerance<double>();
    if (!check(m.data()[0], 3.0, tol, "GPU copy operations")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_lowtrimatrix_large_matrix() {
    int failures = 0;

    LowTriMatrix<double> m(50);

    int expected_size = 50 * 51 / 2;
    if (!check((int)m.size(), expected_size, "Large matrix packed size")) {
        failures += 1;
    }

    if (!check((int)m.shape().first, 50, "Large matrix shape first")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_lowtrimatrix_float_element_access() {
    int failures = 0;

    LowTriMatrix<float> m(3, 0.0f);

    m(0, 0) = 1.5f;
    m(1, 1) = 2.5f;
    m(2, 2) = 3.5f;

    if (!check(m(0, 0), 1.5f, 1e-6f, "Float element (0,0)")) {
        failures += 1;
    }

    if (!check(m(2, 2), 3.5f, 1e-6f, "Float element (2,2)")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_lowtrimatrix_full_lower_access() {
    int failures = 0;

    LowTriMatrix<double> m(4, 0.0);

    // Fill entire lower triangular matrix
    for (int j = 0; j < 4; j++) {
        for (int i = j; i < 4; i++) {
            m(i, j) = i + j * 0.1;
        }
    }

    // Verify some elements
    double tol = get_tolerance<double>();
    if (!check(m(0, 0), 0.0, tol, "Lower access (0,0)")) {
        failures += 1;
    }

    if (!check(m(2, 1), 2.1, tol, "Lower access (2,1)")) {
        failures += 1;
    }

    if (!check(m(3, 2), 3.2, tol, "Lower access (3,2)")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_lowtrimatrix_shape_accessors() {
    int failures = 0;

    LowTriMatrix<double> m(6);

    Shape s = m.shape();
    if (!check((int)s.first, 6, "Shape first (rows)")) {
        failures += 1;
    }

    if (!check((int)s.second, 6, "Shape second (cols)")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_lowtrimatrix_int_type_access() {
    int failures = 0;

    LowTriMatrix<int> m(3, 0);

    m(0, 0) = 10;
    m(1, 0) = 20;
    m(1, 1) = 30;
    m(2, 0) = 40;
    m(2, 1) = 50;
    m(2, 2) = 60;

    if (!check(m(1, 0), 20, "Int access (1,0)")) {
        failures += 1;
    }

    if (!check(m(2, 2), 60, "Int access (2,2)")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_lowtrimatrix_const_access_full() {
    int failures = 0;

    LowTriMatrix<double> m(3, 5.5);

    const LowTriMatrix<double>& const_m = m;

    double tol = get_tolerance<double>();
    if (!check(const_m(0, 0), 5.5, tol, "Const access (0,0)")) {
        failures += 1;
    }

    if (!check(const_m(2, 1), 5.5, tol, "Const access (2,1)")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Main Test Runner
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

    if (total_failures == 0) {
        std::cout << "All GPU LowTriMatrix type tests passed!" << std::endl;
    } else {
        std::cout << "GPU LowTriMatrix type tests: " << total_failures << " failures" << std::endl;
    }

    return total_failures;
}

#endif // _CUDA
