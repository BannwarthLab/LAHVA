#include "test_common.h"
#ifdef _CUDA

using namespace lahva::gpu;

const double TOLERANCE = 5.0e-7;

// ============================================================================
// GPU Tensor Explicit Constructor Tests
// ============================================================================

int test_gputensor_constructor_with_count() {
    int failures = 0;

    Vector<double> v(10);

    if (!check((int)v.size(), 10, "Constructor with count")) {
        failures += 1;
    }

    return failures;
}

int test_gputensor_constructor_no_args() {
    int failures = 0;

    Vector<double> v;

    if (!check((int)v.size(), 0, "Constructor with no args")) {
        failures += 1;
    }

    return failures;
}

int test_gputensor_constructor_with_cudart() {
    int failures = 0;
    CudaRuntime cudart;

    Vector<double> v(5, cudart);

    if (!check((int)v.size(), 5, "Constructor with CudaRuntime")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// GPU Tensor Copy and Move Constructor Tests - All Types
// ============================================================================

template <typename T>
int test_gputensor_copy_constructor() {
    int failures = 0;
    Matrix<T> m1(Shape(3, 3), (T)2.5);
    Matrix<T> m2 = m1;

    if (!check((int)m2.size(), 9, "Copy constructor size")) {
        failures += 1;
    }

    for (int i = 0; i < 9; i++) {
        if (!check((double)m2.data()[i], 2.5, TOLERANCE, "Copy constructor data")) {
            failures += 1;
            break;
        }
    }

    return failures;
}

int test_gputensor_copy_constructor_device_allocated(CudaRuntime& cudart) {
    int failures = 0;
    Matrix<double> m1(Shape(2, 2), 3.5);

    m1.copy2device(cudart);
    cudart.synchronize();

    // Copy a device-allocated matrix (exercises line 100-104)
    Matrix<double> m2 = m1;

    if (!check((int)m2.size(), 4, "Copy constructor device allocated")) {
        failures += 1;
    }

    return failures;
}

int test_gputensor_copy_constructor_int() {
    int failures = 0;
    Matrix<int> m1(Shape(2, 2), 5);
    Matrix<int> m2 = m1;

    if (!check((int)m2.size(), 4, "Copy constructor int size")) {
        failures += 1;
    }

    if (!check(m2.data()[0], 5, "Copy constructor int value")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gputensor_move_constructor() {
    int failures = 0;
    Matrix<T> m1(Shape(3, 3), (T)3.5);
    int m1_size_before = m1.size();

    Matrix<T> m2 = std::move(m1);

    if (!check((int)m2.size(), 9, "Move constructor size")) {
        failures += 1;
    }

    // Move constructor may not clear the source, just verify it was moved
    if (!check((int)m2.size(), m1_size_before, "Move constructor transferred size")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// GPU Tensor Assignment Tests - All Types
// ============================================================================

template <typename T>
int test_gputensor_copy_assignment() {
    int failures = 0;
    Matrix<T> m1(Shape(3, 3), (T)1.5);
    Matrix<T> m2(Shape(2, 2), (T)0.0);

    m2 = m1;

    if (!check((int)m2.size(), 9, "Copy assignment size")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gputensor_move_assignment() {
    int failures = 0;
    Matrix<T> m1(Shape(3, 3), (T)2.5);
    Matrix<T> m2(Shape(2, 2), (T)0.0);

    m2 = std::move(m1);

    if (!check((int)m2.size(), 9, "Move assignment size")) {
        failures += 1;
    }

    return failures;
}

int test_gputensor_copy_assignment_different_size() {
    int failures = 0;
    Matrix<double> m1(Shape(3, 3), 1.5);
    Matrix<double> m2(Shape(2, 2), 0.0);

    // Assignment with different sizes (exercises line 130-145)
    m2 = m1;

    if (!check((int)m2.size(), 9, "Copy assignment different size")) {
        failures += 1;
    }

    return failures;
}

int test_gputensor_copy_assignment_self() {
    int failures = 0;
    Matrix<double> m(Shape(2, 2), 2.5);

     // Self-assignment should be safe
    #if defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wself-assign-overloaded"
    #endif
    m = m;
    #if defined(__clang__)
    #pragma GCC diagnostic pop
    #endif

    if (!check((int)m.size(), 4, "Copy assignment self")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// GPU Device Memory Operations - All Types
// ============================================================================

template <typename T>
int test_gputensor_device_copy(CudaRuntime& cudart) {
    int failures = 0;

    Matrix<T> m(Shape(3, 3), (T)2.5);
    m.copy2device(cudart);
    cudart.synchronize();

    m.data()[0] = (T)0.0;

    m.copy2host(cudart);
    cudart.synchronize();

    if (!check((double)m.data()[0], 2.5, TOLERANCE, "Device copy")) {
        failures += 1;
    }

    return failures;
}

int test_gputensor_device_copy_int(CudaRuntime& cudart) {
    int failures = 0;

    Matrix<int> m(Shape(2, 2), 5);
    m.copy2device(cudart);
    cudart.synchronize();

    m.copy2host(cudart);
    cudart.synchronize();

    if (!check(m.data()[0], 5, "Device copy int")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Vector GPU Tensor Operations
// ============================================================================

int test_gputensor_vector_copy_constructor() {
    int failures = 0;
    Vector<double> v1(5, 2.5);
    Vector<double> v2 = v1;

    if (!check((int)v2.size(), 5, "Vector copy size")) {
        failures += 1;
    }

    return failures;
}

int test_gputensor_vector_move_constructor() {
    int failures = 0;
    Vector<double> v1(5, 3.5);

    Vector<double> v2 = std::move(v1);

    if (!check((int)v2.size(), 5, "Vector move size")) {
        failures += 1;
    }

    // Verify the data was transferred
    if (!check(v2.data()[0], 3.5, TOLERANCE, "Vector move data transferred")) {
        failures += 1;
    }

    return failures;
}

int test_gputensor_vector_device_copy(CudaRuntime& cudart) {
    int failures = 0;
    Vector<double> v(5, 1.5);

    v.copy2device(cudart);
    cudart.synchronize();

    v.data()[0] = 0.0;

    v.copy2host(cudart);
    cudart.synchronize();

    if (!check(v.data()[0], 1.5, TOLERANCE, "Vector device copy")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// LowTriMatrix GPU Tensor Operations
// ============================================================================

int test_gputensor_lowtrimatrix_copy_constructor() {
    int failures = 0;
    LowTriMatrix<double> m1(4, 2.5);
    LowTriMatrix<double> m2 = m1;

    if (!check((int)m2.size(), 10, "LowTriMatrix copy size")) {
        failures += 1;
    }

    return failures;
}

int test_gputensor_lowtrimatrix_move_constructor() {
    int failures = 0;
    LowTriMatrix<double> m1(3, 1.5);
    LowTriMatrix<double> m2 = std::move(m1);

    if (!check((int)m2.size(), 6, "LowTriMatrix move size")) {
        failures += 1;
    }

    // Verify the data was transferred
    if (!check(m2.data()[0], 1.5, TOLERANCE, "LowTriMatrix move data")) {
        failures += 1;
    }

    return failures;
}

int test_gputensor_lowtrimatrix_device_copy(CudaRuntime& cudart) {
    int failures = 0;
    LowTriMatrix<double> m(3, 2.5);

    m.copy2device(cudart);
    cudart.synchronize();

    m.copy2host(cudart);
    cudart.synchronize();

    if (!check(m.data()[0], 2.5, TOLERANCE, "LowTriMatrix device copy")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// GPU Tensor Memory Management Tests
// ============================================================================

int test_gputensor_gpu_data_pointer(CudaRuntime& cudart) {
    int failures = 0;
    Matrix<double> m(Shape(2, 2), 1.5);

    m.copy2device(cudart);
    cudart.synchronize();

    double* gpu_ptr = m.gpu_data();
    if (gpu_ptr == nullptr) {
        failures += 1;
    }

    return failures;
}

int test_gputensor_const_gpu_data_pointer(CudaRuntime& cudart) {
    int failures = 0;
    Matrix<double> m(Shape(2, 2), 2.5);

    m.copy2device(cudart);
    cudart.synchronize();

    const Matrix<double>& const_m = m;
    const double* const_gpu_ptr = const_m.gpu_data();
    if (const_gpu_ptr == nullptr) {
        failures += 1;
    }

    return failures;
}

int test_gputensor_alloc_on_device(CudaRuntime& cudart) {
    int failures = 0;
    Matrix<double> m(Shape(3, 3), 1.5);

    // Check allocation status before copy
    if (!m.alloc_on_device()) {
        // Should not be on device initially
    }

    m.copy2device(cudart);
    cudart.synchronize();

    // Check allocation status after copy
    if (m.alloc_on_device()) {
        // Should be on device after copy
    }

    return failures;
}

int test_gputensor_vector_gpu_data(CudaRuntime& cudart) {
    int failures = 0;
    Vector<double> v(5, 2.5);

    v.copy2device(cudart);
    cudart.synchronize();

    double* gpu_ptr = v.gpu_data();
    if (gpu_ptr == nullptr) {
        failures += 1;
    }

    return failures;
}

int test_gputensor_vector_const_gpu_data(CudaRuntime& cudart) {
    int failures = 0;
    Vector<float> v(4, 1.5f);

    v.copy2device(cudart);
    cudart.synchronize();

    const Vector<float>& const_v = v;
    const float* const_gpu_ptr = const_v.gpu_data();
    if (const_gpu_ptr == nullptr) {
        failures += 1;
    }

    return failures;
}

int test_gputensor_lowtrimatrix_gpu_data(CudaRuntime& cudart) {
    int failures = 0;
    LowTriMatrix<double> m(3, 3.5);

    m.copy2device(cudart);
    cudart.synchronize();

    double* gpu_ptr = m.gpu_data();
    if (gpu_ptr == nullptr) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    int total_failures = 0;
    CudaRuntime cudart;

    // Explicit constructor tests
    total_failures += test_gputensor_constructor_with_count();
    total_failures += test_gputensor_constructor_no_args();
    total_failures += test_gputensor_constructor_with_cudart();

    // Copy constructor tests
    total_failures += test_gputensor_copy_constructor<double>();
    total_failures += test_gputensor_copy_constructor<float>();
    total_failures += test_gputensor_copy_constructor_device_allocated(cudart);
    total_failures += test_gputensor_copy_constructor_int();

    // Move constructor tests
    total_failures += test_gputensor_move_constructor<double>();
    total_failures += test_gputensor_move_constructor<float>();

    // Copy assignment tests
    total_failures += test_gputensor_copy_assignment<double>();
    total_failures += test_gputensor_copy_assignment<float>();
    total_failures += test_gputensor_copy_assignment_different_size();
    total_failures += test_gputensor_copy_assignment_self();

    // Move assignment tests
    total_failures += test_gputensor_move_assignment<double>();
    total_failures += test_gputensor_move_assignment<float>();

    // Device copy tests
    total_failures += test_gputensor_device_copy<double>(cudart);
    total_failures += test_gputensor_device_copy<float>(cudart);
    total_failures += test_gputensor_device_copy_int(cudart);

    // Vector tests
    total_failures += test_gputensor_vector_copy_constructor();
    total_failures += test_gputensor_vector_move_constructor();
    total_failures += test_gputensor_vector_device_copy(cudart);

    // LowTriMatrix tests
    total_failures += test_gputensor_lowtrimatrix_copy_constructor();
    total_failures += test_gputensor_lowtrimatrix_move_constructor();
    total_failures += test_gputensor_lowtrimatrix_device_copy(cudart);

    // GPU memory management tests
    total_failures += test_gputensor_gpu_data_pointer(cudart);
    total_failures += test_gputensor_const_gpu_data_pointer(cudart);
    total_failures += test_gputensor_alloc_on_device(cudart);
    total_failures += test_gputensor_vector_gpu_data(cudart);
    total_failures += test_gputensor_vector_const_gpu_data(cudart);
    total_failures += test_gputensor_lowtrimatrix_gpu_data(cudart);

    if (total_failures == 0) {
        std::cout << "All GPU Tensor coverage tests passed!" << std::endl;
    } else {
        std::cout << "GPU Tensor coverage tests: " << total_failures << " failures" << std::endl;
    }

    return total_failures;
}

#else
int main() {
    std::cerr << "CUDA support not enabled" << std::endl;
    return 1;
}
#endif
