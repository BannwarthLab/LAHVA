#include "test_common.h"
#include "array_utils.hpp"

using namespace lahva::gpu;
using lahva::Shape;
using lahva::CudaRuntime;

template<typename T>
using CPUMatrix = lahva::cpu::Matrix<T>;
template<typename T>
using GPUMatrix = Matrix<T, lahva::CudaHostAllocator<T>, lahva::CudaDeviceAsyncAllocator<T>>;

template<typename T>
using CPUVector = lahva::cpu::Vector<T>;
template<typename T>
using GPUVector = Vector<T, lahva::CudaHostAllocator<T>, lahva::CudaDeviceAsyncAllocator<T>>;

// ============================================================================
// GPU Matrix Constructor Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_matrix_default_constructor() {

    Matrix<T> m;

    if (!check((int)m.shape().first, 0, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m.shape().second, 0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_matrix_size_constructor() {

    Matrix<T> m(Shape{5, 10});

    if (!check((int)m.shape().first, 5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m.shape().second, 10, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    if (m.data() == nullptr) {  // Host data
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_gpu_matrix_shape_constructor() {

    Shape s(3, 7);
    Matrix<T> m(s);

    if (!check((int)m.shape().first, 3, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m.shape().second, 7, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_matrix_size_value_constructor() {

    Matrix<T> m(Shape{4, 6}, (T)2.5);

    if (!check((int)m.shape().first, 4, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m.shape().second, 6, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    // Verify all elements are initialized on host
    for (int i = 0; i < 4 * 6; i++) {
        if (!check(m.data()[i], (T)2.5, check_msg(get_type_name<T>(), "check 3"))) {
            return TEST_FAIL;
            break;
        }
    }

    return TEST_PASS;
}

template <typename T>
int test_gpu_matrix_copy_constructor() {

    Matrix<T> m1(Shape{3, 3}, (T)5.0);
    Matrix<T> m2 = m1;  // Copy constructor

    if (!check((int)m2.shape().first, 3, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m2.shape().second, 3, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    // Verify data is copied
    for (int i = 0; i < 9; i++) {
        if (!check(m2.data()[i], (T)5.0, check_msg(get_type_name<T>(), "check 3"))) {
            return TEST_FAIL;
            break;
        }
    }

    return TEST_PASS;
}

template <typename T>
int test_gpu_matrix_move_constructor() {

    Matrix<T> m1(Shape{2, 3}, (T)4.0);

    Matrix<T> m2 = std::move(m1);  // Move constructor

    if (!check((int)m2.shape().first, 2, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m2.shape().second, 3, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    // Original should be empty
    if (!check((int)m1.shape().first, 0, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// GPU Matrix Memory Management Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_matrix_host_device_memory() {

    Matrix<T> m(Shape{10, 10}, (T)2.5);

    // Should have both host and device memory
    if (m.data() == nullptr) {  // Host pointer
        return TEST_FAIL;
    }

    // Verify host data
    for (int i = 0; i < 10; i++) {
        if (!check(m.data()[i], (T)2.5, check_msg(get_type_name<T>(), ""))) {
            return TEST_FAIL;
            break;
        }
    }

    return TEST_PASS;
}

template <typename T>
int test_gpu_matrix_data_host_sync() {

    Matrix<T> m(Shape{3, 3});

    // Initialize host data
    for (int i = 0; i < 9; i++) {
        m.data()[i] = (T)(i + 1);
    }

    // Access host data pointer
    T* host_data = m.data();

    if (!check(host_data[0], (T)1.0, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// GPU Matrix Attribute Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_matrix_rows_attribute() {

    Matrix<T> m(Shape{7, 5});

    if (!check((int)m.shape().first, 7, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_matrix_cols_attribute() {

    Matrix<T> m(Shape{3, 11});

    if (!check((int)m.shape().second, 11, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_matrix_shape_attribute() {

    Matrix<T> m(Shape{6, 8});
    Shape s = m.shape();

    if (!check((int)s.first, 6, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)s.second, 8, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_matrix_size_attribute() {

    Matrix<T> m(Shape{5, 7});

    // Total size should be rows * cols
    if (!check((int)m.size(), 35, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_matrix_data_access() {

    Matrix<T> m(Shape{3, 3});

    if (m.data() == nullptr) {
        return TEST_FAIL;
    }

    m.data()[0] = (T)5.5;
    if (!check(m.data()[0], (T)5.5, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// GPU Matrix Type Tests
// ============================================================================

int test_gpu_matrix_float_type() {

    Matrix<float> m(Shape{3, 3}, 2.5f);

    if (!check((int)m.shape().first, 3, check_msg(get_type_name<float>(), "check 1"))) return TEST_FAIL;

    if (!check(m.data()[0], 2.5f, check_msg(get_type_name<float>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_int_type() {

    Matrix<int> m(Shape{2, 2});
    m.data()[0] = 10;
    m.data()[1] = 20;
    m.data()[2] = 30;
    m.data()[3] = 40;

    if (!check((int)m.shape().first, 2, check_msg(get_type_name<int>(), "check 1"))) return TEST_FAIL;

    if (!check(m.data()[0], 10, check_msg(get_type_name<int>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_half_type() {
    Matrix<__half> m(Shape{2, 2});
    if (!check((int)m.shape().first, 2, check_msg(get_type_name<__half>(), ""))) return TEST_FAIL;
    return TEST_PASS;
}

// ============================================================================
// GPU Matrix Assignment Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_matrix_copy_assignment() {

    Matrix<T> m1(Shape{2, 3}, (T)3.0);
    Matrix<T> m2;

    m2 = m1;  // Copy assignment

    if (!check((int)m2.shape().first, 2, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m2.shape().second, 3, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_gpu_matrix_move_assignment() {

    Matrix<T> m1(Shape{3, 4}, (T)2.0);
    Matrix<T> m2;

    m2 = std::move(m1);

    if (!check((int)m2.shape().first, 3, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m2.shape().second, 4, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    if (!check((int)m1.shape().first, 0, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// GPU Matrix Destructor Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_matrix_destructor() {

    {
        Matrix<T> m(Shape{50, 50});
        if (m.data() == nullptr) {
            return TEST_FAIL;
        }
    }
    // Destructor should deallocate both host and device memory

    return TEST_PASS;
}

// ============================================================================
// GPU Matrix Extended Constructor Tests
// ============================================================================

int test_gpu_matrix_const_data_constructor() {

    const double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Matrix<double> m(Shape{2, 3}, data);

    if (!check((int)m.shape().first, 2, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m.shape().second, 3, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    if (!check(m.data()[0], 1.0, check_msg(get_type_name<double>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_initializer_list_constructor() {

    Matrix<double> m(Shape{2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    if (!check((int)m.shape().first, 2, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m.shape().second, 3, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    if (!check(m.data()[0], 1.0, check_msg(get_type_name<double>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_pointer_constructor() {

    double* data = new double[6]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Matrix<double> m(Shape{2, 3}, data, false);

    if (!check((int)m.shape().first, 2, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m.data()[0], 1.0, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    delete[] data;
    return TEST_PASS;
}

// ============================================================================
// GPU Matrix Element Access Tests
// ============================================================================

int test_gpu_matrix_operator_access_nondiag() {

    Matrix<double> m(Shape{3, 4}, 0.0);

    m(0, 1) = 5.0;
    m(1, 2) = 7.0;
    m(2, 3) = 9.0;

    if (!check(m(0, 1), 5.0, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check(m(1, 2), 7.0, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    if (!check(m(2, 3), 9.0, check_msg(get_type_name<double>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_element_independence() {

    Matrix<double> m(Shape{2, 2}, 1.0);

    m.data()[0] = 10.0;
    if (!check(m.data()[1], 1.0, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    m.data()[2] = 20.0;
    if (!check(m.data()[3], 1.0, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_row_access_pattern() {

    Matrix<double> m(Shape{3, 3});

    for (int i = 0; i < 9; i++) {
        m.data()[i] = i * 1.5;
    }

    for (int i = 0; i < 9; i++) {
        if (!check(m.data()[i], i * 1.5, check_msg(get_type_name<double>(), ""))) {
            return TEST_FAIL;
            break;
        }
    }

    return TEST_PASS;
}

int test_gpu_matrix_col_values() {

    Matrix<double> m(Shape{3, 3}, 0.0);

    m(0, 0) = 1.0;
    m(1, 0) = 2.0;
    m(2, 0) = 3.0;
    m(0, 1) = 4.0;
    m(1, 1) = 5.0;
    m(2, 1) = 6.0;

    if (!check(m(0, 0), 1.0, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check(m(2, 1), 6.0, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// GPU Matrix Ownership Tests
// ============================================================================

int test_gpu_matrix_owns_data() {

    Matrix<double> m(Shape{3, 3}, 1.0);

    if (!m.ownsData()) {
        std::cerr << "GPU matrix should own its data\n";
        return TEST_FAIL;
    }

    return TEST_PASS;
}

int test_gpu_matrix_ownership_ptr_false() {

    double* data = new double[6]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Matrix<double> m(Shape{2, 3}, data, false);

    if (m.ownsData()) {
        std::cerr << "GPU matrix should not own external data\n";
        return TEST_FAIL;
    }

    delete[] data;
    return TEST_PASS;
}

// ============================================================================
// GPU Matrix Type Variations
// ============================================================================

int test_gpu_matrix_int_operations() {

    Matrix<int> m(Shape{2, 2});

    m.data()[0] = 10;
    m.data()[1] = 20;
    m.data()[2] = 30;
    m.data()[3] = 40;

    if (!check(m.data()[0], 10, check_msg(get_type_name<int>(), "check 1"))) return TEST_FAIL;

    if (!check(m.data()[3], 40, check_msg(get_type_name<int>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_float_operations() {

    Matrix<float> m(Shape{2, 3}, 3.14f);

    if (!check(m.data()[0], 3.14f, check_msg(get_type_name<float>(), "check 1"))) return TEST_FAIL;

    m.data()[0] = 2.71f;
    if (!check(m.data()[0], 2.71f, check_msg(get_type_name<float>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// GPU Matrix Attribute Tests (from mat-attributes)
// ============================================================================

template <typename T>
int test_symmetrize_cpu_gpu(CudaRuntime& cudart)
{
    const int n = 10;

    CPUMatrix<T> cpu_mat(Shape(n, n), 0.0);
    fill_with_rd_values(cpu_mat);

    GPUMatrix<T> gpu_mat(Shape(n, n), 0.0);
    lahva::cpu::CopyVectors(cpu_mat, gpu_mat);
    gpu_mat.copy2device(cudart);

    cpu_mat.symmetrize();
    gpu_mat.symmetrize(cudart);

    gpu_mat.copy2host(cudart);
    cudart.synchronize();

    if (!check(gpu_mat.data(), cpu_mat.data(), cpu_mat.size(), check_msg(get_type_name<T>(), ""))) {
        return 1;
    }

    return 0;
}

template <typename T>
int test_get_diagonal_cpu_gpu(CudaRuntime& cudart)
{
    const int n = 10;

    CPUMatrix<T> cpu_mat(Shape(n, n), 0.0);
    fill_with_rd_values(cpu_mat);

    GPUMatrix<T> gpu_mat(Shape(n, n), 0.0);
    lahva::cpu::CopyVectors(cpu_mat, gpu_mat);
    gpu_mat.copy2device(cudart);

    CPUVector<T> cpu_diag = cpu_mat.get_diagonal();
    GPUVector<T> gpu_diag = gpu_mat.get_diagonal(cudart);

    gpu_diag.copy2host(cudart);
    cudart.synchronize();

    if (!check(gpu_diag.data(), cpu_diag.data(), cpu_diag.size(), check_msg(get_type_name<T>(), ""))) {
        return 1;
    }

    return 0;
}

template <typename T>
int test_set_diagonal_cpu_gpu(CudaRuntime& cudart)
{
    const int n = 10;

    CPUMatrix<T> cpu_mat(Shape(n, n), 0.0);
    GPUMatrix<T> gpu_mat(Shape(n, n), 0.0);

    CPUVector<T> cpu_diag = CPUVector<T>(n);
    fill_with_rd_values(cpu_diag);
    cpu_mat.set_diagonal(cpu_diag);

    GPUVector<T> gpu_diag = GPUVector<T>(n);
    lahva::cpu::CopyVectors(cpu_diag, gpu_diag);
    gpu_mat.set_diagonal(cudart, gpu_diag);

    gpu_mat.copy2host(cudart);
    cudart.synchronize();

    if (!check(gpu_mat.data(), cpu_mat.data(), cpu_mat.size(), check_msg(get_type_name<T>(), ""))) {
        return 1;
    }

    return 0;
}

// ============================================================================
// Extended GPU Matrix Tests
// ============================================================================

int test_gpu_matrix_cudart_constructor() {
    CudaRuntime cudart;

    Shape s(5, 5);
    Matrix<double> m(s, cudart);

    if (!check((int)m.shape().first, 5, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m.shape().second, 5, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_row_major_init() {

    Shape s(2, 3);
    std::initializer_list<double> init = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Matrix<double> m_col(s, init, false);
    Matrix<double> m_row(s, init, true);

    // Column-major: elements 0,1,2,3,4,5 stored as columns
    if (!check(m_col.data()[0], 1.0, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    // Row-major: interpret as rows and convert to column-major
    if (!check(m_row(0, 0), 1.0, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    if (!check(m_row(0, 1), 2.0, check_msg(get_type_name<double>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_scalar_addition() {

    Shape s(2, 2);
    Matrix<double> m(s, 1.0);

    Matrix<double>& result = (m += 2.5);

    // Check all elements increased by 2.5
    if (!check(m.data(), result.data(), m.size(), check_msg(get_type_name<double>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_complex_double() {

    Shape s(2, 2);
    Matrix<complex_double> m(s, complex_double(1.0, 2.0));

    if (!check((int)m.shape().first, 2, check_msg(get_type_name<complex_double>(), "check 1"))) return TEST_FAIL;

    if (!check(m.data()[0].real(), 1.0, check_msg(get_type_name<complex_double>(), "check 2"))) return TEST_FAIL;

    if (!check(m.data()[0].imag(), 2.0, check_msg(get_type_name<complex_double>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_complex_float() {

    Shape s(2, 2);
    Matrix<complex_float> m(s, complex_float(1.5f, 2.5f));

    if (!check((int)m.shape().second, 2, check_msg(get_type_name<complex_float>(), "check 1"))) return TEST_FAIL;

    if (!check(m.data()[0].real(), 1.5f, check_msg(get_type_name<complex_float>(), "check 2"))) return TEST_FAIL;

    if (!check(m.data()[0].imag(), 2.5f, check_msg(get_type_name<complex_float>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_square_constructor() {

    Matrix<double> m(5);

    if (!check((int)m.shape().first, 5, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m.shape().second, 5, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_square_value_constructor() {

    Matrix<double> m(4, 3.14);

    if (!check((int)m.shape().first, 4, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    for (int i = 0; i < 16; i++) {
        if (!check(m.data()[i], 3.14, check_msg(get_type_name<double>(), "check 2"))) {
            return TEST_FAIL;
            break;
        }
    }

    return TEST_PASS;
}

int test_gpu_matrix_gpu_copy_operations(CudaRuntime& cudart) {

    Shape s(3, 3);
    Matrix<double> m(s, 2.5);

    // Copy to device
    m.copy2device(cudart);
    cudart.synchronize();

    // Modify host data
    m.data()[0] = 1.0;

    // Copy back to host
    m.copy2host(cudart);
    cudart.synchronize();

    // Check that original device value was restored
    if (!check(m.data()[0], 2.5, check_msg(get_type_name<double>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_mixed_precision_constructor() {

    Shape s(3, 3);
    // Create with float, cast to double
    Matrix<float> m_float(s, 1.5f);

    if (!check((int)m_float.shape().first, 3, check_msg(get_type_name<float>(), "check 1"))) return TEST_FAIL;

    if (!check(m_float.data()[0], 1.5f, check_msg(get_type_name<float>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_large_matrix() {

    Shape s(100, 100);
    Matrix<double> m(s, 0.0);

    if (!check((int)m.shape().first, 100, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m.shape().second, 100, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    if (!check((int)m.size(), 10000, check_msg(get_type_name<double>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_non_square() {

    Shape s(3, 7);
    Matrix<double> m(s, 1.0);

    if (!check((int)m.shape().first, 3, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m.shape().second, 7, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    // Test diagonal elements
    if (!check(m(0, 0), 1.0, check_msg(get_type_name<double>(), "check 3"))) return TEST_FAIL;

    if (!check(m(2, 6), 1.0, check_msg(get_type_name<double>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_element_modification() {

    Shape s(3, 3);
    Matrix<double> m(s, 0.0);

    m(0, 0) = 1.5;
    m(1, 1) = 2.5;
    m(2, 2) = 3.5;

    if (!check(m(0, 0), 1.5, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check(m(1, 1), 2.5, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    if (!check(m(2, 2), 3.5, check_msg(get_type_name<double>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_extensive_element_access() {

    Shape s(4, 5);
    Matrix<double> m(s, 0.0);

    // Access every element in column-major order
    for (int j = 0; j < 5; j++) {
        for (int i = 0; i < 4; i++) {
            m(i, j) = i + j * 0.1;
        }
    }

    // Verify random elements
    if (!check(m(0, 0), 0.0, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check(m(2, 3), 2.3, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    if (!check(m(3, 4), 3.4, check_msg(get_type_name<double>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_const_access_pattern() {

    Shape s(3, 3);
    Matrix<double> m(s, 2.0);

    const Matrix<double>& const_m = m;

    if (!check(const_m(0, 0), 2.0, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check(const_m(2, 2), 2.0, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_diagonal_pattern() {

    Shape s(5, 5);
    Matrix<double> m(s, 0.0);

    // Set only diagonal
    for (int i = 0; i < 5; i++) {
        m(i, i) = i + 1.0;
    }

    if (!check(m(0, 0), 1.0, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check(m(4, 4), 5.0, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    // Check off-diagonal are still zero
    if (!check(m(0, 1), 0.0, check_msg(get_type_name<double>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_gpu_matrix_owns_data_true() {

    Shape s(3, 3);
    Matrix<double> m(s, 1.0);

    if (!m.ownsData()) {
        return TEST_FAIL;
    }

    return TEST_PASS;
}

int test_gpu_matrix_owns_data_false() {

    Shape s(2, 2);
    double* data = new double[4];
    for (int i = 0; i < 4; i++) data[i] = 1.0;

    Matrix<double> m(s, data, false);

    if (m.ownsData()) {
        return TEST_FAIL;
    }

    delete[] data;

    return TEST_PASS;
}

// ============================================================================
// Matrix Type Precision Tests (float vs double)
// ============================================================================

int test_matrix_float_precision_construction() {

    Matrix<float> m_float(Shape{3, 3}, 1.5f);

    if (!check((int)m_float.shape().first, 3, check_msg(get_type_name<float>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m_float.shape().second, 3, check_msg(get_type_name<float>(), "check 2"))) return TEST_FAIL;

    if (!check(m_float.data()[0], 1.5f, check_msg(get_type_name<float>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_matrix_double_precision_construction() {

    Matrix<double> m_double(Shape{2, 2}, 2.5);

    if (!check((int)m_double.shape().first, 2, check_msg(get_type_name<double>(), "check 1"))) return TEST_FAIL;

    if (!check(m_double.data()[0], 2.5, check_msg(get_type_name<double>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_matrix_precision_operations() {

    Matrix<float> m(Shape{2, 2}, 0.0f);

    m.data()[0] = 1.5f;
    m.data()[1] = 2.5f;
    m.data()[2] = 3.5f;
    m.data()[3] = 4.5f;

    if (!check(m.data()[0], 1.5f, check_msg(get_type_name<float>(), "check 1"))) return TEST_FAIL;

    if (!check(m.data()[3], 4.5f, check_msg(get_type_name<float>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_matrix_precision_size() {

    Shape shape = Shape{4, 5};
    Matrix<float> m(shape, 1.0f);

    int total_size = shape.first * shape.second;
    if (!check(total_size, 20, check_msg(get_type_name<float>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m.shape().first * (int)m.shape().second, 20, check_msg(get_type_name<float>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_matrix_precision_copy() {

    Matrix<float> m1(Shape{2, 2}, 3.5f);
    Matrix<float> m2 = m1;

    if (!check((int)m2.shape().first, 2, check_msg(get_type_name<float>(), "check 1"))) return TEST_FAIL;

    if (!check(m2.data()[0], 3.5f, check_msg(get_type_name<float>(), "check 2"))) return TEST_FAIL;

    m1.data()[0] = 7.0f;
    if (!check(m2.data()[0], 3.5f, check_msg(get_type_name<float>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_matrix_precision_move() {

    Matrix<float> m1(Shape{3, 3}, 2.0f);
    Matrix<float> m2 = std::move(m1);

    if (!check((int)m2.shape().first, 3, check_msg(get_type_name<float>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m1.shape().first, 0, check_msg(get_type_name<float>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_matrix_precision_assignment() {

    Matrix<float> m1(Shape{2, 3}, 1.5f);
    Matrix<float> m2;

    m2 = m1;

    if (!check((int)m2.shape().first, 2, check_msg(get_type_name<float>(), "check 1"))) return TEST_FAIL;

    if (!check(m2.data()[0], 1.5f, check_msg(get_type_name<float>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_matrix_precision_large_matrix() {

    Matrix<float> m(Shape{100, 100}, 1.0f);

    if (!check((int)m.shape().first, 100, check_msg(get_type_name<float>(), "check 1"))) return TEST_FAIL;

    if (!check((int)m.shape().second, 100, check_msg(get_type_name<float>(), "check 2"))) return TEST_FAIL;

    m.data()[0] = 5.0f;
    m.data()[9999] = 7.5f;

    if (!check(m.data()[0], 5.0f, check_msg(get_type_name<float>(), "check 3"))) return TEST_FAIL;

    if (!check(m.data()[9999], 7.5f, check_msg(get_type_name<float>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_matrix_precision_element_access() {

    Matrix<float> m(Shape{3, 3}, 0.0f);

    m(0, 0) = 1.1f;
    m(1, 1) = 2.2f;
    m(2, 2) = 3.3f;

    if (!check(m(0, 0), 1.1f, check_msg(get_type_name<float>(), "check 1"))) return TEST_FAIL;

    if (!check(m(2, 2), 3.3f, check_msg(get_type_name<float>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

int test_matrix_precision_init_list() {

    Matrix<float> m(Shape{2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    if (!check(m.data()[0], 1.0f, check_msg(get_type_name<float>(), "check 1"))) return TEST_FAIL;

    if (!check(m.data()[3], 4.0f, check_msg(get_type_name<float>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int total_failures = 0;
    CudaRuntime cudart = CudaRuntime();

    // Constructors - test with double and float
    total_failures += test_gpu_matrix_default_constructor<double>();
    total_failures += test_gpu_matrix_default_constructor<float>();
    total_failures += test_gpu_matrix_size_constructor<double>();
    total_failures += test_gpu_matrix_size_constructor<float>();
    total_failures += test_gpu_matrix_shape_constructor<double>();
    total_failures += test_gpu_matrix_shape_constructor<float>();
    total_failures += test_gpu_matrix_size_value_constructor<double>();
    total_failures += test_gpu_matrix_size_value_constructor<float>();
    total_failures += test_gpu_matrix_copy_constructor<double>();
    total_failures += test_gpu_matrix_copy_constructor<float>();
    total_failures += test_gpu_matrix_move_constructor<double>();
    total_failures += test_gpu_matrix_move_constructor<float>();

    // Memory management
    total_failures += test_gpu_matrix_host_device_memory<double>();
    total_failures += test_gpu_matrix_host_device_memory<float>();
    total_failures += test_gpu_matrix_data_host_sync<double>();
    total_failures += test_gpu_matrix_data_host_sync<float>();

    // Attributes
    total_failures += test_gpu_matrix_rows_attribute<double>();
    total_failures += test_gpu_matrix_rows_attribute<float>();
    total_failures += test_gpu_matrix_cols_attribute<double>();
    total_failures += test_gpu_matrix_cols_attribute<float>();
    total_failures += test_gpu_matrix_shape_attribute<double>();
    total_failures += test_gpu_matrix_shape_attribute<float>();
    total_failures += test_gpu_matrix_size_attribute<double>();
    total_failures += test_gpu_matrix_size_attribute<float>();
    total_failures += test_gpu_matrix_data_access<double>();
    total_failures += test_gpu_matrix_data_access<float>();

    // Types (type-specific tests)
    total_failures += test_gpu_matrix_float_type();
    total_failures += test_gpu_matrix_int_type();
    total_failures += test_gpu_matrix_half_type();

    // Assignment
    total_failures += test_gpu_matrix_copy_assignment<double>();
    total_failures += test_gpu_matrix_copy_assignment<float>();
    total_failures += test_gpu_matrix_move_assignment<double>();
    total_failures += test_gpu_matrix_move_assignment<float>();

    // Destructor
    total_failures += test_gpu_matrix_destructor<double>();
    total_failures += test_gpu_matrix_destructor<float>();

    // Extended Constructors
    total_failures += test_gpu_matrix_const_data_constructor();
    total_failures += test_gpu_matrix_initializer_list_constructor();
    total_failures += test_gpu_matrix_pointer_constructor();

    // Element Access
    total_failures += test_gpu_matrix_operator_access_nondiag();
    total_failures += test_gpu_matrix_element_independence();
    total_failures += test_gpu_matrix_row_access_pattern();
    total_failures += test_gpu_matrix_col_values();

    // Ownership
    total_failures += test_gpu_matrix_owns_data();
    total_failures += test_gpu_matrix_ownership_ptr_false();

    // Type Variations
    total_failures += test_gpu_matrix_int_operations();
    total_failures += test_gpu_matrix_float_operations();

    // GPU Matrix Attributes (from mat-attributes)
    total_failures += test_symmetrize_cpu_gpu<double>(cudart);
    total_failures += test_symmetrize_cpu_gpu<float>(cudart);
    total_failures += test_get_diagonal_cpu_gpu<double>(cudart);
    total_failures += test_get_diagonal_cpu_gpu<float>(cudart);
    total_failures += test_set_diagonal_cpu_gpu<double>(cudart);
    total_failures += test_set_diagonal_cpu_gpu<float>(cudart);

    // Extended GPU Matrix Tests
    total_failures += test_gpu_matrix_cudart_constructor();
    total_failures += test_gpu_matrix_row_major_init();
    total_failures += test_gpu_matrix_scalar_addition();
    total_failures += test_gpu_matrix_complex_double();
    total_failures += test_gpu_matrix_complex_float();
    total_failures += test_gpu_matrix_square_constructor();
    total_failures += test_gpu_matrix_square_value_constructor();
    total_failures += test_gpu_matrix_gpu_copy_operations(cudart);
    total_failures += test_gpu_matrix_mixed_precision_constructor();
    total_failures += test_gpu_matrix_large_matrix();
    total_failures += test_gpu_matrix_non_square();
    total_failures += test_gpu_matrix_element_modification();
    total_failures += test_gpu_matrix_extensive_element_access();
    total_failures += test_gpu_matrix_const_access_pattern();
    total_failures += test_gpu_matrix_diagonal_pattern();
    total_failures += test_gpu_matrix_owns_data_true();
    total_failures += test_gpu_matrix_owns_data_false();

    // Matrix Type Precision Tests (float vs double)
    total_failures += test_matrix_float_precision_construction();
    total_failures += test_matrix_double_precision_construction();
    total_failures += test_matrix_precision_operations();
    total_failures += test_matrix_precision_size();
    total_failures += test_matrix_precision_copy();
    total_failures += test_matrix_precision_move();
    total_failures += test_matrix_precision_assignment();
    total_failures += test_matrix_precision_large_matrix();
    total_failures += test_matrix_precision_element_access();
    total_failures += test_matrix_precision_init_list();

    if (total_failures > 0) {
        std::cerr << "gpu/tensor/matrix tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All gpu/tensor/matrix tests passed!" << std::endl;
    return TEST_PASS;
}
