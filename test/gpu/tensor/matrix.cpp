#include "test_common.h"
#include "utils.hpp"
#include <random>
#ifdef _CUDA

using namespace lahva::gpu;

template<typename T>
using CPUMatrix = lahva::cpu::Matrix<T>;
template<typename T>
using GPUMatrix = Matrix<T, CudaHostAllocator<T>, CudaDeviceAsyncAllocator<T>>;

template<typename T>
using CPUVector = lahva::cpu::Vector<T>;
template<typename T>
using GPUVector = Vector<T, CudaHostAllocator<T>, CudaDeviceAsyncAllocator<T>>;

template <typename Tensor>
void fill_with_rd_values(Tensor& m)
{
    std::random_device rd;
    std::minstd_rand eng(rd());
    std::normal_distribution<> distr(0.0, 1.0e+3);
    for (size_t i = 0; i < m.size(); i++)
        m.data()[i] = distr(eng);
}

// ============================================================================
// GPU Matrix Constructor Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_matrix_default_constructor() {
    int failures = 0;

    Matrix<T> m;

    if (!check((int)m.shape().first, 0, "GPU default constructor should create empty matrix (rows)")) {
        failures += 1;
    }

    if (!check((int)m.shape().second, 0, "GPU default constructor should create empty matrix (cols)")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_matrix_size_constructor() {
    int failures = 0;

    Matrix<T> m(Shape{5, 10});

    if (!check((int)m.shape().first, 5, "GPU size constructor should set correct rows")) {
        failures += 1;
    }

    if (!check((int)m.shape().second, 10, "GPU size constructor should set correct cols")) {
        failures += 1;
    }

    if (m.data() == nullptr) {  // Host data
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_matrix_shape_constructor() {
    int failures = 0;

    Shape s(3, 7);
    Matrix<T> m(s);

    if (!check((int)m.shape().first, 3, "GPU shape constructor should set correct rows")) {
        failures += 1;
    }

    if (!check((int)m.shape().second, 7, "GPU shape constructor should set correct cols")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_matrix_size_value_constructor() {
    int failures = 0;

    Matrix<T> m(Shape{4, 6}, (T)2.5);

    if (!check((int)m.shape().first, 4, "GPU size+value constructor should set correct rows")) {
        failures += 1;
    }

    if (!check((int)m.shape().second, 6, "GPU size+value constructor should set correct cols")) {
        failures += 1;
    }

    // Verify all elements are initialized on host
    double tol = get_tolerance<T>();
    for (int i = 0; i < 4 * 6; i++) {
        if (!check(m.data()[i], (T)2.5, tol, "GPU elements should be initialized")) {
            failures += 1;
            break;
        }
    }

    return failures;
}

template <typename T>
int test_gpu_matrix_copy_constructor() {
    int failures = 0;

    Matrix<T> m1(Shape{3, 3}, (T)5.0);
    Matrix<T> m2 = m1;  // Copy constructor

    if (!check((int)m2.shape().first, 3, "GPU copy constructor should copy rows")) {
        failures += 1;
    }

    if (!check((int)m2.shape().second, 3, "GPU copy constructor should copy cols")) {
        failures += 1;
    }

    // Verify data is copied
    double tol = get_tolerance<T>();
    for (int i = 0; i < 9; i++) {
        if (!check(m2.data()[i], (T)5.0, tol, "GPU copy constructor should copy data")) {
            failures += 1;
            break;
        }
    }

    return failures;
}

template <typename T>
int test_gpu_matrix_move_constructor() {
    int failures = 0;

    Matrix<T> m1(Shape{2, 3}, (T)4.0);

    Matrix<T> m2 = std::move(m1);  // Move constructor

    if (!check((int)m2.shape().first, 2, "GPU move constructor should transfer rows")) {
        failures += 1;
    }

    if (!check((int)m2.shape().second, 3, "GPU move constructor should transfer cols")) {
        failures += 1;
    }

    // Original should be empty
    if (!check((int)m1.shape().first, 0, "GPU original should be empty after move")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// GPU Matrix Memory Management Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_matrix_host_device_memory() {
    int failures = 0;

    Matrix<T> m(Shape{10, 10}, (T)2.5);

    // Should have both host and device memory
    if (m.data() == nullptr) {  // Host pointer
        failures += 1;
    }

    // Verify host data
    double tol = get_tolerance<T>();
    for (int i = 0; i < 10; i++) {
        if (!check(m.data()[i], (T)2.5, tol, "GPU host memory should be initialized")) {
            failures += 1;
            break;
        }
    }

    return failures;
}

template <typename T>
int test_gpu_matrix_data_host_sync() {
    int failures = 0;

    Matrix<T> m(Shape{3, 3});

    // Initialize host data
    double tol = get_tolerance<T>();
    for (int i = 0; i < 9; i++) {
        m.data()[i] = (T)(i + 1);
    }

    // Access host data pointer
    T* host_data = m.data();
    if (host_data == nullptr) {
        failures += 1;
    }

    if (!check(host_data[0], (T)1.0, tol, "data() should return host data")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// GPU Matrix Attribute Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_matrix_rows_attribute() {
    int failures = 0;

    Matrix<T> m(Shape{7, 5});

    if (!check((int)m.shape().first, 7, "GPU rows() should return correct value")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_matrix_cols_attribute() {
    int failures = 0;

    Matrix<T> m(Shape{3, 11});

    if (!check((int)m.shape().second, 11, "GPU cols() should return correct value")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_matrix_shape_attribute() {
    int failures = 0;

    Matrix<T> m(Shape{6, 8});
    Shape s = m.shape();

    if (!check((int)s.first, 6, "GPU shape().rows should match")) {
        failures += 1;
    }

    if (!check((int)s.second, 8, "GPU shape().cols should match")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_matrix_size_attribute() {
    int failures = 0;

    Matrix<T> m(Shape{5, 7});

    // Total size should be rows * cols
    if (!check((int)m.size(), 35, "GPU size() should return rows*cols")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_matrix_data_access() {
    int failures = 0;

    Matrix<T> m(Shape{3, 3});

    if (m.data() == nullptr) {
        failures += 1;
    }

    m.data()[0] = (T)5.5;
    double tol = get_tolerance<T>();
    if (!check(m.data()[0], (T)5.5, tol, "GPU data access should work")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// GPU Matrix Type Tests
// ============================================================================

int test_gpu_matrix_float_type() {
    int failures = 0;

    Matrix<float> m(Shape{3, 3}, 2.5f);

    if (!check((int)m.shape().first, 3, "GPU float matrix should have correct rows")) {
        failures += 1;
    }

    if (!check(m.data()[0], 2.5f, 1e-6f, "GPU float matrix should store float values")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_int_type() {
    int failures = 0;

    Matrix<int> m(Shape{2, 2});
    m.data()[0] = 10;
    m.data()[1] = 20;
    m.data()[2] = 30;
    m.data()[3] = 40;

    if (!check((int)m.shape().first, 2, "GPU int matrix should have correct rows")) {
        failures += 1;
    }

    if (!check(m.data()[0], 10, "GPU int matrix should store int values")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_half_type() {
    int failures = 0;
    Matrix<__half> m(Shape{2, 2});
    if (!check((int)m.shape().first, 2, "GPU half matrix should have correct rows")) {
        failures += 1;
    }
    return failures;
}

// ============================================================================
// GPU Matrix Assignment Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_matrix_copy_assignment() {
    int failures = 0;

    Matrix<T> m1(Shape{2, 3}, (T)3.0);
    Matrix<T> m2;

    m2 = m1;  // Copy assignment

    if (!check((int)m2.shape().first, 2, "GPU copy assignment should copy rows")) {
        failures += 1;
    }

    if (!check((int)m2.shape().second, 3, "GPU copy assignment should copy cols")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_matrix_move_assignment() {
    int failures = 0;

    Matrix<T> m1(Shape{3, 4}, (T)2.0);
    Matrix<T> m2;

    m2 = std::move(m1);

    if (!check((int)m2.shape().first, 3, "GPU move assignment should transfer rows")) {
        failures += 1;
    }

    if (!check((int)m2.shape().second, 4, "GPU move assignment should transfer cols")) {
        failures += 1;
    }

    if (!check((int)m1.shape().first, 0, "GPU original should be empty after move assignment")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// GPU Matrix Destructor Tests (Templated)
// ============================================================================

template <typename T>
int test_gpu_matrix_destructor() {
    int failures = 0;

    {
        Matrix<T> m(Shape{50, 50});
        if (m.data() == nullptr) {
            failures += 1;
        }
    }
    // Destructor should deallocate both host and device memory

    return failures;
}

// ============================================================================
// GPU Matrix Extended Constructor Tests
// ============================================================================

int test_gpu_matrix_const_data_constructor() {
    int failures = 0;

    const double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Matrix<double> m(Shape{2, 3}, data);

    if (!check((int)m.shape().first, 2, "Const data constructor should set rows")) {
        failures += 1;
    }

    if (!check((int)m.shape().second, 3, "Const data constructor should set cols")) {
        failures += 1;
    }

    double tol = get_tolerance<double>();
    if (!check(m.data()[0], 1.0, tol, "Const data should be copied")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_initializer_list_constructor() {
    int failures = 0;

    Matrix<double> m(Shape{2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    if (!check((int)m.shape().first, 2, "Initializer list constructor should set rows")) {
        failures += 1;
    }

    if (!check((int)m.shape().second, 3, "Initializer list constructor should set cols")) {
        failures += 1;
    }

    double tol = get_tolerance<double>();
    if (!check(m.data()[0], 1.0, tol, "Initializer list data should be set")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_pointer_constructor() {
    int failures = 0;

    double* data = new double[6]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Matrix<double> m(Shape{2, 3}, data, false);

    if (!check((int)m.shape().first, 2, "Pointer constructor should set rows")) {
        failures += 1;
    }

    double tol = get_tolerance<double>();
    if (!check((int)m.data()[0], 1.0, tol, "Pointer constructor should use provided data")) {
        failures += 1;
    }

    delete[] data;
    return failures;
}

// ============================================================================
// GPU Matrix Element Access Tests
// ============================================================================

int test_gpu_matrix_operator_access_nondiag() {
    int failures = 0;

    Matrix<double> m(Shape{3, 4}, 0.0);

    m(0, 1) = 5.0;
    m(1, 2) = 7.0;
    m(2, 3) = 9.0;

    double tol = get_tolerance<double>();
    if (!check(m(0, 1), 5.0, tol, "Operator() [0,1] should be accessible")) {
        failures += 1;
    }

    if (!check(m(1, 2), 7.0, tol, "Operator() [1,2] should be accessible")) {
        failures += 1;
    }

    if (!check(m(2, 3), 9.0, tol, "Operator() [2,3] should be accessible")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_element_independence() {
    int failures = 0;

    Matrix<double> m(Shape{2, 2}, 1.0);

    m.data()[0] = 10.0;
    double tol = get_tolerance<double>();
    if (!check(m.data()[1], 1.0, tol, "Other elements should not change")) {
        failures += 1;
    }

    m.data()[2] = 20.0;
    if (!check(m.data()[3], 1.0, tol, "Other elements should not change")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_row_access_pattern() {
    int failures = 0;

    Matrix<double> m(Shape{3, 3});

    for (int i = 0; i < 9; i++) {
        m.data()[i] = i * 1.5;
    }

    double tol = get_tolerance<double>();
    for (int i = 0; i < 9; i++) {
        if (!check(m.data()[i], i * 1.5, tol, "Row-major access should work")) {
            failures += 1;
            break;
        }
    }

    return failures;
}

int test_gpu_matrix_col_values() {
    int failures = 0;

    Matrix<double> m(Shape{3, 3}, 0.0);

    m(0, 0) = 1.0;
    m(1, 0) = 2.0;
    m(2, 0) = 3.0;
    m(0, 1) = 4.0;
    m(1, 1) = 5.0;
    m(2, 1) = 6.0;

    double tol = get_tolerance<double>();
    if (!check(m(0, 0), 1.0, tol, "Column-based access [0,0]")) {
        failures += 1;
    }

    if (!check(m(2, 1), 6.0, tol, "Column-based access [2,1]")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// GPU Matrix Ownership Tests
// ============================================================================

int test_gpu_matrix_owns_data() {
    int failures = 0;

    Matrix<double> m(Shape{3, 3}, 1.0);

    if (!m.ownsData()) {
        std::cerr << "GPU matrix should own its data\n";
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_ownership_ptr_false() {
    int failures = 0;

    double* data = new double[6]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Matrix<double> m(Shape{2, 3}, data, false);

    if (m.ownsData()) {
        std::cerr << "GPU matrix should not own external data\n";
        failures += 1;
    }

    delete[] data;
    return failures;
}

// ============================================================================
// GPU Matrix Type Variations
// ============================================================================

int test_gpu_matrix_int_operations() {
    int failures = 0;

    Matrix<int> m(Shape{2, 2});

    m.data()[0] = 10;
    m.data()[1] = 20;
    m.data()[2] = 30;
    m.data()[3] = 40;

    if (!check(m.data()[0], 10, "Int matrix element [0]")) {
        failures += 1;
    }

    if (!check(m.data()[3], 40, "Int matrix element [3]")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_float_operations() {
    int failures = 0;

    Matrix<float> m(Shape{2, 3}, 3.14f);

    if (!check(m.data()[0], 3.14f, 1e-6f, "Float matrix initialization")) {
        failures += 1;
    }

    m.data()[0] = 2.71f;
    if (!check(m.data()[0], 2.71f, 1e-6f, "Float matrix modification")) {
        failures += 1;
    }

    return failures;
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
    cpu::CopyVectors(cpu_mat, gpu_mat);
    gpu_mat.copy2device(cudart);

    cpu_mat.symmetrize();
    gpu_mat.symmetrize(cudart);

    gpu_mat.copy2host(cudart);
    cudart.synchronize();

    T eps = std::abs(std::nextafter(cpu_mat.data()[0], +INFINITY) - cpu_mat.data()[0]);
    if (!check(gpu_mat.data(), cpu_mat.data(), eps, cpu_mat.size(), "Symmetrize")) {
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
    cpu::CopyVectors(cpu_mat, gpu_mat);
    gpu_mat.copy2device(cudart);

    CPUVector<T> cpu_diag = cpu_mat.get_diagonal();
    GPUVector<T> gpu_diag = gpu_mat.get_diagonal(cudart);

    gpu_diag.copy2host(cudart);
    cudart.synchronize();

    T eps = std::abs(std::nextafter(cpu_diag.data()[0], +INFINITY) - cpu_diag.data()[0]);
    if (!check(gpu_diag.data(), cpu_diag.data(), eps, cpu_diag.size(), "Get Diagonal")) {
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
    cpu::CopyVectors(cpu_diag, gpu_diag);
    gpu_mat.set_diagonal(cudart, gpu_diag);

    gpu_mat.copy2host(cudart);
    cudart.synchronize();

    T eps = std::abs(std::nextafter(cpu_mat.data()[0], +INFINITY) - cpu_mat.data()[0]);
    if (!check(gpu_mat.data(), cpu_mat.data(), eps, cpu_mat.size(), "Set Diagonal")) {
        return 1;
    }

    return 0;
}

// ============================================================================
// Extended GPU Matrix Tests
// ============================================================================

int test_gpu_matrix_cudart_constructor() {
    int failures = 0;
    CudaRuntime cudart;

    Shape s(5, 5);
    Matrix<double> m(s, cudart);

    if (!check((int)m.shape().first, 5, "CudaRuntime constructor rows")) {
        failures += 1;
    }

    if (!check((int)m.shape().second, 5, "CudaRuntime constructor cols")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_row_major_init() {
    int failures = 0;

    Shape s(2, 3);
    std::initializer_list<double> init = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Matrix<double> m_col(s, init, false);
    Matrix<double> m_row(s, init, true);

    // Column-major: elements 0,1,2,3,4,5 stored as columns
    double tol = get_tolerance<double>();
    if (!check(m_col.data()[0], 1.0, tol, "Column-major init [0]")) {
        failures += 1;
    }

    // Row-major: interpret as rows and convert to column-major
    if (!check(m_row(0, 0), 1.0, tol, "Row-major init (0,0)")) {
        failures += 1;
    }

    if (!check(m_row(0, 1), 2.0, tol, "Row-major init (0,1)")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_scalar_addition() {
    int failures = 0;

    Shape s(2, 2);
    Matrix<double> m(s, 1.0);

    Matrix<double>& result = (m += 2.5);

    // Check all elements increased by 2.5
    double tol = get_tolerance<double>();
    for (int i = 0; i < 4; i++) {
        if (!check(m.data()[i], 3.5, tol, "Scalar addition")) {
            failures += 1;
            break;
        }
    }

    // Check that operator+= returns reference to same object
    if (&result != &m) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_complex_double() {
    int failures = 0;

    Shape s(2, 2);
    Matrix<complex_double> m(s, complex_double(1.0, 2.0));

    if (!check((int)m.shape().first, 2, "Complex double rows")) {
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

int test_gpu_matrix_complex_float() {
    int failures = 0;

    Shape s(2, 2);
    Matrix<complex_float> m(s, complex_float(1.5f, 2.5f));

    if (!check((int)m.shape().second, 2, "Complex float cols")) {
        failures += 1;
    }

    double tol = get_tolerance<double>();
    if (!check(m.data()[0].real(), 1.5f, tol, "Complex float real part")) {
        failures += 1;
    }

    if (!check(m.data()[0].imag(), 2.5f, tol, "Complex float imag part")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_square_constructor() {
    int failures = 0;

    Matrix<double> m(5);

    if (!check((int)m.shape().first, 5, "Square constructor rows")) {
        failures += 1;
    }

    if (!check((int)m.shape().second, 5, "Square constructor cols")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_square_value_constructor() {
    int failures = 0;

    Matrix<double> m(4, 3.14);

    if (!check((int)m.shape().first, 4, "Square value constructor rows")) {
        failures += 1;
    }

    double tol = get_tolerance<double>();
    for (int i = 0; i < 16; i++) {
        if (!check(m.data()[i], 3.14, tol, "Square value constructor init")) {
            failures += 1;
            break;
        }
    }

    return failures;
}

int test_gpu_matrix_gpu_copy_operations(CudaRuntime& cudart) {
    int failures = 0;

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
    double tol = get_tolerance<double>();
    if (!check(m.data()[0], 2.5, tol, "GPU copy to/from device")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_mixed_precision_constructor() {
    int failures = 0;

    Shape s(3, 3);
    // Create with float, cast to double
    Matrix<float> m_float(s, 1.5f);

    if (!check((int)m_float.shape().first, 3, "Mixed precision rows")) {
        failures += 1;
    }

    double tol = get_tolerance<float>();
    if (!check(m_float.data()[0], 1.5f, tol, "Mixed precision value")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_large_matrix() {
    int failures = 0;

    Shape s(100, 100);
    Matrix<double> m(s, 0.0);

    if (!check((int)m.shape().first, 100, "Large matrix rows")) {
        failures += 1;
    }

    if (!check((int)m.shape().second, 100, "Large matrix cols")) {
        failures += 1;
    }

    if (!check((int)m.size(), 10000, "Large matrix size")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_non_square() {
    int failures = 0;

    Shape s(3, 7);
    Matrix<double> m(s, 1.0);

    if (!check((int)m.shape().first, 3, "Non-square matrix rows")) {
        failures += 1;
    }

    if (!check((int)m.shape().second, 7, "Non-square matrix cols")) {
        failures += 1;
    }

    // Test diagonal elements
    double tol = get_tolerance<double>();
    if (!check(m(0, 0), 1.0, tol, "Non-square diagonal (0,0)")) {
        failures += 1;
    }

    if (!check(m(2, 6), 1.0, tol, "Non-square element (2,6)")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_element_modification() {
    int failures = 0;

    Shape s(3, 3);
    Matrix<double> m(s, 0.0);

    m(0, 0) = 1.5;
    m(1, 1) = 2.5;
    m(2, 2) = 3.5;

    double tol = get_tolerance<double>();
    if (!check(m(0, 0), 1.5, tol, "Element (0,0)")) {
        failures += 1;
    }

    if (!check(m(1, 1), 2.5, tol, "Element (1,1)")) {
        failures += 1;
    }

    if (!check(m(2, 2), 3.5, tol, "Element (2,2)")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_extensive_element_access() {
    int failures = 0;

    Shape s(4, 5);
    Matrix<double> m(s, 0.0);

    // Access every element in column-major order
    for (int j = 0; j < 5; j++) {
        for (int i = 0; i < 4; i++) {
            m(i, j) = i + j * 0.1;
        }
    }

    // Verify random elements
    double tol = get_tolerance<double>();
    if (!check(m(0, 0), 0.0, tol, "Element (0,0)")) {
        failures += 1;
    }

    if (!check(m(2, 3), 2.3, tol, "Element (2,3)")) {
        failures += 1;
    }

    if (!check(m(3, 4), 3.4, tol, "Element (3,4)")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_const_access_pattern() {
    int failures = 0;

    Shape s(3, 3);
    Matrix<double> m(s, 2.0);

    const Matrix<double>& const_m = m;

    double tol = get_tolerance<double>();
    if (!check(const_m(0, 0), 2.0, tol, "Const access (0,0)")) {
        failures += 1;
    }

    if (!check(const_m(2, 2), 2.0, tol, "Const access (2,2)")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_diagonal_pattern() {
    int failures = 0;

    Shape s(5, 5);
    Matrix<double> m(s, 0.0);

    // Set only diagonal
    for (int i = 0; i < 5; i++) {
        m(i, i) = i + 1.0;
    }

    double tol = get_tolerance<double>();
    if (!check(m(0, 0), 1.0, tol, "Diagonal (0,0)")) {
        failures += 1;
    }

    if (!check(m(4, 4), 5.0, tol, "Diagonal (4,4)")) {
        failures += 1;
    }

    // Check off-diagonal are still zero
    if (!check(m(0, 1), 0.0, tol, "Off-diagonal (0,1)")) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_owns_data_true() {
    int failures = 0;

    Shape s(3, 3);
    Matrix<double> m(s, 1.0);

    if (!m.ownsData()) {
        failures += 1;
    }

    return failures;
}

int test_gpu_matrix_owns_data_false() {
    int failures = 0;

    Shape s(2, 2);
    double* data = new double[4];
    for (int i = 0; i < 4; i++) data[i] = 1.0;

    Matrix<double> m(s, data, false);

    if (m.ownsData()) {
        failures += 1;
    }

    delete[] data;

    return failures;
}

// ============================================================================
// Matrix Type Precision Tests (float vs double)
// ============================================================================

int test_matrix_float_precision_construction() {
    int failures = 0;

    Matrix<float> m_float(Shape{3, 3}, 1.5f);

    if (!check((int)m_float.shape().first, 3, "Float matrix rows")) {
        failures += 1;
    }

    if (!check((int)m_float.shape().second, 3, "Float matrix cols")) {
        failures += 1;
    }

    if (!check(m_float.data()[0], 1.5f, 1e-6f, "Float matrix data")) {
        failures += 1;
    }

    return failures;
}

int test_matrix_double_precision_construction() {
    int failures = 0;

    Matrix<double> m_double(Shape{2, 2}, 2.5);

    if (!check((int)m_double.shape().first, 2, "Double matrix rows")) {
        failures += 1;
    }

    double tol = get_tolerance<double>();
    if (!check(m_double.data()[0], 2.5, tol, "Double matrix data")) {
        failures += 1;
    }

    return failures;
}

int test_matrix_precision_operations() {
    int failures = 0;

    Matrix<float> m(Shape{2, 2}, 0.0f);

    m.data()[0] = 1.5f;
    m.data()[1] = 2.5f;
    m.data()[2] = 3.5f;
    m.data()[3] = 4.5f;

    if (!check(m.data()[0], 1.5f, 1e-6f, "Precision operations [0]")) {
        failures += 1;
    }

    if (!check(m.data()[3], 4.5f, 1e-6f, "Precision operations [3]")) {
        failures += 1;
    }

    return failures;
}

int test_matrix_precision_size() {
    int failures = 0;

    Shape shape = Shape{4, 5};
    Matrix<float> m(shape, 1.0f);

    int total_size = shape.first * shape.second;
    if (!check(total_size, 20, "Precision matrix total size")) {
        failures += 1;
    }

    if (!check((int)m.shape().first * (int)m.shape().second, 20, "Precision size calculation")) {
        failures += 1;
    }

    return failures;
}

int test_matrix_precision_copy() {
    int failures = 0;

    Matrix<float> m1(Shape{2, 2}, 3.5f);
    Matrix<float> m2 = m1;

    if (!check((int)m2.shape().first, 2, "Precision copy constructor rows")) {
        failures += 1;
    }

    if (!check(m2.data()[0], 3.5f, 1e-6f, "Precision copy constructor data")) {
        failures += 1;
    }

    m1.data()[0] = 7.0f;
    if (!check(m2.data()[0], 3.5f, 1e-6f, "Precision copy independence")) {
        failures += 1;
    }

    return failures;
}

int test_matrix_precision_move() {
    int failures = 0;

    Matrix<float> m1(Shape{3, 3}, 2.0f);
    Matrix<float> m2 = std::move(m1);

    if (!check((int)m2.shape().first, 3, "Precision move rows")) {
        failures += 1;
    }

    if (!check((int)m1.shape().first, 0, "Precision move source emptied")) {
        failures += 1;
    }

    return failures;
}

int test_matrix_precision_assignment() {
    int failures = 0;

    Matrix<float> m1(Shape{2, 3}, 1.5f);
    Matrix<float> m2;

    m2 = m1;

    if (!check((int)m2.shape().first, 2, "Precision assignment rows")) {
        failures += 1;
    }

    if (!check(m2.data()[0], 1.5f, 1e-6f, "Precision assignment data")) {
        failures += 1;
    }

    return failures;
}

int test_matrix_precision_large_matrix() {
    int failures = 0;

    Matrix<float> m(Shape{100, 100}, 1.0f);

    if (!check((int)m.shape().first, 100, "Precision large matrix rows")) {
        failures += 1;
    }

    if (!check((int)m.shape().second, 100, "Precision large matrix cols")) {
        failures += 1;
    }

    m.data()[0] = 5.0f;
    m.data()[9999] = 7.5f;

    if (!check(m.data()[0], 5.0f, 1e-6f, "Precision large matrix [0]")) {
        failures += 1;
    }

    if (!check(m.data()[9999], 7.5f, 1e-6f, "Precision large matrix [9999]")) {
        failures += 1;
    }

    return failures;
}

int test_matrix_precision_element_access() {
    int failures = 0;

    Matrix<float> m(Shape{3, 3}, 0.0f);

    m(0, 0) = 1.1f;
    m(1, 1) = 2.2f;
    m(2, 2) = 3.3f;

    if (!check(m(0, 0), 1.1f, 1e-6f, "Precision element [0,0]")) {
        failures += 1;
    }

    if (!check(m(2, 2), 3.3f, 1e-6f, "Precision element [2,2]")) {
        failures += 1;
    }

    return failures;
}

int test_matrix_precision_init_list() {
    int failures = 0;

    Matrix<float> m(Shape{2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    if (!check(m.data()[0], 1.0f, 1e-6f, "Precision init list [0]")) {
        failures += 1;
    }

    if (!check(m.data()[3], 4.0f, 1e-6f, "Precision init list [3]")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Main Test Runner
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

    if (total_failures == 0) {
        std::cout << "All GPU Matrix type tests passed!" << std::endl;
    } else {
        std::cout << "GPU Matrix type tests: " << total_failures << " failures" << std::endl;
    }

    return total_failures;
}

#else
int main() {
    std::cerr << "CUDA support not enabled" << std::endl;
    return 1;
}
#endif
