#include "test_common.h"

using namespace lahva::cpu;
using lahva::Shape;

// ============================================================================
// BlockMatrix Constructor Tests
// ============================================================================

template <typename T>
int test_blockmatrix_default_constructor() {
    BlockMatrix<T> m;

    Shape s = m.shape();
    if (!check((int)s.first, 0, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((int)s.second, 0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((int)m.num_blocks(), 0, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_copy_constructor() {
    BlockMatrix<T> m1;

    Matrix<T> block(Shape{2, 2}, (T)1.5);
    m1.set_block(0, 0, block);

    BlockMatrix<T> m2 = m1;

    if (!check((int)m2.num_blocks(), 1, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    Shape s = m2.shape();
    if (!check((int)s.first, 2, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((int)s.second, 2, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// BlockMatrix Block Access Tests
// ============================================================================

template <typename T>
int test_blockmatrix_set_single_block() {
    BlockMatrix<T> m;

    Matrix<T> block(Shape{3, 3}, (T)2.0);
    m.set_block(0, 0, block);

    if (!check((int)m.num_blocks(), 1, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    Shape s = m.shape();
    if (!check((int)s.first, 3, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((int)s.second, 3, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    if (!check((T)m(0, 0), (T)2.0, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;
    if (!check((T)m(2, 2), (T)2.0, check_msg(get_type_name<T>(), "check 5"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_set_multiple_blocks() {
    BlockMatrix<T> m;

    Matrix<T> block1(Shape{2, 2}, (T)1.0);
    Matrix<T> block2(Shape{3, 3}, (T)2.0);

    m.set_block(0, 0, block1);
    m.set_block(2, 2, block2);

    if (!check((int)m.num_blocks(), 2, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    Shape s = m.shape();
    if (!check((int)s.first, 5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((int)s.second, 5, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    if (!check((T)m(0, 0), (T)1.0, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;
    if (!check((T)m(2, 2), (T)2.0, check_msg(get_type_name<T>(), "check 5"))) return TEST_FAIL;
    if (!check((T)m(1, 1), (T)1.0, check_msg(get_type_name<T>(), "check 6"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_element_access() {
    BlockMatrix<T> m;

    Matrix<T> block(Shape{2, 2}, {1, 2, 3, 4});
    m.set_block(0, 0, block);

    if (!check((T)m(0, 0), (T)1, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((T)m(0, 1), (T)3, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((T)m(1, 0), (T)2, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check((T)m(1, 1), (T)4, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_has_block() {
    BlockMatrix<T> m;

    Matrix<T> block(Shape{2, 2}, (T)1.0);
    m.set_block(0, 0, block);

    if (!m.has_block(0, 0)) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    }

    if (m.has_block(1, 1)) {
        std::cerr << check_msg(get_type_name<T>(), "check 2") << std::endl;
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_block_overlap_detection() {
    BlockMatrix<T> m;

    Matrix<T> block1(Shape{2, 2}, (T)1.0);
    Matrix<T> block2(Shape{2, 2}, (T)2.0);

    m.set_block(0, 0, block1);

    try {
        m.set_block(1, 1, block2);
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    } catch (const std::runtime_error&) {
        return TEST_PASS;
    }
}

// ============================================================================
// BlockMatrix BLAS Tests - MatrixMatrixProduct
// ============================================================================

template <typename T>
int test_blockmatrix_gemm_basic() {
    BlockMatrix<T> A;
    Matrix<T> B(Shape{4, 3}, (T)1.0);
    Matrix<T> C(Shape{4, 3}, (T)0.0);

    Matrix<T> block_A(Shape{4, 4}, (T)2.0);
    A.set_block(0, 0, block_A);

    MatrixMatrixProduct("N", "N", (T)1.0, A, B, (T)0.0, C);

    if (!check((double)C(0, 0), 8.0, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_gemm_with_scaling() {
    BlockMatrix<T> A;
    Matrix<T> B(Shape{4, 3}, (T)1.0);
    Matrix<T> C(Shape{4, 3}, (T)5.0);

    Matrix<T> block_A(Shape{4, 4}, (T)2.0);
    A.set_block(0, 0, block_A);

    MatrixMatrixProduct("N", "N", (T)0.5, A, B, (T)1.0, C);

    double expected = 0.5 * (4.0 * 2.0 * 1.0) + 5.0;
    if (!check((double)C(0, 0), expected, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_gemm_normal_matrix_as_first() {
    Matrix<T> A(Shape{4, 4}, (T)2.0);
    BlockMatrix<T> B;
    Matrix<T> C(Shape{4, 3}, (T)0.0);

    Matrix<T> block_B(Shape{4, 3}, (T)1.0);
    B.set_block(0, 0, block_B);

    MatrixMatrixProduct("N", "N", (T)1.0, A, B, (T)0.0, C);

    if (!check((double)C(0, 0), 8.0, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// BlockMatrix BLAS Tests - MatrixVectorProduct
// ============================================================================

template <typename T>
int test_blockmatrix_gemv_basic() {
    BlockMatrix<T> A;
    Vector<T> x(4, (T)1.0);
    Vector<T> y(4, (T)0.0);

    Matrix<T> block_A(Shape{4, 4}, (T)2.0);
    A.set_block(0, 0, block_A);

    MatrixVectorProduct("N", (T)1.0, A, x, 1, (T)0.0, y, 1);

    if (!check((double)y[0], 8.0, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_gemv_with_scaling() {
    BlockMatrix<T> A;
    Vector<T> x(4, (T)1.0);
    Vector<T> y(4, (T)5.0);

    Matrix<T> block_A(Shape{4, 4}, (T)2.0);
    A.set_block(0, 0, block_A);

    MatrixVectorProduct("N", (T)0.5, A, x, 1, (T)1.0, y, 1);

    double expected = 0.5 * (4.0 * 2.0) + 5.0;
    if (!check((double)y[0], expected, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_gemv_transpose() {
    BlockMatrix<T> A;
    Vector<T> x(4, (T)1.0);
    Vector<T> y(4, (T)0.0);

    Matrix<T> block_A(Shape{4, 4}, (T)2.0);
    A.set_block(0, 0, block_A);

    MatrixVectorProduct("T", (T)1.0, A, x, 1, (T)0.0, y, 1);

    if (!check((double)y[0], 8.0, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_gemv_normal_matrix_first() {
    Matrix<T> A(Shape{4, 4}, (T)2.0);
    BlockMatrix<T> B;
    Vector<T> x(4, (T)1.0);
    Vector<T> y(4, (T)0.0);

    Matrix<T> block_B(Shape{4, 4}, (T)1.5);
    B.set_block(0, 0, block_B);

    MatrixVectorProduct("N", (T)1.0, A, x, 1, (T)0.0, y, 1);

    if (!check((double)y[0], 8.0, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int total_failures = 0;

    // BlockMatrix constructors - double precision
    total_failures += test_blockmatrix_default_constructor<double>();
    total_failures += test_blockmatrix_copy_constructor<double>();

    // BlockMatrix constructors - single precision
    total_failures += test_blockmatrix_default_constructor<float>();
    total_failures += test_blockmatrix_copy_constructor<float>();

    // BlockMatrix block access - double precision
    total_failures += test_blockmatrix_set_single_block<double>();
    total_failures += test_blockmatrix_set_multiple_blocks<double>();
    total_failures += test_blockmatrix_element_access<double>();
    total_failures += test_blockmatrix_has_block<double>();
    total_failures += test_blockmatrix_block_overlap_detection<double>();

    // BlockMatrix block access - single precision
    total_failures += test_blockmatrix_set_single_block<float>();
    total_failures += test_blockmatrix_set_multiple_blocks<float>();
    total_failures += test_blockmatrix_element_access<float>();
    total_failures += test_blockmatrix_has_block<float>();
    total_failures += test_blockmatrix_block_overlap_detection<float>();

    // BlockMatrix GEMM tests - double precision
    total_failures += test_blockmatrix_gemm_basic<double>();
    total_failures += test_blockmatrix_gemm_with_scaling<double>();
    total_failures += test_blockmatrix_gemm_normal_matrix_as_first<double>();

    // BlockMatrix GEMM tests - single precision
    total_failures += test_blockmatrix_gemm_basic<float>();
    total_failures += test_blockmatrix_gemm_with_scaling<float>();
    total_failures += test_blockmatrix_gemm_normal_matrix_as_first<float>();

    // BlockMatrix GEMV tests - double precision
    total_failures += test_blockmatrix_gemv_basic<double>();
    total_failures += test_blockmatrix_gemv_with_scaling<double>();
    total_failures += test_blockmatrix_gemv_transpose<double>();
    total_failures += test_blockmatrix_gemv_normal_matrix_first<double>();

    // BlockMatrix GEMV tests - single precision
    total_failures += test_blockmatrix_gemv_basic<float>();
    total_failures += test_blockmatrix_gemv_with_scaling<float>();
    total_failures += test_blockmatrix_gemv_transpose<float>();
    total_failures += test_blockmatrix_gemv_normal_matrix_first<float>();

    if (total_failures > 0) {
        std::cerr << "cpu/tensor/block-matrix tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All cpu/tensor/block-matrix tests passed!" << std::endl;
    return TEST_PASS;
}
