#include "test_common.h"

using namespace lahva::cpu;
using lahva::Shape;

// ============================================================================
// BlockDiagMatrix Constructor Tests
// ============================================================================

template <typename T>
int test_blockdiag_uniform_blocks_constructor() {
    size_t n_blocks = 3;
    Shape block_shape = Shape{4, 4};
    BlockDiagMatrix<T> m(n_blocks, block_shape);

    if (!check((int)m.num_blocks(), 3, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    Shape s = m.shape();
    if (!check((int)s.first, 12, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((int)s.second, 12, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_uniform_blocks_value_constructor() {
    size_t n_blocks = 2;
    Shape block_shape = Shape{3, 3};
    BlockDiagMatrix<T> m(n_blocks, block_shape, (T)2.5);

    if (!check((int)m.num_blocks(), 2, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    Shape s = m.shape();
    if (!check((int)s.first, 6, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_varied_blocks_constructor() {
    std::vector<Shape> shapes = {{2, 2}, {3, 3}, {1, 1}};
    BlockDiagMatrix<T> m(shapes);

    if (!check((int)m.num_blocks(), 3, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    Shape s = m.shape();
    if (!check((int)s.first, 6, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((int)s.second, 6, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    auto block_shapes = m.block_shapes();
    if (!check((int)block_shapes.size(), 3, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// BlockDiagMatrix Element Access Tests
// ============================================================================

template <typename T>
int test_blockdiag_element_access() {
    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 2, 3, 4}));
    blocks.push_back(Matrix<T>(Shape{2, 2}, {5, 6, 7, 8}));

    BlockDiagMatrix<T> m(blocks);

    if (!check((T)m(0, 0), (T)1, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((T)m(0, 1), (T)3, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((T)m(1, 0), (T)2, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check((T)m(2, 2), (T)5, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;
    if (!check((T)m(2, 3), (T)7, check_msg(get_type_name<T>(), "check 5"))) return TEST_FAIL;
    if (!check((T)m(0, 2), (T)0, check_msg(get_type_name<T>(), "check 6"))) return TEST_FAIL;
    if (!check((T)m(1, 3), (T)0, check_msg(get_type_name<T>(), "check 7"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// BlockDiagMatrix Operations Tests
// ============================================================================

template <typename T>
int test_blockdiag_get_diagonal() {
    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 0, 0, 2}));
    blocks.push_back(Matrix<T>(Shape{2, 2}, {3, 0, 0, 4}));

    BlockDiagMatrix<T> m(blocks);
    Vector<T> diag = m.get_diagonal();

    if (!check((int)diag.size(), 4, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((T)diag[0], (T)1, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((T)diag[1], (T)2, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check((T)diag[2], (T)3, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;
    if (!check((T)diag[3], (T)4, check_msg(get_type_name<T>(), "check 5"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_set_diagonal() {
    std::vector<Shape> shapes = {{2, 2}, {2, 2}};
    BlockDiagMatrix<T> m(shapes, (T)0);

    Vector<T> diag(4, (T)0);
    diag[0] = (T)5;
    diag[1] = (T)6;
    diag[2] = (T)7;
    diag[3] = (T)8;

    m.set_diagonal(diag);

    Vector<T> diag_check = m.get_diagonal();
    if (!check((T)diag_check[0], (T)5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((T)diag_check[2], (T)7, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_symmetrize() {
    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 3, 2, 4}));

    BlockDiagMatrix<T> m(blocks);
    m.symmetrize();

    if (!check((T)m(0, 0), (T)1, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((T)m(0, 1), (T)2.5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((T)m(1, 0), (T)2.5, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_add_block() {
    BlockDiagMatrix<T> m;

    if (!check((int)m.num_blocks(), 0, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    Matrix<T> block1(Shape{2, 2}, (T)1);
    m.add_block(block1);

    if (!check((int)m.num_blocks(), 1, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    Matrix<T> block2(Shape{3, 3}, (T)2);
    m.add_block(block2);

    if (!check((int)m.num_blocks(), 2, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    Shape s = m.shape();
    if (!check((int)s.first, 5, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;
    if (!check((int)s.second, 5, check_msg(get_type_name<T>(), "check 5"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_copy_constructor() {
    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 2, 3, 4}));
    BlockDiagMatrix<T> m1(blocks);

    BlockDiagMatrix<T> m2 = m1;

    if (!check((int)m2.num_blocks(), 1, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    Shape s = m2.shape();
    if (!check((int)s.first, 2, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int total_failures = 0;

    // BlockDiagMatrix constructors - double precision
    total_failures += test_blockdiag_uniform_blocks_constructor<double>();
    total_failures += test_blockdiag_uniform_blocks_value_constructor<double>();
    total_failures += test_blockdiag_varied_blocks_constructor<double>();

    // BlockDiagMatrix constructors - single precision
    total_failures += test_blockdiag_uniform_blocks_constructor<float>();
    total_failures += test_blockdiag_uniform_blocks_value_constructor<float>();
    total_failures += test_blockdiag_varied_blocks_constructor<float>();

    // BlockDiagMatrix element access - double precision
    total_failures += test_blockdiag_element_access<double>();

    // BlockDiagMatrix element access - single precision
    total_failures += test_blockdiag_element_access<float>();

    // BlockDiagMatrix operations - double precision
    total_failures += test_blockdiag_get_diagonal<double>();
    total_failures += test_blockdiag_set_diagonal<double>();
    total_failures += test_blockdiag_symmetrize<double>();
    total_failures += test_blockdiag_add_block<double>();
    total_failures += test_blockdiag_copy_constructor<double>();

    // BlockDiagMatrix operations - single precision
    total_failures += test_blockdiag_get_diagonal<float>();
    total_failures += test_blockdiag_set_diagonal<float>();
    total_failures += test_blockdiag_symmetrize<float>();
    total_failures += test_blockdiag_add_block<float>();
    total_failures += test_blockdiag_copy_constructor<float>();

    if (total_failures > 0) {
        std::cerr << "cpu/tensor/blockdiagmatrix tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All cpu/tensor/blockdiagmatrix tests passed!" << std::endl;
    return TEST_PASS;
}
