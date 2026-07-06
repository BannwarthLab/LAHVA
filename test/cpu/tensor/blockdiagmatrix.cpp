#include "test_common.h"

using namespace lahva::cpu;

const double TOLERANCE = 5.0e-7;

// ============================================================================
// BlockDiagMatrix Constructor Tests
// ============================================================================

template <typename T>
int test_blockdiag_uniform_blocks_constructor() {
    int failures = 0;

    size_t n_blocks = 3;
    Shape block_shape = Shape{4, 4};
    BlockDiagMatrix<T> m(n_blocks, block_shape);

    if (!check((int)m.num_blocks(), 3, "Should have 3 blocks")) {
        failures += 1;
    }

    Shape s = m.shape();
    if (!check((int)s.first, 12, "Total rows should be 12")) {
        failures += 1;
    }

    if (!check((int)s.second, 12, "Total cols should be 12")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_uniform_blocks_value_constructor() {
    int failures = 0;

    size_t n_blocks = 2;
    Shape block_shape = Shape{3, 3};
    BlockDiagMatrix<T> m(n_blocks, block_shape, (T)2.5);

    if (!check((int)m.num_blocks(), 2, "Should have 2 blocks")) {
        failures += 1;
    }

    Shape s = m.shape();
    if (!check((int)s.first, 6, "Total rows should be 6")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_varied_blocks_constructor() {
    int failures = 0;

    std::vector<Shape> shapes = {{2, 2}, {3, 3}, {1, 1}};
    BlockDiagMatrix<T> m(shapes);

    if (!check((int)m.num_blocks(), 3, "Should have 3 blocks")) {
        failures += 1;
    }

    Shape s = m.shape();
    if (!check((int)s.first, 6, "Total rows should be 6")) {
        failures += 1;
    }

    if (!check((int)s.second, 6, "Total cols should be 6")) {
        failures += 1;
    }

    auto block_shapes = m.block_shapes();
    if (block_shapes.size() != 3) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// BlockDiagMatrix Element Access Tests
// ============================================================================

template <typename T>
int test_blockdiag_element_access() {
    int failures = 0;

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 2, 3, 4}));
    blocks.push_back(Matrix<T>(Shape{2, 2}, {5, 6, 7, 8}));

    BlockDiagMatrix<T> m(blocks);

    // Test first block elements
    if (!check((T)m(0, 0), (T)1, TOLERANCE, "Element (0,0) should be 1")) {
        failures += 1;
    }
    if (!check((T)m(0, 1), (T)3, TOLERANCE, "Element (0,1) should be 3")) {
        failures += 1;
    }
    if (!check((T)m(1, 0), (T)2, TOLERANCE, "Element (1,0) should be 2")) {
        failures += 1;
    }

    // Test second block elements
    if (!check((T)m(2, 2), (T)5, TOLERANCE, "Element (2,2) should be 5")) {
        failures += 1;
    }
    if (!check((T)m(2, 3), (T)7, TOLERANCE, "Element (2,3) should be 7")) {
        failures += 1;
    }

    // Test off-diagonal zeros
    if (!check((T)m(0, 2), (T)0, TOLERANCE, "Element (0,2) should be 0")) {
        failures += 1;
    }
    if (!check((T)m(1, 3), (T)0, TOLERANCE, "Element (1,3) should be 0")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// BlockDiagMatrix Operations Tests
// ============================================================================

template <typename T>
int test_blockdiag_get_diagonal() {
    int failures = 0;

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 0, 0, 2}));
    blocks.push_back(Matrix<T>(Shape{2, 2}, {3, 0, 0, 4}));

    BlockDiagMatrix<T> m(blocks);
    Vector<T> diag = m.get_diagonal();

    if (!check((int)diag.size(), 4, "Diagonal should have 4 elements")) {
        failures += 1;
    }

    if (!check((T)diag[0], (T)1, TOLERANCE, "First diagonal element should be 1")) {
        failures += 1;
    }
    if (!check((T)diag[1], (T)2, TOLERANCE, "Second diagonal element should be 2")) {
        failures += 1;
    }
    if (!check((T)diag[2], (T)3, TOLERANCE, "Third diagonal element should be 3")) {
        failures += 1;
    }
    if (!check((T)diag[3], (T)4, TOLERANCE, "Fourth diagonal element should be 4")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_set_diagonal() {
    int failures = 0;

    std::vector<Shape> shapes = {{2, 2}, {2, 2}};
    BlockDiagMatrix<T> m(shapes, (T)0);

    Vector<T> diag(4, (T)0);
    diag[0] = (T)5;
    diag[1] = (T)6;
    diag[2] = (T)7;
    diag[3] = (T)8;

    m.set_diagonal(diag);

    // Verify diagonal was set
    Vector<T> diag_check = m.get_diagonal();
    if (!check((T)diag_check[0], (T)5, TOLERANCE, "Diagonal element 0 should be 5")) {
        failures += 1;
    }
    if (!check((T)diag_check[2], (T)7, TOLERANCE, "Diagonal element 2 should be 7")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_symmetrize() {
    int failures = 0;

    std::vector<Matrix<T>> blocks;
    // Create asymmetric matrix: [[1, 2], [3, 4]]
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 3, 2, 4}));

    BlockDiagMatrix<T> m(blocks);
    m.symmetrize();

    // After symmetrization: [[1, 2.5], [2.5, 4]]
    if (!check((T)m(0, 0), (T)1, TOLERANCE, "Element (0,0) should be 1")) {
        failures += 1;
    }
    if (!check((T)m(0, 1), (T)2.5, TOLERANCE, "Element (0,1) should be 2.5")) {
        failures += 1;
    }
    if (!check((T)m(1, 0), (T)2.5, TOLERANCE, "Element (1,0) should be 2.5")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_add_block() {
    int failures = 0;

    BlockDiagMatrix<T> m;

    if (!check((int)m.num_blocks(), 0, "Empty block diagonal should have 0 blocks")) {
        failures += 1;
    }

    Matrix<T> block1(Shape{2, 2}, (T)1);
    m.add_block(block1);

    if (!check((int)m.num_blocks(), 1, "Should have 1 block after adding")) {
        failures += 1;
    }

    Matrix<T> block2(Shape{3, 3}, (T)2);
    m.add_block(block2);

    if (!check((int)m.num_blocks(), 2, "Should have 2 blocks after adding")) {
        failures += 1;
    }

    Shape s = m.shape();
    if (!check((int)s.first, 5, "Total rows should be 5")) {
        failures += 1;
    }
    if (!check((int)s.second, 5, "Total cols should be 5")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_copy_constructor() {
    int failures = 0;

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 2, 3, 4}));
    BlockDiagMatrix<T> m1(blocks);

    BlockDiagMatrix<T> m2 = m1;

    if (!check((int)m2.num_blocks(), 1, "Copied matrix should have 1 block")) {
        failures += 1;
    }

    Shape s = m2.shape();
    if (!check((int)s.first, 2, "Copied matrix should have 2 rows")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Main Test Entry Point
// ============================================================================

int main() {
    int failures = 0;

    std::cout << "Testing BlockDiagMatrix<double> constructors..." << std::endl;
    failures += test_blockdiag_uniform_blocks_constructor<double>();
    failures += test_blockdiag_uniform_blocks_value_constructor<double>();
    failures += test_blockdiag_varied_blocks_constructor<double>();

    std::cout << "Testing BlockDiagMatrix<float> constructors..." << std::endl;
    failures += test_blockdiag_uniform_blocks_constructor<float>();
    failures += test_blockdiag_uniform_blocks_value_constructor<float>();
    failures += test_blockdiag_varied_blocks_constructor<float>();

    std::cout << "Testing BlockDiagMatrix<double> element access..." << std::endl;
    failures += test_blockdiag_element_access<double>();

    std::cout << "Testing BlockDiagMatrix<float> element access..." << std::endl;
    failures += test_blockdiag_element_access<float>();

    std::cout << "Testing BlockDiagMatrix<double> operations..." << std::endl;
    failures += test_blockdiag_get_diagonal<double>();
    failures += test_blockdiag_set_diagonal<double>();
    failures += test_blockdiag_symmetrize<double>();
    failures += test_blockdiag_add_block<double>();
    failures += test_blockdiag_copy_constructor<double>();

    std::cout << "Testing BlockDiagMatrix<float> operations..." << std::endl;
    failures += test_blockdiag_get_diagonal<float>();
    failures += test_blockdiag_set_diagonal<float>();
    failures += test_blockdiag_symmetrize<float>();
    failures += test_blockdiag_add_block<float>();
    failures += test_blockdiag_copy_constructor<float>();

    if (failures == 0) {
        std::cout << "All BlockDiagMatrix tests passed!" << std::endl;
    } else {
        std::cout << "BlockDiagMatrix tests failed with " << failures << " failures!" << std::endl;
    }

    return failures;
}
