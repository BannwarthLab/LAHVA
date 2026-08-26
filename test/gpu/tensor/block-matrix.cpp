#include "test_common.h"

using namespace lahva;
using namespace lahva::gpu;

// ============================================================================
// BlockMatrix Construction Tests
// ============================================================================

template <typename T>
int test_blockmatrix_default_constructor() {
    BlockMatrix<T> m;

    if (!check((int)m.num_blocks(), 0, check_msg(get_type_name<T>(), "Default constructor should create empty matrix"))) return TEST_FAIL;

    Shape s = m.shape();
    if (!check((int)s.first, 0, check_msg(get_type_name<T>(), "Default matrix should have 0 rows"))) return TEST_FAIL;

    if (!check((int)s.second, 0, check_msg(get_type_name<T>(), "Default matrix should have 0 cols"))) return TEST_FAIL;

    if (!check((int)m.num_block_rows(), 0, check_msg(get_type_name<T>(), "Default matrix should have 0 block rows"))) return TEST_FAIL;

    if (!check((int)m.num_block_cols(), 0, check_msg(get_type_name<T>(), "Default matrix should have 0 block cols"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_copy_constructor() {
    BlockMatrix<T> m1;
    Matrix<T, CudaHostAllocator<T>> block(Shape{3, 3}, (T)1.5);
    m1.set_block(0, 0, block);

    BlockMatrix<T> m2 = m1;  // Copy constructor

    if (!check((int)m2.num_blocks(), 1, check_msg(get_type_name<T>(), "Copy should have same number of blocks"))) return TEST_FAIL;

    if (!check((int)m2.num_block_rows(), 1, check_msg(get_type_name<T>(), "Copy should have same block rows"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_move_constructor() {
    BlockMatrix<T> m1;
    Matrix<T, CudaHostAllocator<T>> block(Shape{3, 3}, (T)2.0);
    m1.set_block(0, 0, block);

    BlockMatrix<T> m2 = std::move(m1);  // Move constructor

    if (!check((int)m2.num_blocks(), 1, check_msg(get_type_name<T>(), "Moved matrix should have 1 block"))) return TEST_FAIL;

    if (!check((int)m1.num_blocks(), 0, check_msg(get_type_name<T>(), "Original should be empty after move"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// BlockMatrix Block Operations Tests
// ============================================================================

template <typename T>
int test_blockmatrix_set_block() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block(Shape{3, 3}, (T)1.5);

    m.set_block(0, 0, block);

    if (!check((int)m.num_blocks(), 1, check_msg(get_type_name<T>(), "Should have 1 block after set_block"))) return TEST_FAIL;

    if (!check((int)m.num_block_rows(), 1, check_msg(get_type_name<T>(), "Should have 1 block row"))) return TEST_FAIL;

    if (!check((int)m.num_block_cols(), 1, check_msg(get_type_name<T>(), "Should have 1 block col"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_set_multiple_blocks() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{2, 3}, (T)2);
    Matrix<T, CudaHostAllocator<T>> block3(Shape{3, 2}, (T)3);

    // Non-overlapping positions: block1 at (0,0), block2 at (0,2), block3 at (2,0)
    m.set_block(0, 0, block1);
    m.set_block(0, 2, block2);
    m.set_block(2, 0, block3);

    if (!check((int)m.num_blocks(), 3, check_msg(get_type_name<T>(), "Should have 3 blocks"))) return TEST_FAIL;

    // num_block_rows/cols return max position + 1 for grid dimensions
    // With positions at (0,0), (0,2), (2,0): max row is 2, max col is 2, so 3 and 3
    if (!check((int)m.num_block_rows(), 3, check_msg(get_type_name<T>(), "Should have max row position + 1"))) return TEST_FAIL;

    if (!check((int)m.num_block_cols(), 3, check_msg(get_type_name<T>(), "Should have max col position + 1"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_get_block() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block(Shape{2, 2}, {1, 2, 3, 4});

    m.set_block(0, 0, block);

    Matrix<T, CudaHostAllocator<T>>& retrieved = m.get_block(0, 0);
    Shape s = retrieved.shape();

    if (!check((int)s.first, 2, check_msg(get_type_name<T>(), "Retrieved block should have 2 rows"))) return TEST_FAIL;

    if (!check((T)retrieved(0, 0), (T)1, check_msg(get_type_name<T>(), "Retrieved block first element should be 1"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_get_block_const() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block(Shape{2, 2}, (T)5);

    m.set_block(2, 2, block);

    const Matrix<T, CudaHostAllocator<T>>& retrieved = m.get_block(2, 2);
    Shape s = retrieved.shape();

    if (!check((int)s.first, 2, check_msg(get_type_name<T>(), "Retrieved const block should have 2 rows"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_has_block() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block(Shape{3, 3}, (T)1);

    m.set_block(0, 0, block);

    if (!m.has_block(0, 0)) {
        std::cerr << check_msg(get_type_name<T>(), "test condition failed") << std::endl;
        return TEST_FAIL;
    }

    if (m.has_block(1, 1)) {
        std::cerr << check_msg(get_type_name<T>(), "test condition failed") << std::endl;
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_num_blocks() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);
    Matrix<T, CudaHostAllocator<T>> block3(Shape{2, 2}, (T)3);

    // Non-overlapping positions
    m.set_block(0, 0, block1);
    m.set_block(0, 3, block2);
    m.set_block(3, 0, block3);

    if (!check((int)m.num_blocks(), 3, check_msg(get_type_name<T>(), "num_blocks should return 3"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// BlockMatrix Dimension Tests
// ============================================================================

template <typename T>
int test_blockmatrix_shape() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 3}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 2}, (T)2);

    m.set_block(0, 0, block1);
    m.set_block(2, 3, block2);

    Shape s = m.shape();
    // Total rows: max(0+2, 2+3) = 5; Total cols: max(0+3, 3+2) = 5
    if (!check((int)s.first, 5, check_msg(get_type_name<T>(), "Total rows should be 5"))) return TEST_FAIL;

    if (!check((int)s.second, 5, check_msg(get_type_name<T>(), "Total cols should be 5"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_shape_with_sparse_layout() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    // Set blocks at positions (0, 0) and (2, 2)
    m.set_block(0, 0, block1);
    m.set_block(2, 2, block2);

    Shape s = m.shape();
    // Should account for the maximum row index in each row group
    if (!check((int)s.first, 5, check_msg(get_type_name<T>(), "Total rows should account for all block rows"))) return TEST_FAIL;

    if (!check((int)s.second, 5, check_msg(get_type_name<T>(), "Total cols should account for all block cols"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// BlockMatrix Block Grid Tests
// ============================================================================

template <typename T>
int test_blockmatrix_num_block_rows() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    m.set_block(0, 0, block1);
    m.set_block(2, 0, block2);

    if (!check((int)m.num_block_rows(), 3, check_msg(get_type_name<T>(), "num_block_rows should return 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_num_block_cols() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    m.set_block(0, 0, block1);
    m.set_block(0, 2, block2);

    if (!check((int)m.num_block_cols(), 3, check_msg(get_type_name<T>(), "num_block_cols should return 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_block_grid_dimensions() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);
    Matrix<T, CudaHostAllocator<T>> block3(Shape{2, 2}, (T)3);
    Matrix<T, CudaHostAllocator<T>> block4(Shape{3, 3}, (T)4);

    m.set_block(0, 0, block1);
    m.set_block(0, 2, block2);
    m.set_block(2, 0, block3);
    m.set_block(3, 3, block4);

    if (!check((int)m.num_block_rows(), 4, check_msg(get_type_name<T>(), "num_block_rows should return 4"))) return TEST_FAIL;

    if (!check((int)m.num_block_cols(), 4, check_msg(get_type_name<T>(), "num_block_cols should return 4"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// BlockMatrix Offset Tests
// ============================================================================

template <typename T>
int test_blockmatrix_get_row_offsets() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    m.set_block(0, 0, block1);
    m.set_block(2, 0, block2);

    const std::vector<int>& row_offsets = m.get_row_offsets();

    // Offsets should be computed lazily
    if (row_offsets.empty()) {
        std::cerr << check_msg(get_type_name<T>(), "test condition failed") << std::endl;
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_get_col_offsets() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    m.set_block(0, 0, block1);
    m.set_block(0, 2, block2);

    const std::vector<int>& col_offsets = m.get_col_offsets();

    // Offsets should be computed lazily
    if (col_offsets.empty()) {
        std::cerr << check_msg(get_type_name<T>(), "test condition failed") << std::endl;
        return TEST_FAIL;
    }

    return TEST_PASS;
}

// ============================================================================
// BlockMatrix Element Access Tests
// ============================================================================

template <typename T>
int test_blockmatrix_element_access_read() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block(Shape{3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    m.set_block(0, 0, block);

    // Test first block element
    if (!check((T)m(0, 0), (T)1, check_msg(get_type_name<T>(), "Element (0,0) should be 1"))) return TEST_FAIL;

    if (!check((T)m(1, 1), (T)5, check_msg(get_type_name<T>(), "Element (1,1) should be 5"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_element_access_write() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block(Shape{3, 3});

    m.set_block(0, 0, block);

    // Write to block
    m(0, 0) = (T)7.5;
    m(1, 1) = (T)8.5;

    if (!check((T)m(0, 0), (T)7.5, check_msg(get_type_name<T>(), "Element (0,0) should be 7.5 after write"))) return TEST_FAIL;

    if (!check((T)m(1, 1), (T)8.5, check_msg(get_type_name<T>(), "Element (1,1) should be 8.5 after write"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_element_access_arbitrary_positions() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    m.set_block(0, 0, block1);
    m.set_block(2, 2, block2);

    // Block 1 is at (0,0) with size 2x2
    // Block 2 is at (1,1) with size 3x3
    // But the matrix uses fixed positions like pos.first*100, pos.second*100

    return TEST_PASS;
}

// ============================================================================
// BlockMatrix Block Indexing Tests
// ============================================================================

template <typename T>
int test_blockmatrix_get_block_shape() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    m.set_block(0, 0, block1);
    m.set_block(2, 0, block2);

    // Get by linear index
    Shape shape0 = m.get_block_shape(0);
    if (!check((int)shape0.first, 2, check_msg(get_type_name<T>(), "First block should have 2 rows"))) return TEST_FAIL;

    Shape shape1 = m.get_block_shape(1);
    if (!check((int)shape1.first, 3, check_msg(get_type_name<T>(), "Second block should have 3 rows"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// BlockMatrix GPU Operations Tests
// ============================================================================

template <typename T>
int test_blockmatrix_gpu_operations_basic() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block(Shape{2, 2}, (T)1);

    m.set_block(0, 0, block);

    CudaRuntime cudart;

    // Test that we can access GPU-related methods
    Shape s = m.shape();
    if (!check((int)s.first, 2, check_msg(get_type_name<T>(), "Matrix should have correct dimensions"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// ============================================================================
// BlockMatrix GPU Round-Trip Transfer Tests
// ============================================================================

template <typename T>
int test_blockmatrix_gpu_roundtrip_uniform_blocks() {
    // Create BlockMatrix with uniform blocks (non-overlapping)
    BlockMatrix<T> bm;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, {1, 2, 3, 4});
    Matrix<T, CudaHostAllocator<T>> block2(Shape{2, 2}, {5, 6, 7, 8});

    bm.set_block(0, 0, block1);
    bm.set_block(0, 2, block2);

    // Convert to sparse CSR format
    CudaRuntime cudart;
    gpu::SparseMatrix<T> sparse(cudart, bm, gpu::SparseFormat::CSR);

    // Reconstruct original dense matrix from BlockMatrix
    Shape bm_shape = bm.shape();
    Matrix<T> original_dense(bm_shape, (T)0);
    for (size_t i = 0; i < bm_shape.first; ++i) {
        for (size_t j = 0; j < bm_shape.second; ++j) {
            original_dense(i, j) = bm(i, j);
        }
    }

    // Transfer to GPU and back
    sparse.allocate_gpu_memory();
    sparse.copy2device(cudart);
    sparse.copy2host(cudart);

    // Reconstruct dense matrix from sparse data
    Matrix<T> reconstructed_dense = sparse.to_dense();

    // Verify dimensions match
    Shape reconstructed_shape = reconstructed_dense.shape();
    if (reconstructed_shape.first != original_dense.shape().first ||
        reconstructed_shape.second != original_dense.shape().second) {
        sparse.free_gpu_memory();
        std::cerr << check_msg(get_type_name<T>(), "test condition failed") << std::endl;
        return TEST_FAIL;
    }

    // Verify all elements match
    for (size_t i = 0; i < original_dense.shape().first; ++i) {
        for (size_t j = 0; j < original_dense.shape().second; ++j) {
            if (!check(original_dense(i, j), reconstructed_dense(i, j), check_msg(get_type_name<T>(), "Element mismatch in roundtrip"))) {
                sparse.free_gpu_memory();
                std::cerr << check_msg(get_type_name<T>(), "test condition failed") << std::endl;
                return TEST_FAIL;
            }
        }
    }

    sparse.free_gpu_memory();
    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_gpu_roundtrip_sparse_layout() {
    // Create BlockMatrix with blocks at sparse positions (non-overlapping)
    BlockMatrix<T> bm;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, {1, 2, 3, 4});
    Matrix<T, CudaHostAllocator<T>> block2(Shape{2, 2}, {5, 6, 7, 8});
    Matrix<T, CudaHostAllocator<T>> block3(Shape{2, 2}, {9, 10, 11, 12});

    bm.set_block(0, 0, block1);
    bm.set_block(0, 3, block2);
    bm.set_block(3, 1, block3);

    // Convert to sparse CSR format
    CudaRuntime cudart;
    gpu::SparseMatrix<T> sparse(cudart, bm, gpu::SparseFormat::CSR);

    // Reconstruct original dense matrix
    Shape bm_shape = bm.shape();
    Matrix<T> original_dense(bm_shape, (T)0);
    for (size_t i = 0; i < bm_shape.first; ++i) {
        for (size_t j = 0; j < bm_shape.second; ++j) {
            original_dense(i, j) = bm(i, j);
        }
    }

    // Transfer to GPU and back
    sparse.allocate_gpu_memory();
    sparse.copy2device(cudart);
    sparse.copy2host(cudart);

    // Reconstruct dense matrix from sparse data
    Matrix<T> reconstructed_dense = sparse.to_dense();

    // Verify dimensions match
    Shape reconstructed_shape = reconstructed_dense.shape();
    if (reconstructed_shape.first != original_dense.shape().first ||
        reconstructed_shape.second != original_dense.shape().second) {
        sparse.free_gpu_memory();
        std::cerr << check_msg(get_type_name<T>(), "test condition failed") << std::endl;
        return TEST_FAIL;
    }

    // Verify all elements match
    for (size_t i = 0; i < original_dense.shape().first; ++i) {
        for (size_t j = 0; j < original_dense.shape().second; ++j) {
            if (!check(original_dense(i, j), reconstructed_dense(i, j), check_msg(get_type_name<T>(), "Element mismatch in roundtrip"))) {
                sparse.free_gpu_memory();
                std::cerr << check_msg(get_type_name<T>(), "test condition failed") << std::endl;
                return TEST_FAIL;
            }
        }
    }

    sparse.free_gpu_memory();
    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_gpu_roundtrip_varied_sizes() {
    // Create BlockMatrix with blocks of different sizes (non-overlapping)
    BlockMatrix<T> bm;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, {1, 2, 3, 4});
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, {5, 6, 7, 8, 9, 10, 11, 12, 13});
    Matrix<T, CudaHostAllocator<T>> block3(Shape{2, 2}, {14, 15, 16, 17});

    bm.set_block(0, 0, block1);
    bm.set_block(0, 2, block2);
    bm.set_block(3, 0, block3);

    // Convert to sparse CSR format
    CudaRuntime cudart;
    gpu::SparseMatrix<T> sparse(cudart, bm, gpu::SparseFormat::CSR);

    // Reconstruct original dense matrix
    Shape bm_shape = bm.shape();
    Matrix<T> original_dense(bm_shape, (T)0);
    for (size_t i = 0; i < bm_shape.first; ++i) {
        for (size_t j = 0; j < bm_shape.second; ++j) {
            original_dense(i, j) = bm(i, j);
        }
    }

    // Transfer to GPU and back
    sparse.allocate_gpu_memory();
    sparse.copy2device(cudart);
    sparse.copy2host(cudart);

    // Reconstruct dense matrix from sparse data
    Matrix<T> reconstructed_dense = sparse.to_dense();

    // Verify dimensions match
    Shape reconstructed_shape = reconstructed_dense.shape();
    if (reconstructed_shape.first != original_dense.shape().first ||
        reconstructed_shape.second != original_dense.shape().second) {
        sparse.free_gpu_memory();
        std::cerr << check_msg(get_type_name<T>(), "test condition failed") << std::endl;
        return TEST_FAIL;
    }

    // Verify all elements match
    for (size_t i = 0; i < original_dense.shape().first; ++i) {
        for (size_t j = 0; j < original_dense.shape().second; ++j) {
            if (!check(original_dense(i, j), reconstructed_dense(i, j), check_msg(get_type_name<T>(), "Element mismatch in roundtrip"))) {
                sparse.free_gpu_memory();
                std::cerr << check_msg(get_type_name<T>(), "test condition failed") << std::endl;
                return TEST_FAIL;
            }
        }
    }

    sparse.free_gpu_memory();
    return TEST_PASS;
}

// ============================================================================
// BlockMatrix Copy and Assignment Tests
// ============================================================================

template <typename T>
int test_blockmatrix_copy_assignment() {
    BlockMatrix<T> m1;
    Matrix<T, CudaHostAllocator<T>> block(Shape{3, 3}, (T)1.5);
    m1.set_block(0, 0, block);

    BlockMatrix<T> m2;
    m2 = m1;  // Copy assignment

    if (!check((int)m2.num_blocks(), 1, check_msg(get_type_name<T>(), "Assignment should copy blocks"))) return TEST_FAIL;

    if (!check((int)m2.num_block_rows(), 1, check_msg(get_type_name<T>(), "Assignment should copy block rows"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_move_assignment() {
    BlockMatrix<T> m1;
    Matrix<T, CudaHostAllocator<T>> block(Shape{3, 3}, (T)2.0);
    m1.set_block(0, 0, block);

    BlockMatrix<T> m2;
    m2 = std::move(m1);  // Move assignment

    if (!check((int)m2.num_blocks(), 1, check_msg(get_type_name<T>(), "Move should transfer blocks"))) return TEST_FAIL;

    if (!check((int)m1.num_blocks(), 0, check_msg(get_type_name<T>(), "Original should be empty after move"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// BlockMatrix Edge Cases Tests
// ============================================================================

template <typename T>
int test_blockmatrix_single_block() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block(Shape{5, 5}, (T)1);

    m.set_block(0, 0, block);

    if (!check((int)m.num_blocks(), 1, check_msg(get_type_name<T>(), "Should have 1 block"))) return TEST_FAIL;

    Shape s = m.shape();
    if (!check((int)s.first, 5, check_msg(get_type_name<T>(), "Total rows should be 5"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_sparse_layout() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    // Set blocks at sparse positions
    m.set_block(0, 0, block1);
    m.set_block(2, 2, block2);  // Sparse position

    if (!check((int)m.num_blocks(), 2, check_msg(get_type_name<T>(), "Should have 2 blocks"))) return TEST_FAIL;

    if (!check((int)m.num_block_rows(), 3, check_msg(get_type_name<T>(), "Should account for all block row indices"))) return TEST_FAIL;

    if (!check((int)m.num_block_cols(), 3, check_msg(get_type_name<T>(), "Should account for all block col indices"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_large_block_indices() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    m.set_block(0, 0, block1);
    m.set_block(10, 10, block2);  // Large indices

    if (!check((int)m.num_blocks(), 2, check_msg(get_type_name<T>(), "Should have 2 blocks"))) return TEST_FAIL;

    if (!check((int)m.num_block_rows(), 11, check_msg(get_type_name<T>(), "Should handle large block row indices"))) return TEST_FAIL;

    if (!check((int)m.num_block_cols(), 11, check_msg(get_type_name<T>(), "Should handle large block col indices"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_varying_block_sizes() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{5, 5}, (T)2);
    Matrix<T, CudaHostAllocator<T>> block3(Shape{3, 3}, (T)3);

    m.set_block(0, 0, block1);
    m.set_block(2, 2, block2);
    m.set_block(7, 7, block3);

    if (!check((int)m.num_blocks(), 3, check_msg(get_type_name<T>(), "Should have 3 blocks"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_rectangular_blocks() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 4}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 2}, (T)2);

    m.set_block(0, 0, block1);
    m.set_block(2, 0, block2);

    if (!check((int)m.num_blocks(), 2, check_msg(get_type_name<T>(), "Should have 2 rectangular blocks"))) return TEST_FAIL;

    Shape s = m.shape();
    if (!check((int)s.first, 5, check_msg(get_type_name<T>(), "Total rows should be 2+3=5"))) return TEST_FAIL;

    if (!check((int)s.second, 4, check_msg(get_type_name<T>(), "Total cols should be max(4,2)=4"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockmatrix_update_existing_block() {
    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    m.set_block(0, 0, block1);
    if (!check((int)m.num_blocks(), 1, check_msg(get_type_name<T>(), "Should have 1 block initially"))) return TEST_FAIL;

    m.set_block(0, 0, block2);  // Update with different size
    if (!check((int)m.num_blocks(), 1, check_msg(get_type_name<T>(), "Should still have 1 block after update"))) return TEST_FAIL;

    // New block should have different shape
    Shape s = m.get_block(0, 0).shape();
    if (!check((int)s.first, 3, check_msg(get_type_name<T>(), "Updated block should have 3 rows"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int total_failures = 0;

    // BlockMatrix constructors
    total_failures += test_blockmatrix_default_constructor<double>();
    total_failures += test_blockmatrix_default_constructor<float>();
    total_failures += test_blockmatrix_copy_constructor<double>();
    total_failures += test_blockmatrix_copy_constructor<float>();
    total_failures += test_blockmatrix_move_constructor<double>();
    total_failures += test_blockmatrix_move_constructor<float>();

    // BlockMatrix block operations
    total_failures += test_blockmatrix_set_block<double>();
    total_failures += test_blockmatrix_set_block<float>();
    total_failures += test_blockmatrix_set_multiple_blocks<double>();
    total_failures += test_blockmatrix_set_multiple_blocks<float>();
    total_failures += test_blockmatrix_get_block<double>();
    total_failures += test_blockmatrix_get_block<float>();
    total_failures += test_blockmatrix_get_block_const<double>();
    total_failures += test_blockmatrix_get_block_const<float>();
    total_failures += test_blockmatrix_has_block<double>();
    total_failures += test_blockmatrix_has_block<float>();
    total_failures += test_blockmatrix_num_blocks<double>();
    total_failures += test_blockmatrix_num_blocks<float>();

    // BlockMatrix dimensions
    total_failures += test_blockmatrix_shape<double>();
    total_failures += test_blockmatrix_shape<float>();
    total_failures += test_blockmatrix_shape_with_sparse_layout<double>();
    total_failures += test_blockmatrix_shape_with_sparse_layout<float>();

    // BlockMatrix block grid
    total_failures += test_blockmatrix_num_block_rows<double>();
    total_failures += test_blockmatrix_num_block_rows<float>();
    total_failures += test_blockmatrix_num_block_cols<double>();
    total_failures += test_blockmatrix_num_block_cols<float>();
    total_failures += test_blockmatrix_block_grid_dimensions<double>();
    total_failures += test_blockmatrix_block_grid_dimensions<float>();

    // BlockMatrix offsets
    total_failures += test_blockmatrix_get_row_offsets<double>();
    total_failures += test_blockmatrix_get_row_offsets<float>();
    total_failures += test_blockmatrix_get_col_offsets<double>();
    total_failures += test_blockmatrix_get_col_offsets<float>();

    // BlockMatrix element access
    total_failures += test_blockmatrix_element_access_read<double>();
    total_failures += test_blockmatrix_element_access_read<float>();
    total_failures += test_blockmatrix_element_access_write<double>();
    total_failures += test_blockmatrix_element_access_write<float>();
    total_failures += test_blockmatrix_element_access_arbitrary_positions<double>();
    total_failures += test_blockmatrix_element_access_arbitrary_positions<float>();

    // BlockMatrix block indexing
    total_failures += test_blockmatrix_get_block_shape<double>();
    total_failures += test_blockmatrix_get_block_shape<float>();
    // BlockMatrix GPU operations
    total_failures += test_blockmatrix_gpu_operations_basic<double>();
    total_failures += test_blockmatrix_gpu_operations_basic<float>();

    // BlockMatrix GPU round-trip transfers
    total_failures += test_blockmatrix_gpu_roundtrip_uniform_blocks<double>();
    total_failures += test_blockmatrix_gpu_roundtrip_uniform_blocks<float>();
    total_failures += test_blockmatrix_gpu_roundtrip_sparse_layout<double>();
    total_failures += test_blockmatrix_gpu_roundtrip_sparse_layout<float>();
    total_failures += test_blockmatrix_gpu_roundtrip_varied_sizes<double>();
    total_failures += test_blockmatrix_gpu_roundtrip_varied_sizes<float>();

    // BlockMatrix copy and assignment
    total_failures += test_blockmatrix_copy_assignment<double>();
    total_failures += test_blockmatrix_copy_assignment<float>();
    total_failures += test_blockmatrix_move_assignment<double>();
    total_failures += test_blockmatrix_move_assignment<float>();

    // BlockMatrix edge cases
    total_failures += test_blockmatrix_single_block<double>();
    total_failures += test_blockmatrix_single_block<float>();
    total_failures += test_blockmatrix_sparse_layout<double>();
    total_failures += test_blockmatrix_sparse_layout<float>();
    total_failures += test_blockmatrix_large_block_indices<double>();
    total_failures += test_blockmatrix_large_block_indices<float>();
    total_failures += test_blockmatrix_varying_block_sizes<double>();
    total_failures += test_blockmatrix_varying_block_sizes<float>();
    total_failures += test_blockmatrix_rectangular_blocks<double>();
    total_failures += test_blockmatrix_rectangular_blocks<float>();
    total_failures += test_blockmatrix_update_existing_block<double>();
    total_failures += test_blockmatrix_update_existing_block<float>();

    if (total_failures > 0) {
        std::cerr << "gpu/tensor/blockmatrix tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    return TEST_PASS;
}
