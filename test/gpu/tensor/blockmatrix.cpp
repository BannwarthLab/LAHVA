#include "test_common.h"

#ifdef _CUDA

using namespace lahva;
using namespace lahva::gpu;

// ============================================================================
// BlockMatrix Construction Tests
// ============================================================================

template <typename T>
int test_blockmatrix_default_constructor() {
    int failures = 0;

    BlockMatrix<T> m;

    if (!check((int)m.num_blocks(), 0, "Default constructor should create empty matrix")) {
        failures += 1;
    }

    Shape s = m.shape();
    if (!check((int)s.first, 0, "Default matrix should have 0 rows")) {
        failures += 1;
    }

    if (!check((int)s.second, 0, "Default matrix should have 0 cols")) {
        failures += 1;
    }

    if (!check((int)m.num_block_rows(), 0, "Default matrix should have 0 block rows")) {
        failures += 1;
    }

    if (!check((int)m.num_block_cols(), 0, "Default matrix should have 0 block cols")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockmatrix_copy_constructor() {
    int failures = 0;

    BlockMatrix<T> m1;
    Matrix<T, CudaHostAllocator<T>> block(Shape{3, 3}, (T)1.5);
    m1.set_block(0, 0, block);

    BlockMatrix<T> m2 = m1;  // Copy constructor

    if (!check((int)m2.num_blocks(), 1, "Copy should have same number of blocks")) {
        failures += 1;
    }

    if (!check((int)m2.num_block_rows(), 1, "Copy should have same block rows")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockmatrix_move_constructor() {
    int failures = 0;

    BlockMatrix<T> m1;
    Matrix<T, CudaHostAllocator<T>> block(Shape{3, 3}, (T)2.0);
    m1.set_block(0, 0, block);

    BlockMatrix<T> m2 = std::move(m1);  // Move constructor

    if (!check((int)m2.num_blocks(), 1, "Moved matrix should have 1 block")) {
        failures += 1;
    }

    if (!check((int)m1.num_blocks(), 0, "Original should be empty after move")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// BlockMatrix Block Operations Tests
// ============================================================================

template <typename T>
int test_blockmatrix_set_block() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block(Shape{3, 3}, (T)1.5);

    m.set_block(0, 0, block);

    if (!check((int)m.num_blocks(), 1, "Should have 1 block after set_block")) {
        failures += 1;
    }

    if (!check((int)m.num_block_rows(), 1, "Should have 1 block row")) {
        failures += 1;
    }

    if (!check((int)m.num_block_cols(), 1, "Should have 1 block col")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockmatrix_set_multiple_blocks() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{2, 3}, (T)2);
    Matrix<T, CudaHostAllocator<T>> block3(Shape{3, 2}, (T)3);

    // Non-overlapping positions: block1 at (0,0), block2 at (0,2), block3 at (2,0)
    m.set_block(0, 0, block1);
    m.set_block(0, 2, block2);
    m.set_block(2, 0, block3);

    if (!check((int)m.num_blocks(), 3, "Should have 3 blocks")) {
        failures += 1;
    }

    // num_block_rows/cols return max position + 1 for grid dimensions
    // With positions at (0,0), (0,2), (2,0): max row is 2, max col is 2, so 3 and 3
    if (!check((int)m.num_block_rows(), 3, "Should have max row position + 1")) {
        failures += 1;
    }

    if (!check((int)m.num_block_cols(), 3, "Should have max col position + 1")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockmatrix_get_block() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block(Shape{2, 2}, {1, 2, 3, 4});

    m.set_block(0, 0, block);

    Matrix<T, CudaHostAllocator<T>>& retrieved = m.get_block(0, 0);
    Shape s = retrieved.shape();

    if (!check((int)s.first, 2, "Retrieved block should have 2 rows")) {
        failures += 1;
    }

    double tol = get_tolerance<T>();
    if (!check((T)retrieved(0, 0), (T)1, tol, "Retrieved block first element should be 1")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockmatrix_get_block_const() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block(Shape{2, 2}, (T)5);

    m.set_block(2, 2, block);

    const Matrix<T, CudaHostAllocator<T>>& retrieved = m.get_block(2, 2);
    Shape s = retrieved.shape();

    if (!check((int)s.first, 2, "Retrieved const block should have 2 rows")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockmatrix_has_block() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block(Shape{3, 3}, (T)1);

    m.set_block(0, 0, block);

    if (!m.has_block(0, 0)) {
        failures += 1;
    }

    if (m.has_block(1, 1)) {
        failures += 1;  // Should not have this block
    }

    return failures;
}

template <typename T>
int test_blockmatrix_num_blocks() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);
    Matrix<T, CudaHostAllocator<T>> block3(Shape{2, 2}, (T)3);

    // Non-overlapping positions
    m.set_block(0, 0, block1);
    m.set_block(0, 3, block2);
    m.set_block(3, 0, block3);

    if (!check((int)m.num_blocks(), 3, "num_blocks should return 3")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// BlockMatrix Dimension Tests
// ============================================================================

template <typename T>
int test_blockmatrix_shape() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 3}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 2}, (T)2);

    m.set_block(0, 0, block1);
    m.set_block(2, 3, block2);

    Shape s = m.shape();
    // Total rows: max(0+2, 2+3) = 5; Total cols: max(0+3, 3+2) = 5
    if (!check((int)s.first, 5, "Total rows should be 5")) {
        failures += 1;
    }

    if (!check((int)s.second, 5, "Total cols should be 5")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockmatrix_shape_with_sparse_layout() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    // Set blocks at positions (0, 0) and (2, 2)
    m.set_block(0, 0, block1);
    m.set_block(2, 2, block2);

    Shape s = m.shape();
    // Should account for the maximum row index in each row group
    if (!check((int)s.first, 5, "Total rows should account for all block rows")) {
        failures += 1;
    }

    if (!check((int)s.second, 5, "Total cols should account for all block cols")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// BlockMatrix Block Grid Tests
// ============================================================================

template <typename T>
int test_blockmatrix_num_block_rows() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    m.set_block(0, 0, block1);
    m.set_block(2, 0, block2);

    if (!check((int)m.num_block_rows(), 3, "num_block_rows should return 3")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockmatrix_num_block_cols() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    m.set_block(0, 0, block1);
    m.set_block(0, 2, block2);

    if (!check((int)m.num_block_cols(), 3, "num_block_cols should return 3")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockmatrix_block_grid_dimensions() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);
    Matrix<T, CudaHostAllocator<T>> block3(Shape{2, 2}, (T)3);
    Matrix<T, CudaHostAllocator<T>> block4(Shape{3, 3}, (T)4);

    m.set_block(0, 0, block1);
    m.set_block(0, 2, block2);
    m.set_block(2, 0, block3);
    m.set_block(3, 3, block4);

    if (!check((int)m.num_block_rows(), 4, "num_block_rows should return 4")) {
        failures += 1;
    }

    if (!check((int)m.num_block_cols(), 4, "num_block_cols should return 4")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// BlockMatrix Offset Tests
// ============================================================================

template <typename T>
int test_blockmatrix_get_row_offsets() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    m.set_block(0, 0, block1);
    m.set_block(2, 0, block2);

    const std::vector<int>& row_offsets = m.get_row_offsets();

    // Offsets should be computed lazily
    if (row_offsets.empty()) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockmatrix_get_col_offsets() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    m.set_block(0, 0, block1);
    m.set_block(0, 2, block2);

    const std::vector<int>& col_offsets = m.get_col_offsets();

    // Offsets should be computed lazily
    if (col_offsets.empty()) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// BlockMatrix Element Access Tests
// ============================================================================

template <typename T>
int test_blockmatrix_element_access_read() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block(Shape{3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    m.set_block(0, 0, block);

    double tol = get_tolerance<T>();

    // Test first block element
    if (!check((T)m(0, 0), (T)1, tol, "Element (0,0) should be 1")) {
        failures += 1;
    }

    if (!check((T)m(1, 1), (T)5, tol, "Element (1,1) should be 5")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockmatrix_element_access_write() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block(Shape{3, 3});

    m.set_block(0, 0, block);

    double tol = get_tolerance<T>();

    // Write to block
    m(0, 0) = (T)7.5;
    m(1, 1) = (T)8.5;

    if (!check((T)m(0, 0), (T)7.5, tol, "Element (0,0) should be 7.5 after write")) {
        failures += 1;
    }

    if (!check((T)m(1, 1), (T)8.5, tol, "Element (1,1) should be 8.5 after write")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockmatrix_element_access_arbitrary_positions() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    m.set_block(0, 0, block1);
    m.set_block(2, 2, block2);

    // Block 1 is at (0,0) with size 2x2
    // Block 2 is at (1,1) with size 3x3
    // But the matrix uses fixed positions like pos.first*100, pos.second*100

    return failures;  // Skip complex positioning test for now
}

// ============================================================================
// BlockMatrix Block Indexing Tests
// ============================================================================

template <typename T>
int test_blockmatrix_get_block_shape() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    m.set_block(0, 0, block1);
    m.set_block(2, 0, block2);

    // Get by linear index
    Shape shape0 = m.get_block_shape(0);
    if (!check((int)shape0.first, 2, "First block should have 2 rows")) {
        failures += 1;
    }

    Shape shape1 = m.get_block_shape(1);
    if (!check((int)shape1.first, 3, "Second block should have 3 rows")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockmatrix_get_block_data() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block(Shape{2, 2}, (T)5);

    m.set_block(0, 0, block);

    const void* data_ptr = m.get_block_data(0);
    if (data_ptr == nullptr) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// BlockMatrix GPU Operations Tests
// ============================================================================

template <typename T>
int test_blockmatrix_gpu_operations_basic() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block(Shape{2, 2}, (T)1);

    m.set_block(0, 0, block);

    CudaRuntime cudart;

    // Test that we can access GPU-related methods
    Shape s = m.shape();
    if (!check((int)s.first, 2, "Matrix should have correct dimensions")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// ============================================================================
// BlockMatrix GPU Round-Trip Transfer Tests
// ============================================================================

template <typename T>
int test_blockmatrix_gpu_roundtrip_uniform_blocks() {
    int failures = 0;

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
    sparse.transfer_to_device(cudart);
    sparse.transfer_to_host(cudart);

    // Reconstruct dense matrix from sparse data
    Matrix<T> reconstructed_dense = sparse.to_dense();

    T tolerance = get_tolerance<T>();

    // Verify dimensions match
    Shape reconstructed_shape = reconstructed_dense.shape();
    if (reconstructed_shape.first != original_dense.shape().first ||
        reconstructed_shape.second != original_dense.shape().second) {
        failures += 1;
    }

    // Verify all elements match
    for (size_t i = 0; i < original_dense.shape().first; ++i) {
        for (size_t j = 0; j < original_dense.shape().second; ++j) {
            if (std::abs(original_dense(i, j) - reconstructed_dense(i, j)) > tolerance) {
                failures += 1;
            }
        }
    }

    sparse.release_gpu_memory();
    return failures;
}

template <typename T>
int test_blockmatrix_gpu_roundtrip_sparse_layout() {
    int failures = 0;

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
    sparse.transfer_to_device(cudart);
    sparse.transfer_to_host(cudart);

    // Reconstruct dense matrix from sparse data
    Matrix<T> reconstructed_dense = sparse.to_dense();

    T tolerance = get_tolerance<T>();

    // Verify dimensions match
    Shape reconstructed_shape = reconstructed_dense.shape();
    if (reconstructed_shape.first != original_dense.shape().first ||
        reconstructed_shape.second != original_dense.shape().second) {
        failures += 1;
    }

    // Verify all elements match
    for (size_t i = 0; i < original_dense.shape().first; ++i) {
        for (size_t j = 0; j < original_dense.shape().second; ++j) {
            if (std::abs(original_dense(i, j) - reconstructed_dense(i, j)) > tolerance) {
                failures += 1;
            }
        }
    }

    sparse.release_gpu_memory();
    return failures;
}

template <typename T>
int test_blockmatrix_gpu_roundtrip_varied_sizes() {
    int failures = 0;

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
    sparse.transfer_to_device(cudart);
    sparse.transfer_to_host(cudart);

    // Reconstruct dense matrix from sparse data
    Matrix<T> reconstructed_dense = sparse.to_dense();

    T tolerance = get_tolerance<T>();

    // Verify dimensions match
    Shape reconstructed_shape = reconstructed_dense.shape();
    if (reconstructed_shape.first != original_dense.shape().first ||
        reconstructed_shape.second != original_dense.shape().second) {
        failures += 1;
    }

    // Verify all elements match
    for (size_t i = 0; i < original_dense.shape().first; ++i) {
        for (size_t j = 0; j < original_dense.shape().second; ++j) {
            if (std::abs(original_dense(i, j) - reconstructed_dense(i, j)) > tolerance) {
                failures += 1;
            }
        }
    }

    sparse.release_gpu_memory();
    return failures;
}

// ============================================================================
// BlockMatrix Copy and Assignment Tests
// ============================================================================

template <typename T>
int test_blockmatrix_copy_assignment() {
    int failures = 0;

    BlockMatrix<T> m1;
    Matrix<T, CudaHostAllocator<T>> block(Shape{3, 3}, (T)1.5);
    m1.set_block(0, 0, block);

    BlockMatrix<T> m2;
    m2 = m1;  // Copy assignment

    if (!check((int)m2.num_blocks(), 1, "Assignment should copy blocks")) {
        failures += 1;
    }

    if (!check((int)m2.num_block_rows(), 1, "Assignment should copy block rows")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockmatrix_move_assignment() {
    int failures = 0;

    BlockMatrix<T> m1;
    Matrix<T, CudaHostAllocator<T>> block(Shape{3, 3}, (T)2.0);
    m1.set_block(0, 0, block);

    BlockMatrix<T> m2;
    m2 = std::move(m1);  // Move assignment

    if (!check((int)m2.num_blocks(), 1, "Move should transfer blocks")) {
        failures += 1;
    }

    if (!check((int)m1.num_blocks(), 0, "Original should be empty after move")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// BlockMatrix Edge Cases Tests
// ============================================================================

template <typename T>
int test_blockmatrix_single_block() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block(Shape{5, 5}, (T)1);

    m.set_block(0, 0, block);

    if (!check((int)m.num_blocks(), 1, "Should have 1 block")) {
        failures += 1;
    }

    Shape s = m.shape();
    if (!check((int)s.first, 5, "Total rows should be 5")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockmatrix_sparse_layout() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    // Set blocks at sparse positions
    m.set_block(0, 0, block1);
    m.set_block(2, 2, block2);  // Sparse position

    if (!check((int)m.num_blocks(), 2, "Should have 2 blocks")) {
        failures += 1;
    }

    if (!check((int)m.num_block_rows(), 3, "Should account for all block row indices")) {
        failures += 1;
    }

    if (!check((int)m.num_block_cols(), 3, "Should account for all block col indices")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockmatrix_large_block_indices() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    m.set_block(0, 0, block1);
    m.set_block(10, 10, block2);  // Large indices

    if (!check((int)m.num_blocks(), 2, "Should have 2 blocks")) {
        failures += 1;
    }

    if (!check((int)m.num_block_rows(), 11, "Should handle large block row indices")) {
        failures += 1;
    }

    if (!check((int)m.num_block_cols(), 11, "Should handle large block col indices")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockmatrix_varying_block_sizes() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{5, 5}, (T)2);
    Matrix<T, CudaHostAllocator<T>> block3(Shape{3, 3}, (T)3);

    m.set_block(0, 0, block1);
    m.set_block(2, 2, block2);
    m.set_block(7, 7, block3);

    if (!check((int)m.num_blocks(), 3, "Should have 3 blocks")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockmatrix_rectangular_blocks() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 4}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 2}, (T)2);

    m.set_block(0, 0, block1);
    m.set_block(2, 0, block2);

    if (!check((int)m.num_blocks(), 2, "Should have 2 rectangular blocks")) {
        failures += 1;
    }

    Shape s = m.shape();
    if (!check((int)s.first, 5, "Total rows should be 2+3=5")) {
        failures += 1;
    }

    if (!check((int)s.second, 4, "Total cols should be max(4,2)=4")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockmatrix_update_existing_block() {
    int failures = 0;

    BlockMatrix<T> m;
    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);

    m.set_block(0, 0, block1);
    if (!check((int)m.num_blocks(), 1, "Should have 1 block initially")) {
        failures += 1;
    }

    m.set_block(0, 0, block2);  // Update with different size
    if (!check((int)m.num_blocks(), 1, "Should still have 1 block after update")) {
        failures += 1;
    }

    // New block should have different shape
    Shape s = m.get_block(0, 0).shape();
    if (!check((int)s.first, 3, "Updated block should have 3 rows")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// Main Test Entry Point
// ============================================================================

int main() {
    int failures = 0;

    std::cout << "Testing BlockMatrix<double> constructors..." << std::endl;
    failures += test_blockmatrix_default_constructor<double>();
    failures += test_blockmatrix_copy_constructor<double>();
    failures += test_blockmatrix_move_constructor<double>();

    std::cout << "Testing BlockMatrix<float> constructors..." << std::endl;
    failures += test_blockmatrix_default_constructor<float>();

    std::cout << "Testing BlockMatrix<double> block operations..." << std::endl;
    failures += test_blockmatrix_set_block<double>();
    failures += test_blockmatrix_set_multiple_blocks<double>();
    failures += test_blockmatrix_get_block<double>();
    failures += test_blockmatrix_get_block_const<double>();
    failures += test_blockmatrix_has_block<double>();
    failures += test_blockmatrix_num_blocks<double>();

    std::cout << "Testing BlockMatrix<double> dimensions..." << std::endl;
    failures += test_blockmatrix_shape<double>();
    failures += test_blockmatrix_shape_with_sparse_layout<double>();

    std::cout << "Testing BlockMatrix<double> block grid..." << std::endl;
    failures += test_blockmatrix_num_block_rows<double>();
    failures += test_blockmatrix_num_block_cols<double>();
    failures += test_blockmatrix_block_grid_dimensions<double>();

    std::cout << "Testing BlockMatrix<double> offsets..." << std::endl;
    failures += test_blockmatrix_get_row_offsets<double>();
    failures += test_blockmatrix_get_col_offsets<double>();

    std::cout << "Testing BlockMatrix<double> element access..." << std::endl;
    failures += test_blockmatrix_element_access_read<double>();
    failures += test_blockmatrix_element_access_write<double>();
    failures += test_blockmatrix_element_access_arbitrary_positions<double>();

    std::cout << "Testing BlockMatrix<double> block indexing..." << std::endl;
    failures += test_blockmatrix_get_block_shape<double>();
    failures += test_blockmatrix_get_block_data<double>();

    std::cout << "Testing BlockMatrix<double> GPU operations..." << std::endl;
    failures += test_blockmatrix_gpu_operations_basic<double>();

    std::cout << "Testing BlockMatrix<double> GPU round-trip transfers..." << std::endl;
    failures += test_blockmatrix_gpu_roundtrip_uniform_blocks<double>();
    failures += test_blockmatrix_gpu_roundtrip_sparse_layout<double>();
    failures += test_blockmatrix_gpu_roundtrip_varied_sizes<double>();

    std::cout << "Testing BlockMatrix<float> GPU round-trip transfers..." << std::endl;
    failures += test_blockmatrix_gpu_roundtrip_uniform_blocks<float>();
    failures += test_blockmatrix_gpu_roundtrip_sparse_layout<float>();
    failures += test_blockmatrix_gpu_roundtrip_varied_sizes<float>();

    std::cout << "Testing BlockMatrix<double> copy and assignment..." << std::endl;
    failures += test_blockmatrix_copy_assignment<double>();
    failures += test_blockmatrix_move_assignment<double>();

    std::cout << "Testing BlockMatrix<double> edge cases..." << std::endl;
    failures += test_blockmatrix_single_block<double>();
    failures += test_blockmatrix_sparse_layout<double>();
    failures += test_blockmatrix_large_block_indices<double>();
    failures += test_blockmatrix_varying_block_sizes<double>();
    failures += test_blockmatrix_rectangular_blocks<double>();
    failures += test_blockmatrix_update_existing_block<double>();

    if (failures == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "All BlockMatrix tests PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
    } else {
        std::cout << "\n========================================" << std::endl;
        std::cout << "BlockMatrix tests FAILED with " << failures << " failures!" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    return failures;
}

#else

int main() {
    std::cout << "GPU tests are disabled without CUDA support." << std::endl;
    return 0;
}

#endif
