#include "test_common.h"

#ifdef _CUDA

using namespace lahva;
using namespace lahva::gpu;

// ============================================================================
// BlockDiagMatrix Construction Tests
// ============================================================================

template <typename T>
int test_blockdiag_default_constructor() {
    int failures = 0;

    BlockDiagMatrix<T> m;

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

    return failures;
}

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
int test_blockdiag_uniform_blocks_with_value_constructor() {
    int failures = 0;

    size_t n_blocks = 2;
    Shape block_shape = Shape{3, 3};
    T init_val = (T)2.5;
    BlockDiagMatrix<T> m(n_blocks, block_shape, init_val);

    if (!check((int)m.num_blocks(), 2, "Should have 2 blocks")) {
        failures += 1;
    }

    // Check first block element
    double tol = get_tolerance<T>();
    if (!check((T)m(0, 0), init_val, tol, "First element should be initialized value")) {
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

template <typename T>
int test_blockdiag_varied_blocks_with_value_constructor() {
    int failures = 0;

    std::vector<Shape> shapes = {{2, 2}, {3, 3}};
    T init_val = (T)3.0;
    BlockDiagMatrix<T> m(shapes, init_val);

    if (!check((int)m.num_blocks(), 2, "Should have 2 blocks")) {
        failures += 1;
    }

    double tol = get_tolerance<T>();
    if (!check((T)m(0, 0), init_val, tol, "Block element should be initialized")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_from_matrices_constructor() {
    int failures = 0;

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, (T)1));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{3, 3}, (T)2));

    BlockDiagMatrix<T> m(blocks);

    if (!check((int)m.num_blocks(), 2, "Should have 2 blocks")) {
        failures += 1;
    }

    Shape s = m.shape();
    if (!check((int)s.first, 5, "Total rows should be 5")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_copy_constructor() {
    int failures = 0;

    BlockDiagMatrix<T> m1(2, Shape{3, 3}, (T)1.5);
    BlockDiagMatrix<T> m2 = m1;  // Copy constructor

    if (!check((int)m2.num_blocks(), 2, "Copy should have same number of blocks")) {
        failures += 1;
    }

    Shape s = m2.shape();
    if (!check((int)s.first, 6, "Copy should have same total rows")) {
        failures += 1;
    }

    double tol = get_tolerance<T>();
    if (!check((T)m2(0, 0), (T)1.5, tol, "Copy should have same data")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_move_constructor() {
    int failures = 0;

    BlockDiagMatrix<T> m1(2, Shape{3, 3}, (T)2.0);
    BlockDiagMatrix<T> m2 = std::move(m1);  // Move constructor

    if (!check((int)m2.num_blocks(), 2, "Moved matrix should have 2 blocks")) {
        failures += 1;
    }

    if (!check((int)m1.num_blocks(), 0, "Original should be empty after move")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// BlockDiagMatrix Block Operations Tests
// ============================================================================

template <typename T>
int test_blockdiag_get_block() {
    int failures = 0;

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {5, 6, 7, 8}));

    BlockDiagMatrix<T> m(blocks);

    const Matrix<T, CudaHostAllocator<T>>& block0 = m.get_block(0);
    Shape s = block0.shape();

    if (!check((int)s.first, 2, "First block should have 2 rows")) {
        failures += 1;
    }

    double tol = get_tolerance<T>();
    if (!check((T)block0(0, 0), (T)1, tol, "First block first element should be 1")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_add_block() {
    int failures = 0;

    BlockDiagMatrix<T> m;

    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    m.add_block(block1);

    if (!check((int)m.num_blocks(), 1, "Should have 1 block after adding")) {
        failures += 1;
    }

    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);
    m.add_block(block2);

    if (!check((int)m.num_blocks(), 2, "Should have 2 blocks after adding")) {
        failures += 1;
    }

    Shape s = m.shape();
    if (!check((int)s.first, 5, "Total rows should be 5")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_num_blocks() {
    int failures = 0;

    std::vector<Shape> shapes = {{2, 2}, {3, 3}, {4, 4}};
    BlockDiagMatrix<T> m(shapes);

    if (!check((int)m.num_blocks(), 3, "num_blocks should return 3")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_block_shapes() {
    int failures = 0;

    std::vector<Shape> shapes = {{2, 2}, {3, 3}, {1, 1}};
    BlockDiagMatrix<T> m(shapes);

    auto block_shapes = m.block_shapes();

    if (block_shapes.size() != 3) {
        failures += 1;
    }

    if (block_shapes[0][0] != 2 || block_shapes[0][1] != 2) {
        failures += 1;
    }

    if (block_shapes[1][0] != 3 || block_shapes[1][1] != 3) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// BlockDiagMatrix Dimension Tests
// ============================================================================

template <typename T>
int test_blockdiag_shape() {
    int failures = 0;

    std::vector<Shape> shapes = {{2, 2}, {3, 3}, {1, 1}};
    BlockDiagMatrix<T> m(shapes);

    Shape s = m.shape();
    if (!check((int)s.first, 6, "shape().first (rows) should be 6")) {
        failures += 1;
    }

    if (!check((int)s.second, 6, "shape().second (cols) should be 6")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_rows() {
    int failures = 0;

    std::vector<Shape> shapes = {{2, 2}, {3, 3}, {4, 4}};
    BlockDiagMatrix<T> m(shapes);

    Shape s = m.shape();
    if (!check((int)s.first, 9, "Total rows should be 2+3+4=9")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_cols() {
    int failures = 0;

    std::vector<Shape> shapes = {{2, 2}, {3, 3}, {4, 4}};
    BlockDiagMatrix<T> m(shapes);

    Shape s = m.shape();
    if (!check((int)s.second, 9, "Total cols should be 2+3+4=9")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// BlockDiagMatrix Offset Tests
// ============================================================================

template <typename T>
int test_blockdiag_get_row_offsets() {
    int failures = 0;

    std::vector<Shape> shapes = {{2, 2}, {3, 3}, {1, 1}};
    BlockDiagMatrix<T> m(shapes);

    const std::vector<int>& row_offsets = m.get_row_offsets();

    if (row_offsets.size() != 4) {  // n_blocks + 1
        failures += 1;
    }

    if (row_offsets[0] != 0) {
        failures += 1;
    }

    if (row_offsets[1] != 2) {
        failures += 1;
    }

    if (row_offsets[2] != 5) {
        failures += 1;
    }

    if (row_offsets[3] != 6) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_get_col_offsets() {
    int failures = 0;

    std::vector<Shape> shapes = {{2, 2}, {3, 3}, {1, 1}};
    BlockDiagMatrix<T> m(shapes);

    const std::vector<int>& col_offsets = m.get_col_offsets();

    if (col_offsets.size() != 4) {  // n_blocks + 1
        failures += 1;
    }

    if (col_offsets[0] != 0) {
        failures += 1;
    }

    if (col_offsets[1] != 2) {
        failures += 1;
    }

    if (col_offsets[2] != 5) {
        failures += 1;
    }

    if (col_offsets[3] != 6) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// BlockDiagMatrix Element Access Tests
// ============================================================================

template <typename T>
int test_blockdiag_element_access_read() {
    int failures = 0;

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {5, 6, 7, 8}));

    BlockDiagMatrix<T> m(blocks);

    double tol = get_tolerance<T>();

    // Test first block elements
    if (!check((T)m(0, 0), (T)1, tol, "Element (0,0) should be 1")) {
        failures += 1;
    }

    if (!check((T)m(0, 1), (T)3, tol, "Element (0,1) should be 3")) {
        failures += 1;
    }

    // Test second block elements
    if (!check((T)m(2, 2), (T)5, tol, "Element (2,2) should be 5")) {
        failures += 1;
    }

    // Test off-diagonal zeros
    if (!check((T)m(0, 2), (T)0, tol, "Element (0,2) should be 0")) {
        failures += 1;
    }

    if (!check((T)m(2, 0), (T)0, tol, "Element (2,0) should be 0")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_element_access_write() {
    int failures = 0;

    BlockDiagMatrix<T> m(2, Shape{3, 3});

    double tol = get_tolerance<T>();

    // Write to first block
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

// ============================================================================
// BlockDiagMatrix GPU Operations Tests
// ============================================================================

template <typename T>
int test_blockdiag_ensure_on_gpu() {
    int failures = 0;

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, (T)1));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, (T)2));

    BlockDiagMatrix<T> m(blocks);

    const GPUBlockDiagData<T>& gpu_data = m.ensure_on_gpu();

    if (gpu_data.d_data == nullptr) {
        failures += 1;
    }

    if (gpu_data.num_blocks != 2) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_is_on_gpu() {
    int failures = 0;

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, (T)1));

    BlockDiagMatrix<T> m(blocks);

    if (m.is_on_gpu()) {
        failures += 1;  // Initially should not be on GPU
    }

    m.ensure_on_gpu();

    if (!m.is_on_gpu()) {
        failures += 1;  // Should be on GPU after ensure_on_gpu
    }

    return failures;
}

template <typename T>
int test_blockdiag_free_gpu_cache() {
    int failures = 0;

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, (T)1));

    BlockDiagMatrix<T> m(blocks);

    m.ensure_on_gpu();

    if (!m.is_on_gpu()) {
        failures += 1;
    }

    m.free_gpu_cache();

    if (m.is_on_gpu()) {
        failures += 1;  // Should be freed
    }

    return failures;
}

template <typename T>
int test_blockdiag_to_gpu() {
    int failures = 0;

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, (T)1));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, (T)2));

    BlockDiagMatrix<T> m(blocks);

    CudaRuntime cudart;
    m.to_gpu(cudart);

    if (!m.is_on_gpu()) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// BlockDiagMatrix Sparse Conversion Tests
// ============================================================================

template <typename T>
int test_blockdiag_get_diagonal() {
    int failures = 0;

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 0, 0, 2}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {3, 0, 0, 4}));

    BlockDiagMatrix<T> m(blocks);
    cpu::Vector<T, CudaHostAllocator<T>> diag = m.get_diagonal();

    if (!check((int)diag.size(), 4, "Diagonal should have 4 elements")) {
        failures += 1;
    }

    double tol = get_tolerance<T>();

    if (!check((T)diag[0], (T)1, tol, "First diagonal element should be 1")) {
        failures += 1;
    }

    if (!check((T)diag[1], (T)2, tol, "Second diagonal element should be 2")) {
        failures += 1;
    }

    if (!check((T)diag[2], (T)3, tol, "Third diagonal element should be 3")) {
        failures += 1;
    }

    if (!check((T)diag[3], (T)4, tol, "Fourth diagonal element should be 4")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_set_diagonal() {
    int failures = 0;

    BlockDiagMatrix<T> m(2, Shape{2, 2});

    cpu::Vector<T, CudaHostAllocator<T>> diag({(T)5, (T)6, (T)7, (T)8});

    m.set_diagonal(diag);

    double tol = get_tolerance<T>();

    if (!check((T)m(0, 0), (T)5, tol, "First diagonal element should be 5")) {
        failures += 1;
    }

    if (!check((T)m(1, 1), (T)6, tol, "Second diagonal element should be 6")) {
        failures += 1;
    }

    if (!check((T)m(2, 2), (T)7, tol, "Third diagonal element should be 7")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_symmetrize() {
    int failures = 0;

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 3, 2, 4}));

    BlockDiagMatrix<T> m(blocks);
    m.symmetrize();

    double tol = get_tolerance<T>();

    if (!check((T)m(0, 0), (T)1, tol, "Element (0,0) should be 1")) {
        failures += 1;
    }

    if (!check((T)m(0, 1), (T)2.5, tol, "Element (0,1) should be 2.5")) {
        failures += 1;
    }

    if (!check((T)m(1, 0), (T)2.5, tol, "Element (1,0) should be 2.5")) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// BlockDiagMatrix Edge Cases Tests
// ============================================================================

template <typename T>
int test_blockdiag_single_block() {
    int failures = 0;

    BlockDiagMatrix<T> m(1, Shape{5, 5}, (T)1);

    if (!check((int)m.num_blocks(), 1, "Should have 1 block")) {
        failures += 1;
    }

    Shape s = m.shape();
    if (!check((int)s.first, 5, "Total rows should be 5")) {
        failures += 1;
    }

    double tol = get_tolerance<T>();
    if (!check((T)m(0, 0), (T)1, tol, "Element should be accessible")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_non_square_blocks() {
    int failures = 0;

    std::vector<Shape> shapes = {{2, 4}, {3, 2}};
    BlockDiagMatrix<T> m(shapes);

    Shape s = m.shape();
    if (!check((int)s.first, 5, "Total rows should be 5")) {
        failures += 1;
    }

    if (!check((int)s.second, 6, "Total cols should be 6")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_non_uniform_block_sizes() {
    int failures = 0;

    std::vector<Shape> shapes = {{1, 1}, {2, 2}, {4, 4}, {3, 3}};
    BlockDiagMatrix<T> m(shapes);

    if (!check((int)m.num_blocks(), 4, "Should have 4 blocks")) {
        failures += 1;
    }

    Shape s = m.shape();
    if (!check((int)s.first, 10, "Total rows should be 1+2+4+3=10")) {
        failures += 1;
    }

    if (!check((int)s.second, 10, "Total cols should be 1+2+4+3=10")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_copy_assignment() {
    int failures = 0;

    BlockDiagMatrix<T> m1(2, Shape{3, 3}, (T)1.5);
    BlockDiagMatrix<T> m2;

    m2 = m1;  // Copy assignment

    if (!check((int)m2.num_blocks(), 2, "Assignment should copy number of blocks")) {
        failures += 1;
    }

    Shape s = m2.shape();
    if (!check((int)s.first, 6, "Assignment should copy total rows")) {
        failures += 1;
    }

    double tol = get_tolerance<T>();
    if (!check((T)m2(0, 0), (T)1.5, tol, "Assignment should copy data")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_move_assignment() {
    int failures = 0;

    BlockDiagMatrix<T> m1(2, Shape{3, 3}, (T)2.0);
    BlockDiagMatrix<T> m2;

    m2 = std::move(m1);  // Move assignment

    if (!check((int)m2.num_blocks(), 2, "Move should transfer blocks")) {
        failures += 1;
    }

    if (!check((int)m1.num_blocks(), 0, "Original should be empty after move")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_get_block_shape() {
    int failures = 0;

    std::vector<Shape> shapes = {{2, 2}, {3, 3}, {1, 1}};
    BlockDiagMatrix<T> m(shapes);

    Shape block_shape = m.get_block_shape(0);
    if (!check((int)block_shape.first, 2, "Block 0 should have 2 rows")) {
        failures += 1;
    }

    block_shape = m.get_block_shape(1);
    if (!check((int)block_shape.first, 3, "Block 1 should have 3 rows")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_blockdiag_get_block_data() {
    int failures = 0;

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, (T)5));

    BlockDiagMatrix<T> m(blocks);

    const void* data_ptr = m.get_block_data(0);
    if (data_ptr == nullptr) {
        failures += 1;
    }

    return failures;
}

// ============================================================================
// GPU Round-Trip Transfer Tests (CPU -> GPU -> CPU with data verification)
// ============================================================================

template <typename T>
int test_sparse_matrix_gpu_roundtrip_csr() {
    int failures = 0;

    // Create block diagonal matrix with known values
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {5, 6, 7, 8}));

    BlockDiagMatrix<T> bdm(blocks);

    // Convert to sparse CSR format
    CudaRuntime cudart;
    gpu::SparseMatrix<T> sparse_csr(cudart, bdm, gpu::SparseFormat::CSR);

    // Reconstruct original dense matrix from BlockDiagMatrix
    Shape bdm_shape = bdm.shape();
    Matrix<T> original_dense(bdm_shape, (T)0);
    for (size_t i = 0; i < bdm_shape.first; ++i) {
        for (size_t j = 0; j < bdm_shape.second; ++j) {
            original_dense(i, j) = bdm(i, j);
        }
    }

    // Transfer to GPU and back
    sparse_csr.allocate_gpu_memory();
    sparse_csr.transfer_to_device(cudart);
    sparse_csr.transfer_to_host(cudart);

    // Reconstruct dense matrix from sparse data
    Matrix<T> reconstructed_dense = sparse_csr.to_dense();

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

    sparse_csr.release_gpu_memory();
    return failures;
}

template <typename T>
int test_sparse_matrix_gpu_roundtrip_bsr() {
    int failures = 0;

    // Create block diagonal matrix with known values
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{3, 3}, {10, 11, 12, 13, 14, 15, 16, 17, 18}));

    BlockDiagMatrix<T> bdm(blocks);

    // Convert to sparse BSR format
    CudaRuntime cudart;
    gpu::SparseMatrix<T> sparse_bsr(cudart, bdm, gpu::SparseFormat::BSR);

    // Reconstruct original dense matrix from BlockDiagMatrix
    Shape bdm_shape = bdm.shape();
    Matrix<T> original_dense(bdm_shape, (T)0);
    for (size_t i = 0; i < bdm_shape.first; ++i) {
        for (size_t j = 0; j < bdm_shape.second; ++j) {
            original_dense(i, j) = bdm(i, j);
        }
    }

    // Transfer to GPU and back
    sparse_bsr.allocate_gpu_memory();
    sparse_bsr.transfer_to_device(cudart);
    sparse_bsr.transfer_to_host(cudart);

    // Reconstruct dense matrix from sparse BSR data
    Matrix<T> reconstructed_dense = sparse_bsr.to_dense();

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

    sparse_bsr.release_gpu_memory();
    return failures;
}

template <typename T>
int test_sparse_matrix_gpu_roundtrip_varied_sizes() {
    int failures = 0;

    // Create blocks of different sizes
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{3, 3}, {5, 6, 7, 8, 9, 10, 11, 12, 13}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {14, 15, 16, 17}));

    BlockDiagMatrix<T> bdm(blocks);

    CudaRuntime cudart;
    gpu::SparseMatrix<T> sparse(cudart, bdm, gpu::SparseFormat::CSR);

    // Reconstruct original dense matrix from BlockDiagMatrix
    Shape bdm_shape = bdm.shape();
    Matrix<T> original_dense(bdm_shape, (T)0);
    for (size_t i = 0; i < bdm_shape.first; ++i) {
        for (size_t j = 0; j < bdm_shape.second; ++j) {
            original_dense(i, j) = bdm(i, j);
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
// Main Test Entry Point
// ============================================================================

int main() {
    int failures = 0;

    std::cout << "Testing BlockDiagMatrix<double> constructors..." << std::endl;
    failures += test_blockdiag_default_constructor<double>();
    failures += test_blockdiag_uniform_blocks_constructor<double>();
    failures += test_blockdiag_uniform_blocks_with_value_constructor<double>();
    failures += test_blockdiag_varied_blocks_constructor<double>();
    failures += test_blockdiag_varied_blocks_with_value_constructor<double>();
    failures += test_blockdiag_from_matrices_constructor<double>();
    failures += test_blockdiag_copy_constructor<double>();
    failures += test_blockdiag_move_constructor<double>();

    std::cout << "Testing BlockDiagMatrix<float> constructors..." << std::endl;
    failures += test_blockdiag_default_constructor<float>();
    failures += test_blockdiag_uniform_blocks_constructor<float>();
    failures += test_blockdiag_varied_blocks_constructor<float>();

    std::cout << "Testing BlockDiagMatrix<double> block operations..." << std::endl;
    failures += test_blockdiag_get_block<double>();
    failures += test_blockdiag_add_block<double>();
    failures += test_blockdiag_num_blocks<double>();
    failures += test_blockdiag_block_shapes<double>();

    std::cout << "Testing BlockDiagMatrix<double> dimensions..." << std::endl;
    failures += test_blockdiag_shape<double>();
    failures += test_blockdiag_rows<double>();
    failures += test_blockdiag_cols<double>();

    std::cout << "Testing BlockDiagMatrix<double> offsets..." << std::endl;
    failures += test_blockdiag_get_row_offsets<double>();
    failures += test_blockdiag_get_col_offsets<double>();

    std::cout << "Testing BlockDiagMatrix<double> element access..." << std::endl;
    failures += test_blockdiag_element_access_read<double>();
    failures += test_blockdiag_element_access_write<double>();

    std::cout << "Testing BlockDiagMatrix<double> GPU operations..." << std::endl;
    failures += test_blockdiag_ensure_on_gpu<double>();
    failures += test_blockdiag_is_on_gpu<double>();
    failures += test_blockdiag_free_gpu_cache<double>();
    failures += test_blockdiag_to_gpu<double>();

    std::cout << "Testing SparseMatrix<double> GPU round-trip transfers..." << std::endl;
    failures += test_sparse_matrix_gpu_roundtrip_csr<double>();
    failures += test_sparse_matrix_gpu_roundtrip_bsr<double>();
    failures += test_sparse_matrix_gpu_roundtrip_varied_sizes<double>();

    std::cout << "Testing SparseMatrix<float> GPU round-trip transfers..." << std::endl;
    failures += test_sparse_matrix_gpu_roundtrip_csr<float>();
    failures += test_sparse_matrix_gpu_roundtrip_bsr<float>();
    failures += test_sparse_matrix_gpu_roundtrip_varied_sizes<float>();

    std::cout << "Testing BlockDiagMatrix<double> diagonal operations..." << std::endl;
    failures += test_blockdiag_get_diagonal<double>();
    failures += test_blockdiag_set_diagonal<double>();
    failures += test_blockdiag_symmetrize<double>();

    std::cout << "Testing BlockDiagMatrix<double> edge cases..." << std::endl;
    failures += test_blockdiag_single_block<double>();
    failures += test_blockdiag_non_square_blocks<double>();
    failures += test_blockdiag_non_uniform_block_sizes<double>();
    failures += test_blockdiag_copy_assignment<double>();
    failures += test_blockdiag_move_assignment<double>();
    failures += test_blockdiag_get_block_shape<double>();
    failures += test_blockdiag_get_block_data<double>();

    if (failures == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "All BlockDiagMatrix tests PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
    } else {
        std::cout << "\n========================================" << std::endl;
        std::cout << "BlockDiagMatrix tests FAILED with " << failures << " failures!" << std::endl;
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
