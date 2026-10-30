#include "test_common.h"

using namespace lahva;
using namespace lahva::gpu;

// ============================================================================
// BlockDiagMatrix Construction Tests
// ============================================================================

template <typename T>
int test_blockdiag_default_constructor() {
    BlockDiagMatrix<T> m;

    if (!check((int)m.num_blocks(), 0, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    Shape s = m.shape();
    if (!check((int)s.first, 0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((int)s.second, 0, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

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
int test_blockdiag_uniform_blocks_with_value_constructor() {
    size_t n_blocks = 2;
    Shape block_shape = Shape{3, 3};
    T init_val = (T)2.5;
    BlockDiagMatrix<T> m(n_blocks, block_shape, init_val);

    if (!check((int)m.num_blocks(), 2, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((T)m(0, 0), init_val, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

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
    if (block_shapes.size() != 3) {
        std::cerr << check_msg(get_type_name<T>(), "check 4") << std::endl;
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_varied_blocks_with_value_constructor() {
    std::vector<Shape> shapes = {{2, 2}, {3, 3}};
    T init_val = (T)3.0;
    BlockDiagMatrix<T> m(shapes, init_val);

    if (!check((int)m.num_blocks(), 2, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((T)m(0, 0), init_val, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_from_matrices_constructor() {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, (T)1));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{3, 3}, (T)2));

    BlockDiagMatrix<T> m(blocks);

    if (!check((int)m.num_blocks(), 2, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    Shape s = m.shape();
    if (!check((int)s.first, 5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_copy_constructor() {
    BlockDiagMatrix<T> m1(2, Shape{3, 3}, (T)1.5);
    BlockDiagMatrix<T> m2 = m1;  // Copy constructor

    if (!check((int)m2.num_blocks(), 2, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    Shape s = m2.shape();
    if (!check((int)s.first, 6, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((T)m2(0, 0), (T)1.5, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_move_constructor() {
    BlockDiagMatrix<T> m1(2, Shape{3, 3}, (T)2.0);
    BlockDiagMatrix<T> m2 = std::move(m1);  // Move constructor

    if (!check((int)m2.num_blocks(), 2, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((int)m1.num_blocks(), 0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// BlockDiagMatrix Block Operations Tests
// ============================================================================

template <typename T>
int test_blockdiag_get_block() {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {5, 6, 7, 8}));

    BlockDiagMatrix<T> m(blocks);

    const Matrix<T, CudaHostAllocator<T>>& block0 = m.get_block(0);
    Shape s = block0.shape();

    if (!check((int)s.first, 2, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((T)block0(0, 0), (T)1, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_add_block() {
    BlockDiagMatrix<T> m;

    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    m.add_block(block1);

    if (!check((int)m.num_blocks(), 1, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    Matrix<T, CudaHostAllocator<T>> block2(Shape{3, 3}, (T)2);
    m.add_block(block2);

    if (!check((int)m.num_blocks(), 2, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    Shape s = m.shape();
    if (!check((int)s.first, 5, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_add_block_at_index() {
    BlockDiagMatrix<T> m;

    Matrix<T, CudaHostAllocator<T>> block1(Shape{2, 2}, (T)1);
    m.add_block(block1);
    Matrix<T, CudaHostAllocator<T>> block3(Shape{3, 3}, (T)3);
    m.add_block(block3);

    if (!check((int)m.num_blocks(), 2, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    Matrix<T, CudaHostAllocator<T>> block2(Shape{1, 1}, (T)2);
    m.add_block(block2, 1);

    if (!check((int)m.num_blocks(), 3, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    Shape s = m.shape();
    if (!check((int)s.first, 6, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check((int)s.second, 6, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;
    if (!check((int)m.get_block_row(0), 0, check_msg(get_type_name<T>(), "check 5"))) return TEST_FAIL;
    if (!check((int)m.get_block_col(0), 0, check_msg(get_type_name<T>(), "check 6"))) return TEST_FAIL;
    if (!check((int)m.get_block_row(1), 2, check_msg(get_type_name<T>(), "check 7"))) return TEST_FAIL;
    if (!check((int)m.get_block_col(1), 2, check_msg(get_type_name<T>(), "check 8"))) return TEST_FAIL;
    if (!check((int)m.get_block_row(2), 3, check_msg(get_type_name<T>(), "check 9"))) return TEST_FAIL;
    if (!check((int)m.get_block_col(2), 3, check_msg(get_type_name<T>(), "check 10"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_num_blocks() {
    std::vector<Shape> shapes = {{2, 2}, {3, 3}, {4, 4}};
    BlockDiagMatrix<T> m(shapes);

    if (!check((int)m.num_blocks(), 3, check_msg(get_type_name<T>(), ""))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_block_shapes() {
    std::vector<Shape> shapes = {{2, 2}, {3, 3}, {1, 1}};
    BlockDiagMatrix<T> m(shapes);

    auto block_shapes = m.block_shapes();

    if (!check((int)block_shapes.size(), 3, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    if (!check((int)block_shapes[0][0], 2, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((int)block_shapes[0][1], 2, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    if (!check((int)block_shapes[1][0], 3, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;
    if (!check((int)block_shapes[1][1], 3, check_msg(get_type_name<T>(), "check 5"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// BlockDiagMatrix Dimension Tests
// ============================================================================

template <typename T>
int test_blockdiag_shape() {
    std::vector<Shape> shapes = {{2, 2}, {3, 3}, {1, 1}};
    BlockDiagMatrix<T> m(shapes);

    Shape s = m.shape();
    if (!check((int)s.first, 6, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((int)s.second, 6, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_rows() {
    std::vector<Shape> shapes = {{2, 2}, {3, 3}, {4, 4}};
    BlockDiagMatrix<T> m(shapes);

    Shape s = m.shape();
    if (!check((int)s.first, 9, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_cols() {
    std::vector<Shape> shapes = {{2, 2}, {3, 3}, {4, 4}};
    BlockDiagMatrix<T> m(shapes);

    Shape s = m.shape();
    if (!check((int)s.second, 9, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// BlockDiagMatrix Offset Tests
// ============================================================================

template <typename T>
int test_blockdiag_get_row_offsets() {
    std::vector<Shape> shapes = {{2, 2}, {3, 3}, {1, 1}};
    BlockDiagMatrix<T> m(shapes);

    const std::vector<int>& row_offsets = m.get_row_offsets();

    if (!check((int)row_offsets.size(), 4, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((int)row_offsets[0], 0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((int)row_offsets[1], 2, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check((int)row_offsets[2], 5, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;
    if (!check((int)row_offsets[3], 6, check_msg(get_type_name<T>(), "check 5"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_get_col_offsets() {
    std::vector<Shape> shapes = {{2, 2}, {3, 3}, {1, 1}};
    BlockDiagMatrix<T> m(shapes);

    const std::vector<int>& col_offsets = m.get_col_offsets();

    if (!check((int)col_offsets.size(), 4, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((int)col_offsets[0], 0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((int)col_offsets[1], 2, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check((int)col_offsets[2], 5, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;
    if (!check((int)col_offsets[3], 6, check_msg(get_type_name<T>(), "check 5"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// BlockDiagMatrix Element Access Tests
// ============================================================================

template <typename T>
int test_blockdiag_element_access_read() {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {5, 6, 7, 8}));

    BlockDiagMatrix<T> m(blocks);

    if (!check((T)m(0, 0), (T)1, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((T)m(0, 1), (T)3, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((T)m(2, 2), (T)5, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check((T)m(0, 2), (T)0, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;
    if (!check((T)m(2, 0), (T)0, check_msg(get_type_name<T>(), "check 5"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_element_access_write() {
    BlockDiagMatrix<T> m(2, Shape{3, 3});

    m(0, 0) = (T)7.5;
    m(1, 1) = (T)8.5;

    if (!check((T)m(0, 0), (T)7.5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((T)m(1, 1), (T)8.5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// BlockDiagMatrix GPU Operations Tests
// ============================================================================

template <typename T>
int test_blockdiag_ensure_on_gpu(const CudaRuntime& cudart) {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, (T)1));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, (T)2));

    BlockDiagMatrix<T> m(blocks);

    const GPUBlockDiagData<T>& gpu_data = m.ensure_on_gpu(cudart);

    if (gpu_data.d_data == nullptr) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    }
    if (!check((int)gpu_data.num_blocks, 2, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_ensure_on_gpu_without_blocks(const CudaRuntime& cudart) {
    const BlockDiagMatrix<T> m;

    try {
        m.ensure_on_gpu(cudart);
    } catch (const std::invalid_argument &) {
        return TEST_PASS;
    }

    std::cerr << check_msg(get_type_name<T>(), "expected std::invalid_argument") << std::endl;
    return TEST_FAIL;
}

// Other processes on the GPU can disturb the measured memory, so this only fails when
// nearly all of the repeated caches are missing from the free memory
template <typename T>
int test_blockdiag_repeated_copy2device_frees_cache(const CudaRuntime& cudart) {
    const size_t n = 4096;
    const size_t cache_bytes = n * n * sizeof(T);
    const int n_repeats = 5;

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{n, n}, (T)1));
    const BlockDiagMatrix<T> m(blocks);

    size_t free_before = 0;
    size_t free_after = 0;
    size_t total = 0;

    m.copy2device(cudart);
    get_cuda_error(cudaMemGetInfo(&free_before, &total));

    for (int i = 0; i < n_repeats; i++) {
        m.copy2device(cudart);
    }
    get_cuda_error(cudaMemGetInfo(&free_after, &total));

    const size_t lost = (free_before > free_after) ? free_before - free_after : 0;
    if (lost >= (n_repeats - 1) * cache_bytes) {
        std::cerr << check_msg(get_type_name<T>(), "device memory leaked: ") << lost << " bytes" << std::endl;
        return TEST_FAIL;
    }

    return TEST_PASS;
}

// ============================================================================
// BlockDiagMatrix Sparse Conversion Tests
// ============================================================================

template <typename T>
int test_blockdiag_get_diagonal() {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 0, 0, 2}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {3, 0, 0, 4}));

    BlockDiagMatrix<T> m(blocks);
    cpu::Vector<T, CudaHostAllocator<T>> diag = m.get_diagonal();

    if (!check((int)diag.size(), 4, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((T)diag[0], (T)1, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((T)diag[1], (T)2, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check((T)diag[2], (T)3, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;
    if (!check((T)diag[3], (T)4, check_msg(get_type_name<T>(), "check 5"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_set_diagonal() {
    BlockDiagMatrix<T> m(2, Shape{2, 2});

    cpu::Vector<T, CudaHostAllocator<T>> diag({(T)5, (T)6, (T)7, (T)8});

    m.set_diagonal(diag);

    if (!check((T)m(0, 0), (T)5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((T)m(1, 1), (T)6, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((T)m(2, 2), (T)7, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_symmetrize() {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 3, 2, 4}));

    BlockDiagMatrix<T> m(blocks);
    m.symmetrize();

    if (!check((T)m(0, 0), (T)1, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((T)m(0, 1), (T)2.5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((T)m(1, 0), (T)2.5, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// BlockDiagMatrix Edge Cases Tests
// ============================================================================

template <typename T>
int test_blockdiag_single_block() {
    BlockDiagMatrix<T> m(1, Shape{5, 5}, (T)1);

    if (!check((int)m.num_blocks(), 1, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    Shape s = m.shape();
    if (!check((int)s.first, 5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((T)m(0, 0), (T)1, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_non_square_blocks() {
    std::vector<Shape> shapes = {{2, 4}, {3, 2}};
    BlockDiagMatrix<T> m(shapes);

    Shape s = m.shape();
    if (!check((int)s.first, 5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((int)s.second, 6, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_non_uniform_block_sizes() {
    std::vector<Shape> shapes = {{1, 1}, {2, 2}, {4, 4}, {3, 3}};
    BlockDiagMatrix<T> m(shapes);

    if (!check((int)m.num_blocks(), 4, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    Shape s = m.shape();
    if (!check((int)s.first, 10, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((int)s.second, 10, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_copy_assignment() {
    BlockDiagMatrix<T> m1(2, Shape{3, 3}, (T)1.5);
    BlockDiagMatrix<T> m2;

    m2 = m1;  // Copy assignment

    if (!check((int)m2.num_blocks(), 2, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    Shape s = m2.shape();
    if (!check((int)s.first, 6, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((T)m2(0, 0), (T)1.5, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_move_assignment() {
    BlockDiagMatrix<T> m1(2, Shape{3, 3}, (T)2.0);
    BlockDiagMatrix<T> m2;

    m2 = std::move(m1);  // Move assignment

    if (!check((int)m2.num_blocks(), 2, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((int)m1.num_blocks(), 0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_get_block_shape() {
    std::vector<Shape> shapes = {{2, 2}, {3, 3}, {1, 1}};
    BlockDiagMatrix<T> m(shapes);

    Shape block_shape = m.get_block_shape(0);
    if (!check((int)block_shape.first, 2, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    block_shape = m.get_block_shape(1);
    if (!check((int)block_shape.first, 3, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_blockdiag_get_block_data() {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, (T)5));

    BlockDiagMatrix<T> m(blocks);

    const void* data_ptr = m.get_block_data(0);
    if (data_ptr == nullptr) {
        std::cerr << check_msg(get_type_name<T>(), "") << std::endl;
        return TEST_FAIL;
    }

    return TEST_PASS;
}

// ============================================================================
// GPU Round-Trip Transfer Tests (CPU -> GPU -> CPU with data verification)
// ============================================================================

template <typename T>
int test_sparse_matrix_gpu_roundtrip_csr() {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {5, 6, 7, 8}));

    BlockDiagMatrix<T> bdm(blocks);

    CudaRuntime cudart;
    gpu::SparseMatrix<T> sparse_csr(cudart, bdm, gpu::SparseFormat::CSR);

    Shape bdm_shape = bdm.shape();
    Matrix<T> original_dense(bdm_shape, (T)0);
    for (size_t i = 0; i < bdm_shape.first; ++i) {
        for (size_t j = 0; j < bdm_shape.second; ++j) {
            original_dense(i, j) = bdm(i, j);
        }
    }

    sparse_csr.allocate_gpu_memory();
    sparse_csr.copy2device(cudart);
    sparse_csr.copy2host(cudart);
    Matrix<T> reconstructed_dense = sparse_csr.to_dense();

    Shape reconstructed_shape = reconstructed_dense.shape();
    if (reconstructed_shape.first != original_dense.shape().first ||
        reconstructed_shape.second != original_dense.shape().second) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    }

    // Verify all elements match
    for (size_t i = 0; i < original_dense.shape().first; ++i) {
        for (size_t j = 0; j < original_dense.shape().second; ++j) {
            if (!check(original_dense(i, j), reconstructed_dense(i, j), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
        }
    }

    sparse_csr.free_gpu_memory();
    return TEST_PASS;
}

template <typename T>
int test_sparse_matrix_gpu_roundtrip_bsr() {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{3, 3}, {10, 11, 12, 13, 14, 15, 16, 17, 18}));

    BlockDiagMatrix<T> bdm(blocks);

    CudaRuntime cudart;
    gpu::SparseMatrix<T> sparse_bsr(cudart, bdm, gpu::SparseFormat::BSR);

    Shape bdm_shape = bdm.shape();
    Matrix<T> original_dense(bdm_shape, (T)0);
    for (size_t i = 0; i < bdm_shape.first; ++i) {
        for (size_t j = 0; j < bdm_shape.second; ++j) {
            original_dense(i, j) = bdm(i, j);
        }
    }

    sparse_bsr.allocate_gpu_memory();
    sparse_bsr.copy2device(cudart);
    sparse_bsr.copy2host(cudart);

    Matrix<T> reconstructed_dense = sparse_bsr.to_dense();

    Shape reconstructed_shape = reconstructed_dense.shape();
    if (reconstructed_shape.first != original_dense.shape().first ||
        reconstructed_shape.second != original_dense.shape().second) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    }

    for (size_t i = 0; i < original_dense.shape().first; ++i) {
        for (size_t j = 0; j < original_dense.shape().second; ++j) {
            if (!check(original_dense(i, j), reconstructed_dense(i, j), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
        }
    }

    sparse_bsr.free_gpu_memory();
    return TEST_PASS;
}

template <typename T>
int test_sparse_matrix_gpu_roundtrip_varied_sizes() {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{3, 3}, {5, 6, 7, 8, 9, 10, 11, 12, 13}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {14, 15, 16, 17}));

    BlockDiagMatrix<T> bdm(blocks);

    CudaRuntime cudart;
    gpu::SparseMatrix<T> sparse(cudart, bdm, gpu::SparseFormat::CSR);

    Shape bdm_shape = bdm.shape();
    Matrix<T> original_dense(bdm_shape, (T)0);
    for (size_t i = 0; i < bdm_shape.first; ++i) {
        for (size_t j = 0; j < bdm_shape.second; ++j) {
            original_dense(i, j) = bdm(i, j);
        }
    }

    sparse.allocate_gpu_memory();
    sparse.copy2device(cudart);
    sparse.copy2host(cudart);

    Matrix<T> reconstructed_dense = sparse.to_dense();

    Shape reconstructed_shape = reconstructed_dense.shape();
    if (reconstructed_shape.first != original_dense.shape().first ||
        reconstructed_shape.second != original_dense.shape().second) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    }

    // Verify all elements match
    for (size_t i = 0; i < original_dense.shape().first; ++i) {
        for (size_t j = 0; j < original_dense.shape().second; ++j) {
            if (!check(original_dense(i, j), reconstructed_dense(i, j), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
        }
    }

    sparse.free_gpu_memory();
    return TEST_PASS;
}

// ============================================================================
// BlockDiagMatrix Sparse Cache Tests
// ============================================================================

template <typename T>
cudaDataType_t sparse_precision() {
    return std::is_same<T, double>::value ? CUDA_R_64F : CUDA_R_32F;
}

// Checks that the host and device copies of the cached sparse data both match the block matrix
template <typename T>
bool sparse_matches(const BlockDiagMatrix<T> &bdm, const gpu::SparseMatrix<T> &sparse) {
    Matrix<T> dense = sparse.to_dense();
    Shape s = bdm.shape();
    for (size_t i = 0; i < s.first; ++i) {
        for (size_t j = 0; j < s.second; ++j) {
            if (dense(i, j) != bdm(i, j)) return false;
        }
    }

    const auto &data = sparse.get_sparse_data();
    std::vector<T> device_values(data.h_values.size());
    get_cuda_error(cudaMemcpy(device_values.data(), data.d_values, device_values.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return device_values == data.h_values;
}

template <typename T>
int test_blockdiag_sparse_cache_value_mutators() {
    CudaRuntime cudart;
    for (gpu::SparseFormat format : {gpu::SparseFormat::CSR, gpu::SparseFormat::BSR}) {
        std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
        blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
        blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {5, 6, 7, 8}));
        BlockDiagMatrix<T> bdm(blocks);

        const auto *cached = &bdm.get_sparse(cudart, format, sparse_precision<T>());

        bdm += (T)1;
        if (!check((int)(sparse_matches(bdm, bdm.get_sparse(cudart, format, sparse_precision<T>()))), 1,
                   check_msg(get_type_name<T>(), "Values should be refreshed after operator+="))) return TEST_FAIL;

        bdm.set_diagonal(cpu::Vector<T, CudaHostAllocator<T>>({(T)-1, (T)-2, (T)-3, (T)-4}));
        if (!check((int)(sparse_matches(bdm, bdm.get_sparse(cudart, format, sparse_precision<T>()))), 1,
                   check_msg(get_type_name<T>(), "Values should be refreshed after set_diagonal"))) return TEST_FAIL;

        bdm.symmetrize();
        if (!check((int)(sparse_matches(bdm, bdm.get_sparse(cudart, format, sparse_precision<T>()))), 1,
                   check_msg(get_type_name<T>(), "Values should be refreshed after symmetrize"))) return TEST_FAIL;

        bdm(3, 2) = (T)99;
        const auto *after_element = &bdm.get_sparse(cudart, format, sparse_precision<T>());
        if (!check((int)(sparse_matches(bdm, *after_element)), 1,
                   check_msg(get_type_name<T>(), "Values should be refreshed after element write"))) return TEST_FAIL;

        if (!check((int)(after_element == cached), 1, check_msg(get_type_name<T>(), "Value changes should reuse the cached object"))) return TEST_FAIL;
        if (!check((int)bdm.sparse_build_count(), 1, check_msg(get_type_name<T>(), "Value changes should not rebuild"))) return TEST_FAIL;
        if (!check((int)bdm.sparse_values_refresh_count(), 4, check_msg(get_type_name<T>(), "Each value change should refresh once"))) return TEST_FAIL;
    }
    return TEST_PASS;
}

template <typename T>
int test_blockdiag_sparse_cache_structural_mutators() {
    CudaRuntime cudart;
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    BlockDiagMatrix<T> bdm(blocks);
    bdm.get_sparse(cudart, gpu::SparseFormat::CSR, sparse_precision<T>());

    bdm.add_block(Matrix<T, CudaHostAllocator<T>>(Shape{3, 3}, (T)2));
    const auto &after_add = bdm.get_sparse(cudart, gpu::SparseFormat::CSR, sparse_precision<T>());
    if (!check((int)bdm.sparse_build_count(), 2, check_msg(get_type_name<T>(), "add_block should trigger a rebuild"))) return TEST_FAIL;
    if (!check((int)after_add.nnz(), 13, check_msg(get_type_name<T>(), "Rebuilt matrix should contain the new block"))) return TEST_FAIL;
    if (!check((int)(sparse_matches(bdm, after_add)), 1, check_msg(get_type_name<T>(), "Rebuilt matrix should match after add_block"))) return TEST_FAIL;

    // Copy assignment replaces the data and must drop the cache
    std::vector<Matrix<T, CudaHostAllocator<T>>> other_blocks;
    other_blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {9, 8, 7, 6}));
    BlockDiagMatrix<T> other(other_blocks);
    bdm = other;
    const auto &after_assign = bdm.get_sparse(cudart, gpu::SparseFormat::CSR, sparse_precision<T>());
    if (!check((int)after_assign.nnz(), 4, check_msg(get_type_name<T>(), "Copy-assigned matrix should rebuild"))) return TEST_FAIL;
    if (!check((int)(sparse_matches(bdm, after_assign)), 1, check_msg(get_type_name<T>(), "Copy-assigned matrix should match"))) return TEST_FAIL;

    // A copy does not share the cache
    BlockDiagMatrix<T> copy(bdm);
    if (!check((int)copy.sparse_build_count(), 0, check_msg(get_type_name<T>(), "Copy should not share the cache"))) return TEST_FAIL;
    if (!check((int)(&copy.get_sparse(cudart, gpu::SparseFormat::CSR, sparse_precision<T>()) != &after_assign), 1,
               check_msg(get_type_name<T>(), "Copy should have its own sparse matrix"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int total_failures = 0;
    CudaRuntime cudart;

    // BlockDiagMatrix constructors
    total_failures += test_blockdiag_default_constructor<double>();
    total_failures += test_blockdiag_default_constructor<float>();
    total_failures += test_blockdiag_uniform_blocks_constructor<double>();
    total_failures += test_blockdiag_uniform_blocks_constructor<float>();
    total_failures += test_blockdiag_uniform_blocks_with_value_constructor<double>();
    total_failures += test_blockdiag_uniform_blocks_with_value_constructor<float>();
    total_failures += test_blockdiag_varied_blocks_constructor<double>();
    total_failures += test_blockdiag_varied_blocks_constructor<float>();
    total_failures += test_blockdiag_varied_blocks_with_value_constructor<double>();
    total_failures += test_blockdiag_varied_blocks_with_value_constructor<float>();
    total_failures += test_blockdiag_from_matrices_constructor<double>();
    total_failures += test_blockdiag_from_matrices_constructor<float>();
    total_failures += test_blockdiag_copy_constructor<double>();
    total_failures += test_blockdiag_copy_constructor<float>();
    total_failures += test_blockdiag_move_constructor<double>();
    total_failures += test_blockdiag_move_constructor<float>();

    // BlockDiagMatrix block operations
    total_failures += test_blockdiag_get_block<double>();
    total_failures += test_blockdiag_get_block<float>();
    total_failures += test_blockdiag_add_block<double>();
    total_failures += test_blockdiag_add_block<float>();
    total_failures += test_blockdiag_add_block_at_index<double>();
    total_failures += test_blockdiag_add_block_at_index<float>();
    total_failures += test_blockdiag_num_blocks<double>();
    total_failures += test_blockdiag_num_blocks<float>();
    total_failures += test_blockdiag_block_shapes<double>();
    total_failures += test_blockdiag_block_shapes<float>();

    // BlockDiagMatrix dimensions
    total_failures += test_blockdiag_shape<double>();
    total_failures += test_blockdiag_shape<float>();
    total_failures += test_blockdiag_rows<double>();
    total_failures += test_blockdiag_rows<float>();
    total_failures += test_blockdiag_cols<double>();
    total_failures += test_blockdiag_cols<float>();

    // BlockDiagMatrix offsets
    total_failures += test_blockdiag_get_row_offsets<double>();
    total_failures += test_blockdiag_get_row_offsets<float>();
    total_failures += test_blockdiag_get_col_offsets<double>();
    total_failures += test_blockdiag_get_col_offsets<float>();

    // BlockDiagMatrix element access
    total_failures += test_blockdiag_element_access_read<double>();
    total_failures += test_blockdiag_element_access_read<float>();
    total_failures += test_blockdiag_element_access_write<double>();
    total_failures += test_blockdiag_element_access_write<float>();

    // BlockDiagMatrix GPU operations
    total_failures += test_blockdiag_ensure_on_gpu<double>(cudart);
    total_failures += test_blockdiag_ensure_on_gpu<float>(cudart);
    total_failures += test_blockdiag_ensure_on_gpu_without_blocks<double>(cudart);
    total_failures += test_blockdiag_ensure_on_gpu_without_blocks<float>(cudart);
    total_failures += test_blockdiag_repeated_copy2device_frees_cache<double>(cudart);
    total_failures += test_blockdiag_repeated_copy2device_frees_cache<float>(cudart);
    // GPU round-trip transfers
    total_failures += test_sparse_matrix_gpu_roundtrip_csr<double>();
    total_failures += test_sparse_matrix_gpu_roundtrip_csr<float>();
    total_failures += test_sparse_matrix_gpu_roundtrip_bsr<double>();
    total_failures += test_sparse_matrix_gpu_roundtrip_bsr<float>();
    total_failures += test_sparse_matrix_gpu_roundtrip_varied_sizes<double>();
    total_failures += test_sparse_matrix_gpu_roundtrip_varied_sizes<float>();

    // BlockDiagMatrix diagonal operations
    total_failures += test_blockdiag_get_diagonal<double>();
    total_failures += test_blockdiag_get_diagonal<float>();
    total_failures += test_blockdiag_set_diagonal<double>();
    total_failures += test_blockdiag_set_diagonal<float>();
    total_failures += test_blockdiag_symmetrize<double>();
    total_failures += test_blockdiag_symmetrize<float>();

    // BlockDiagMatrix edge cases
    total_failures += test_blockdiag_single_block<double>();
    total_failures += test_blockdiag_single_block<float>();
    total_failures += test_blockdiag_non_square_blocks<double>();
    total_failures += test_blockdiag_non_square_blocks<float>();
    total_failures += test_blockdiag_non_uniform_block_sizes<double>();
    total_failures += test_blockdiag_non_uniform_block_sizes<float>();
    total_failures += test_blockdiag_copy_assignment<double>();
    total_failures += test_blockdiag_copy_assignment<float>();
    total_failures += test_blockdiag_move_assignment<double>();
    total_failures += test_blockdiag_move_assignment<float>();
    total_failures += test_blockdiag_get_block_shape<double>();
    total_failures += test_blockdiag_get_block_shape<float>();
    total_failures += test_blockdiag_get_block_data<double>();
    total_failures += test_blockdiag_get_block_data<float>();

    // BlockDiagMatrix sparse cache
    total_failures += test_blockdiag_sparse_cache_value_mutators<double>();
    total_failures += test_blockdiag_sparse_cache_value_mutators<float>();
    total_failures += test_blockdiag_sparse_cache_structural_mutators<double>();
    total_failures += test_blockdiag_sparse_cache_structural_mutators<float>();

    if (total_failures > 0) {
        std::cerr << "gpu/tensor/blockdiagmatrix tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All gpu/tensor/blockdiagmatrix tests passed!" << std::endl;
    return TEST_PASS;
}
