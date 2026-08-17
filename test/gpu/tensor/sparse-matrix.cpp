#include "test_common.h"

using namespace lahva;
using namespace lahva::gpu;

// ============================================================================
// SparseMatrix Construction and Initialization Tests
// ============================================================================

template <typename T>
int test_sparse_matrix_default_constructor() {
    SparseMatrix<T> sparse;

    if (!check((int)sparse.rows(), 0, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((int)sparse.cols(), 0, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((int)sparse.nnz(), 0, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check((int)sparse.num_blocks(), 0, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;
    if (sparse.is_initialized()) {
        std::cerr << check_msg(get_type_name<T>(), "check 4") << std::endl;
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_sparse_matrix_csr_constructor() {
    SparseMatrix<T> sparse(SparseFormat::CSR);

    if (sparse.get_format() != SparseFormat::CSR) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    }
    if (sparse.is_initialized()) {
        std::cerr << check_msg(get_type_name<T>(), "check 2") << std::endl;
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_sparse_matrix_bsr_constructor() {
    SparseMatrix<T> sparse(SparseFormat::BSR);

    if (sparse.get_format() != SparseFormat::BSR) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    }
    if (sparse.is_initialized()) {
        std::cerr << check_msg(get_type_name<T>(), "check 2") << std::endl;
        return TEST_FAIL;
    }

    return TEST_PASS;
}

// ============================================================================
// SparseMatrix Conversion and Initialization Tests
// ============================================================================

template <typename T>
int test_sparse_matrix_convert_uniform_blocks_csr(const CudaRuntime& cudart) {
    // Create a simple BlockDiagMatrix
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {5, 6, 7, 8}));
    BlockDiagMatrix<T> bdm(blocks);

    // Convert to sparse CSR format
    SparseMatrix<T> sparse(cudart, bdm, SparseFormat::CSR);

    if (!sparse.is_initialized()) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    }
    if (!check((int)sparse.rows(), 4, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((int)sparse.cols(), 4, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check((int)sparse.nnz(), 8, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;
    if (!check((int)sparse.num_blocks(), 2, check_msg(get_type_name<T>(), "check 5"))) return TEST_FAIL;
    if (sparse.get_format() != SparseFormat::CSR) {
        std::cerr << check_msg(get_type_name<T>(), "check 6") << std::endl;
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_sparse_matrix_convert_uniform_blocks_bsr(const CudaRuntime& cudart) {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{3, 3}, {10, 11, 12, 13, 14, 15, 16, 17, 18}));
    BlockDiagMatrix<T> bdm(blocks);
    SparseMatrix<T> sparse(cudart, bdm, SparseFormat::BSR);

    if (!sparse.is_initialized()) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    }
    if (!check((int)sparse.rows(), 6, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((int)sparse.cols(), 6, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check((int)sparse.nnz(), 18, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;
    if (sparse.get_format() != SparseFormat::BSR) {
        std::cerr << check_msg(get_type_name<T>(), "check 5") << std::endl;
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
int test_sparse_matrix_convert_varied_block_sizes_csr(const CudaRuntime& cudart) {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{3, 3}, {5, 6, 7, 8, 9, 10, 11, 12, 13}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{1, 1}, (T)14));
    BlockDiagMatrix<T> bdm(blocks);
    SparseMatrix<T> sparse(cudart, bdm, SparseFormat::CSR);

    if (!sparse.is_initialized()) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    }
    if (!check((int)sparse.rows(), 6, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((int)sparse.cols(), 6, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check((int)sparse.nnz(), 14, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;
    if (!check((int)sparse.num_blocks(), 3, check_msg(get_type_name<T>(), "check 5"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// SparseMatrix Data Access Tests
// ============================================================================

template <typename T>
int test_sparse_matrix_get_sparse_data(const CudaRuntime& cudart) {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, (T)1));
    BlockDiagMatrix<T> bdm(blocks);
    SparseMatrix<T> sparse(cudart, bdm, SparseFormat::CSR);
    
    const GPUSparseBlockDiagData<T>& data = sparse.get_sparse_data();

    if (data.h_values.size() == 0) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    }
    if (data.h_row_offsets.size() == 0) {
        std::cerr << check_msg(get_type_name<T>(), "check 2") << std::endl;
        return TEST_FAIL;
    }
    if (data.h_col_indices.size() == 0) {
        std::cerr << check_msg(get_type_name<T>(), "check 3") << std::endl;
        return TEST_FAIL;
    }

    return TEST_PASS;
}

// ============================================================================
// SparseMatrix GPU Memory Management Tests
// ============================================================================

template <typename T>
int test_sparse_matrix_allocate_gpu_memory(const CudaRuntime& cudart) {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, (T)1));
    BlockDiagMatrix<T> bdm(blocks);
    SparseMatrix<T> sparse(cudart, bdm, SparseFormat::CSR);

    sparse.allocate_gpu_memory();

    if (!sparse.is_initialized()) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    }
    if (!check((int)sparse.nnz() > 0, 1, check_msg(get_type_name<T>(), "check 2"))) {
        sparse.release_gpu_memory();
        return TEST_FAIL;
    }

    sparse.release_gpu_memory();
    return TEST_PASS;
}

template <typename T>
int test_sparse_matrix_transfer_to_device(const CudaRuntime& cudart) {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    BlockDiagMatrix<T> bdm(blocks);
    SparseMatrix<T> sparse(cudart, bdm, SparseFormat::CSR);

    sparse.allocate_gpu_memory();
    sparse.transfer_to_device(cudart);

    // Verify sparse matrix is still valid
    if (!sparse.is_initialized()) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        sparse.release_gpu_memory();
        return TEST_FAIL;
    }

    sparse.release_gpu_memory();
    return TEST_PASS;
}

template <typename T>
int test_sparse_matrix_transfer_to_host(const CudaRuntime& cudart) {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    BlockDiagMatrix<T> bdm(blocks);
    SparseMatrix<T> sparse(cudart, bdm, SparseFormat::CSR);

    sparse.allocate_gpu_memory();
    sparse.transfer_to_device(cudart);
    sparse.transfer_to_host(cudart);

    if (!sparse.is_initialized()) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        sparse.release_gpu_memory();
        return TEST_FAIL;
    }

    sparse.release_gpu_memory();
    return TEST_PASS;
}

template <typename T>
int test_sparse_matrix_release_gpu_memory(const CudaRuntime& cudart) {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, (T)1));
    BlockDiagMatrix<T> bdm(blocks);
    SparseMatrix<T> sparse(cudart, bdm, SparseFormat::CSR);

    if (!sparse.is_initialized()) {
        std::cerr << check_msg(get_type_name<T>(), "check 1") << std::endl;
        return TEST_FAIL;
    }

    sparse.allocate_gpu_memory();
    sparse.release_gpu_memory();

    if (sparse.is_initialized()) {
        std::cerr << check_msg(get_type_name<T>(), "check 2") << std::endl;
        return TEST_FAIL;
    }
    if (!check((int)sparse.rows(), 0, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check((int)sparse.cols(), 0, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// SparseMatrix Dense Reconstruction Tests
// ============================================================================

template <typename T>
int test_sparse_matrix_to_dense_csr(const CudaRuntime& cudart) {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {5, 6, 7, 8}));
    BlockDiagMatrix<T> bdm(blocks);
    SparseMatrix<T> sparse(cudart, bdm, SparseFormat::CSR);

    Matrix<T> dense = sparse.to_dense();

    Shape s = dense.shape();
    if (!check((int)s.first, 4, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((int)s.second, 4, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(dense(0, 0), (T)1, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check(dense(2, 2), (T)5, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;
    if (!check(dense(0, 2), (T)0, check_msg(get_type_name<T>(), "check 5"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_sparse_matrix_to_dense_bsr(const CudaRuntime& cudart) {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {5, 6, 7, 8}));
    BlockDiagMatrix<T> bdm(blocks);
    SparseMatrix<T> sparse(cudart, bdm, SparseFormat::BSR);

    Matrix<T> dense = sparse.to_dense();

    Shape s = dense.shape();
    if (!check((int)s.first, 4, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((int)s.second, 4, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check(dense(0, 0), (T)1, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check(dense(2, 2), (T)5, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// SparseMatrix Format Comparison Tests
// ============================================================================

template <typename T>
int test_sparse_matrix_csr_vs_bsr_format(const CudaRuntime& cudart) {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {5, 6, 7, 8}));
    BlockDiagMatrix<T> bdm(blocks);

    // Create both CSR and BSR versions
    SparseMatrix<T> sparse_csr(cudart, bdm, SparseFormat::CSR);
    SparseMatrix<T> sparse_bsr(cudart, bdm, SparseFormat::BSR);

    if (!check((int)sparse_csr.rows(), (int)sparse_bsr.rows(), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((int)sparse_csr.cols(), (int)sparse_bsr.cols(), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((int)sparse_csr.nnz(), (int)sparse_bsr.nnz(), check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    // Dense reconstructions should match
    Matrix<T> dense_csr = sparse_csr.to_dense();
    Matrix<T> dense_bsr = sparse_bsr.to_dense();

    for (size_t i = 0; i < dense_csr.shape().first; ++i) {
        for (size_t j = 0; j < dense_csr.shape().second; ++j) {
            if (!check(dense_csr(i, j), dense_bsr(i, j), check_msg(get_type_name<T>(), "check 4"))) {
                return TEST_FAIL;
            }
        }
    }

    return TEST_PASS;
}

// ============================================================================
// SparseMatrix Edge Cases Tests
// ============================================================================

template <typename T>
int test_sparse_matrix_single_block(const CudaRuntime& cudart) {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{5, 5}, (T)3));
    BlockDiagMatrix<T> bdm(blocks);
    SparseMatrix<T> sparse(cudart, bdm, SparseFormat::CSR);

    if (!check((int)sparse.num_blocks(), 1, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((int)sparse.rows(), 5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((int)sparse.nnz(), 25, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    Matrix<T> dense = sparse.to_dense();
    if (!check(dense(0, 0), (T)3, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
}

template <typename T>
int test_sparse_matrix_large_block_structure(const CudaRuntime& cudart) {
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    for (int i = 0; i < 5; ++i) {
        blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{3, 3}, (T)(i + 1)));
    }
    BlockDiagMatrix<T> bdm(blocks);
    SparseMatrix<T> sparse(cudart, bdm, SparseFormat::CSR);

    if (!check((int)sparse.num_blocks(), 5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check((int)sparse.rows(), 15, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;
    if (!check((int)sparse.nnz(), 45, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;

    return TEST_PASS;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int total_failures = 0;
    CudaRuntime cudart;

    // SparseMatrix constructors
    total_failures += test_sparse_matrix_default_constructor<double>();
    total_failures += test_sparse_matrix_default_constructor<float>();
    total_failures += test_sparse_matrix_csr_constructor<double>();
    total_failures += test_sparse_matrix_csr_constructor<float>();
    total_failures += test_sparse_matrix_bsr_constructor<double>();
    total_failures += test_sparse_matrix_bsr_constructor<float>();

    // SparseMatrix conversion
    total_failures += test_sparse_matrix_convert_uniform_blocks_csr<double>(cudart);
    total_failures += test_sparse_matrix_convert_uniform_blocks_csr<float>(cudart);
    total_failures += test_sparse_matrix_convert_uniform_blocks_bsr<double>(cudart);
    total_failures += test_sparse_matrix_convert_uniform_blocks_bsr<float>(cudart);
    total_failures += test_sparse_matrix_convert_varied_block_sizes_csr<double>(cudart);
    total_failures += test_sparse_matrix_convert_varied_block_sizes_csr<float>(cudart);

    // SparseMatrix data access
    total_failures += test_sparse_matrix_get_sparse_data<double>(cudart);
    total_failures += test_sparse_matrix_get_sparse_data<float>(cudart);

    // SparseMatrix GPU memory management
    total_failures += test_sparse_matrix_allocate_gpu_memory<double>(cudart);
    total_failures += test_sparse_matrix_allocate_gpu_memory<float>(cudart);
    total_failures += test_sparse_matrix_transfer_to_device<double>(cudart);
    total_failures += test_sparse_matrix_transfer_to_device<float>(cudart);
    total_failures += test_sparse_matrix_transfer_to_host<double>(cudart);
    total_failures += test_sparse_matrix_transfer_to_host<float>(cudart);
    total_failures += test_sparse_matrix_release_gpu_memory<double>(cudart);
    total_failures += test_sparse_matrix_release_gpu_memory<float>(cudart);

    // SparseMatrix dense reconstruction
    total_failures += test_sparse_matrix_to_dense_csr<double>(cudart);
    total_failures += test_sparse_matrix_to_dense_csr<float>(cudart);
    total_failures += test_sparse_matrix_to_dense_bsr<double>(cudart);
    total_failures += test_sparse_matrix_to_dense_bsr<float>(cudart);

    // SparseMatrix format comparison
    total_failures += test_sparse_matrix_csr_vs_bsr_format<double>(cudart);
    total_failures += test_sparse_matrix_csr_vs_bsr_format<float>(cudart);

    // SparseMatrix edge cases
    total_failures += test_sparse_matrix_single_block<double>(cudart);
    total_failures += test_sparse_matrix_single_block<float>(cudart);
    total_failures += test_sparse_matrix_large_block_structure<double>(cudart);
    total_failures += test_sparse_matrix_large_block_structure<float>(cudart);

    if (total_failures > 0) {
        std::cerr << "gpu/tensor/sparse-matrix tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All gpu/tensor/sparse-matrix tests passed!" << std::endl;
    return TEST_PASS;
}
