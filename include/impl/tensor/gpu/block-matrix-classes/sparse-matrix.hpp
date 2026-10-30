#pragma once

#include <algorithm>
#include <cstring>
#include <numeric>
#include "impl/tensor/gpu/block-matrix-classes/block-matrix.hpp"
#include "runtime.hpp"

namespace lahva
{
    namespace gpu
    {

        /// @brief Helper struct for GPU sparse block-diagonal matrix data
        template <typename T>
        struct GPUSparseBlockDiagData
        {
            // Host memory
            std::vector<T> h_values;        ///< Host copy of sparse values
            std::vector<int> h_row_offsets; ///< Host copy of row offsets (for CSR/BSR)
            std::vector<int> h_col_indices; ///< Host copy of column indices

            // Device memory
            T *d_values = nullptr;                   ///< Device pointer to sparse values
            int *d_row_offsets = nullptr;            ///< Device pointer to row offsets
            int *d_col_indices = nullptr;            ///< Device pointer to column indices
            SparseFormat format = SparseFormat::CSR; ///< Sparse format (CSR or BSR)

            // BSR specific
            int ell_block_rows = 0;               ///< Number of block rows in BSR
            int ell_block_cols = 0;               ///< Number of block columns in BSR
            int ell_num_blocks_per_row = 0;       ///< Maximum blocks per row
            std::vector<int> block_value_offsets; ///< Value array offsets for each original block index (for BSR)
        };

        /// @brief GPU-based sparse representation of block diagonal matrix in CSR or BSR format
        ///
        /// Stores a block diagonal matrix directly on the GPU in either Compressed Sparse Row (CSR)
        /// or Block Sparse Row (BSR) format, enabling efficient sparse matrix operations without
        /// CPU-GPU transfers during computation. Inherits from Tensor_ to provide consistent
        /// GPU tensor semantics and memory management.
        template <typename T>
        class SparseMatrix : public virtual Tensor_<T>
        {
        public:
            /// @brief Default constructor
            /// @param[in] format sparse format (CSR or BSR) - defaults to CSR
            SparseMatrix(SparseFormat format = SparseFormat::CSR)
                : m_(0), n_(0), nnz_(0), format_(format),
                  mat_descr_(nullptr), initialized_(false), num_blocks_(0)
            {
            }

            /// @brief Construct and convert from a GPU BlockMatrix
            /// @param[in] cudart CUDA runtime for device operations
            /// @param[in] block_matrix source block matrix to convert
            /// @param[in] format sparse format (CSR or BSR) - defaults to CSR
            SparseMatrix(const CudaRuntime &cudart,
                         const gpu::BlockMatrix_<T> &block_matrix,
                         SparseFormat format = SparseFormat::CSR)
                : m_(0), n_(0), nnz_(0), format_(format),
                  mat_descr_(nullptr), initialized_(false), num_blocks_(0),
                  device_id_(cudart.device_id())
            {
                // Free existing GPU memory
                free_gpu_memory();

                num_blocks_ = block_matrix.num_blocks();

                if (num_blocks_ == 0)
                {
                    return;
                }

                // Calculate total size, nnz, and get matrix dimensions from BlockMatrix
                int64_t total_nnz = 0;

                block_row_sizes_.clear();
                block_col_sizes_.clear();
                block_row_positions_.clear();
                block_col_positions_.clear();

                for (size_t i = 0; i < block_matrix.num_blocks(); ++i)
                {
                    Shape block_shape = block_matrix.get_block_shape(i);
                    int64_t block_rows = block_shape.first;
                    int64_t block_cols = block_shape.second;

                    block_row_sizes_.push_back(block_rows);
                    block_col_sizes_.push_back(block_cols);

                    total_nnz += block_rows * block_cols;
                }

                // Get actual matrix dimensions from BlockMatrix
                Shape matrix_shape = block_matrix.shape();
                m_ = matrix_shape.first;
                n_ = matrix_shape.second;
                nnz_ = total_nnz;

                row_offsets_ = block_matrix.get_row_offsets();
                col_offsets_ = block_matrix.get_col_offsets();

                // Get element-space positions for each block
                for (size_t i = 0; i < block_matrix.num_blocks(); ++i)
                {
                    block_row_positions_.push_back(block_matrix.get_block_row(i));
                    block_col_positions_.push_back(block_matrix.get_block_col(i));
                }

                // Validate BSR format - all blocks must be the same size and be square
                if (format_ == SparseFormat::BSR)
                {
                    bool bsr_supported = block_row_sizes_[0] == block_col_sizes_[0];
                    for (size_t i = 1; i < block_row_sizes_.size(); ++i)
                    {
                        if (block_row_sizes_[i] != block_row_sizes_[0] ||
                            block_col_sizes_[i] != block_col_sizes_[0])
                        {
                            bsr_supported = false;
                            break;
                        }
                    }
                    if (!bsr_supported)
                    {
                        format_ = SparseFormat::CSR;
                    }
                }

                if (format_ == SparseFormat::CSR)
                {
                    convert_to_csr_gpu(cudart, block_matrix);
                }
                else if (format_ == SparseFormat::BSR)
                {
                    convert_to_bsr_gpu(cudart, block_matrix);
                }

                initialized_ = true;
            }

            /// @brief Copying is disabled: the object owns raw device allocations and a cuSPARSE descriptor
            SparseMatrix(const SparseMatrix &) = delete;
            SparseMatrix &operator=(const SparseMatrix &) = delete;

            /// @brief Destructor - frees all GPU memory
            virtual ~SparseMatrix()
            {
                free_gpu_memory();
            }

            /// @brief Get the CUDA device the data was uploaded to
            int device_id() const { return device_id_; }

            /// @brief Re-pack the values from the source block matrix and upload only the values array
            /// @param[in] cudart CUDA runtime for device operations
            /// @param[in] block_matrix source block matrix; must have the same block structure as at construction
            /// @note Row offsets, column indices, allocations and the cuSPARSE descriptor are reused
            void refresh_values(const CudaRuntime &cudart, const gpu::BlockMatrix_<T> &block_matrix)
            {
                if (!initialized_ || static_cast<int>(block_matrix.num_blocks()) != num_blocks_)
                {
                    throw std::logic_error("SparseMatrix::refresh_values: block structure changed since construction");
                }

                pack_values(block_matrix);

                if (cudart.asyncCopy())
                {
                    get_cuda_error(cudaMemcpyAsync(sparse_data_.d_values, sparse_data_.h_values.data(),
                                                   sparse_data_.h_values.size() * sizeof(T),
                                                   cudaMemcpyHostToDevice, cudart.getStream()));
                }
                else
                {
                    get_cuda_error(cudaMemcpy(sparse_data_.d_values, sparse_data_.h_values.data(),
                                              sparse_data_.h_values.size() * sizeof(T),
                                              cudaMemcpyHostToDevice));
                }
                cudart.synchronize();
            }

            /// @brief Get a device scratch buffer of at least the requested size for cuSPARSE operations
            /// @param[in] bytes required size in bytes
            /// @return device pointer (nullptr if bytes == 0); grows but never shrinks between calls
            /// @note The buffer is shared by all operations on this matrix; operations must be stream-ordered
            void *get_workspace(size_t bytes) const
            {
                if (bytes > workspace_size_)
                {
                    if (workspace_)
                    {
                        get_cuda_error(cudaFree(workspace_));
                        workspace_ = nullptr;
                        workspace_size_ = 0;
                    }
                    get_cuda_error(cudaMalloc(&workspace_, bytes));
                    workspace_size_ = bytes;
                }
                return workspace_;
            }

            /// @brief Get number of matrix rows
            /// @return total number of rows
            int64_t rows() const { return m_; }

            /// @brief Get number of matrix columns
            /// @return total number of columns
            int64_t cols() const { return n_; }

            /// @brief Get number of non-zero elements
            /// @return total number of non-zero elements
            int64_t nnz() const { return nnz_; }

            /// @brief Get number of blocks
            /// @return total number of blocks
            int num_blocks() const { return num_blocks_; }

            /// @brief Check if sparse matrix is initialized
            /// @return true if matrix has been initialized, false otherwise
            bool is_initialized() const { return initialized_; }

            /// @brief Get sparse format (CSR or BSR)
            /// @return current sparse format
            SparseFormat get_format() const { return format_; }

            /// @brief Get reference to sparse matrix data (host + device pointers)
            const GPUSparseBlockDiagData<T> &get_sparse_data() const { return sparse_data_; }

            /// @brief Get cusparse sparse matrix descriptor
            cusparseSpMatDescr_t get_descriptor() const { return mat_descr_; }

            /// @brief Allocate GPU memory for sparse matrix data
            /// @note Only allocates memory if host data exists and device memory hasn't been allocated yet
            void allocate_gpu_memory()
            {
                if (sparse_data_.d_values == nullptr && sparse_data_.h_values.size() > 0)
                {
                    get_cuda_error(cudaMalloc(&sparse_data_.d_values, sparse_data_.h_values.size() * sizeof(T)));
                }
                if (sparse_data_.d_row_offsets == nullptr && sparse_data_.h_row_offsets.size() > 0)
                {
                    get_cuda_error(cudaMalloc(&sparse_data_.d_row_offsets, sparse_data_.h_row_offsets.size() * sizeof(int)));
                }
                if (sparse_data_.d_col_indices == nullptr && sparse_data_.h_col_indices.size() > 0)
                {
                    get_cuda_error(cudaMalloc(&sparse_data_.d_col_indices, sparse_data_.h_col_indices.size() * sizeof(int)));
                }
            }

            /// @brief Create cuSPARSE matrix descriptor from sparse matrix data
            /// @param[in] cudart CUDA runtime instance
            /// @param[in] precision CUDA data type for the matrix values
            /// @note Must be called after copy2device() to ensure GPU data is available
            /// @note Uses lazy initialization; only recreates descriptor if it doesn't already exist
            void create_descriptor(const CudaRuntime &cudart, cudaDataType_t precision)
            {
                // If descriptor already exists, reuse it
                if (mat_descr_ != nullptr)
                {
                    return;
                }

                if (format_ == SparseFormat::BSR)
                {
                    // BSR format requires uniform block size
                    if (block_row_sizes_.empty() || block_col_sizes_.empty())
                    {
                        throw std::invalid_argument("create_descriptor: No block size information available for BSR format");
                    }

                    int blockDim_m = block_row_sizes_[0];
                    int blockDim_n = block_col_sizes_[0];
                    int64_t mb = m_ / blockDim_m;
                    int64_t nb = n_ / blockDim_n;

                    get_cusparse_error(cusparseCreateBsr(&mat_descr_,
                                                         mb, nb, num_blocks_,
                                                         blockDim_m, blockDim_n,
                                                         sparse_data_.d_row_offsets,
                                                         sparse_data_.d_col_indices,
                                                         sparse_data_.d_values,
                                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                         CUSPARSE_INDEX_BASE_ZERO,
                                                         precision,
                                                         CUSPARSE_ORDER_ROW));
                }
                else if (format_ == SparseFormat::CSR)
                {
                    get_cusparse_error(cusparseCreateCsr(&mat_descr_,
                                                         m_, n_, nnz_,
                                                         sparse_data_.d_row_offsets,
                                                         sparse_data_.d_col_indices,
                                                         sparse_data_.d_values,
                                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                         CUSPARSE_INDEX_BASE_ZERO, precision));
                }
                else
                {
                    throw std::invalid_argument("create_descriptor: Unsupported sparse format");
                }
            }

            /// @brief Reconstruct dense matrix from sparse data
            /// @return dense matrix representation with sparse elements expanded to full dimensions
            /// @note Supports both CSR and BSR formats; fills structural zeros appropriately
            Matrix<T> to_dense() const
            {
                // Initialize with zeros (for structural zeros in sparse layout)
                Matrix<T> dense(Shape{static_cast<unsigned int>(m_), static_cast<unsigned int>(n_)}, (T)0);

                if (format_ == SparseFormat::CSR)
                {
                    // Reconstruct from CSR format
                    const T* values = sparse_data_.h_values.data();
                    const int* col_indices = sparse_data_.h_col_indices.data();
                    const int* row_offsets = sparse_data_.h_row_offsets.data();
                    T* dense_data = dense.data();
                    int64_t dense_stride = m_; // Column-major stride

                    for (int row = 0; row < static_cast<int>(m_); ++row)
                    {
                        for (int idx = row_offsets[row]; idx < row_offsets[row + 1]; ++idx)
                        {
                            int col = col_indices[idx];
                            dense_data[col * dense_stride + row] = values[idx];
                        }
                    }
                }
                else if (format_ == SparseFormat::BSR)
                {
                    // Reconstruct from BSR format using block positions
                    T* dense_data = dense.data();
                    int64_t dense_stride = m_; // Column-major stride
                    const T* values = sparse_data_.h_values.data();

                    for (int block_idx = 0; block_idx < num_blocks_; ++block_idx)
                    {
                        int block_rows = block_row_sizes_[block_idx];
                        int block_cols = block_col_sizes_[block_idx];
                        int element_row = block_row_positions_[block_idx];
                        int element_col = block_col_positions_[block_idx];
                        int value_offset = sparse_data_.block_value_offsets[block_idx];

                        // Copy block data (stored row-major) to dense matrix (column-major)
                        const T* src = values + value_offset;
                        for (int i = 0; i < block_rows; ++i)
                        {
                            for (int j = 0; j < block_cols; ++j)
                            {
                                dense_data[((element_col + j) * dense_stride) + (element_row + i)] =
                                    src[i * block_cols + j];
                            }
                        }
                    }
                }

                return dense;
            }

        public:
            // Sparse matrix dimensions
            int64_t m_;           ///< Number of rows
            int64_t n_;           ///< Number of columns
            int64_t nnz_;         ///< Number of non-zero elements
            SparseFormat format_; ///< Selected sparse format (CSR or BSR)

            // Unified sparse matrix data structure (host + device)
            GPUSparseBlockDiagData<T> sparse_data_;

            // cuSPARSE descriptor and state
            cusparseSpMatDescr_t mat_descr_;
            bool initialized_;

            // Block metadata for reconstruction
            int num_blocks_;
            std::vector<int> block_row_sizes_;
            std::vector<int> block_col_sizes_;
            std::vector<int> row_offsets_;
            std::vector<int> col_offsets_;
            std::vector<int> block_row_positions_; ///< Element-space row positions for each block
            std::vector<int> block_col_positions_; ///< Element-space column positions for each block

            /// CSR only: index into the values array where row r of block b starts,
            /// stored at [sum of rows of blocks < b] + r
            std::vector<int> csr_value_offsets_;

            int device_id_ = -1; ///< CUDA device the data was uploaded to

            // Reusable cuSPARSE workspace (see get_workspace())
            mutable void *workspace_ = nullptr;
            mutable size_t workspace_size_ = 0;

            /// @brief Free all GPU memory and reset state
            /// @note Also destroys the cuSPARSE descriptor and clears block metadata
            void free_gpu_memory()
            {
                if (sparse_data_.d_values)
                {
                    get_cuda_error(cudaFree(sparse_data_.d_values));
                    sparse_data_.d_values = nullptr;
                }
                if (sparse_data_.d_row_offsets)
                {
                    get_cuda_error(cudaFree(sparse_data_.d_row_offsets));
                    sparse_data_.d_row_offsets = nullptr;
                }
                if (sparse_data_.d_col_indices)
                {
                    get_cuda_error(cudaFree(sparse_data_.d_col_indices));
                    sparse_data_.d_col_indices = nullptr;
                }

                if (mat_descr_ != nullptr)
                {
                    cusparseDestroySpMat(mat_descr_);
                    mat_descr_ = nullptr;
                }
                if (workspace_)
                {
                    get_cuda_error(cudaFree(workspace_));
                    workspace_ = nullptr;
                    workspace_size_ = 0;
                }

                block_row_positions_.clear();
                block_col_positions_.clear();
                initialized_ = false;
                m_ = 0;
                n_ = 0;
                nnz_ = 0;
            }

            /// @brief Convert GPU BlockMatrix to CSR (Compressed Sparse Row) format
            /// @param[in] cudart CUDA runtime for device operations
            /// @param[in] block_matrix source block matrix to convert
            /// @note Updates sparse_data_ member with CSR-formatted data and transfers to GPU
            /// @note Column indices within each row are sorted, as blocks are visited in column order
            void convert_to_csr_gpu(const CudaRuntime &cudart,
                                    const gpu::BlockMatrix_<T> &block_matrix)
            {
                int total_rows = static_cast<int>(m_);
                int total_nnz = static_cast<int>(nnz_);

                // Prepare sparse data in member variable
                sparse_data_.format = format_;
                sparse_data_.h_values.resize(total_nnz);
                sparse_data_.h_row_offsets.assign(total_rows + 1, 0);
                sparse_data_.h_col_indices.resize(total_nnz);

                // Count non-zeros per row
                for (int b = 0; b < num_blocks_; ++b)
                {
                    for (int r = 0; r < block_row_sizes_[b]; ++r)
                    {
                        sparse_data_.h_row_offsets[block_row_positions_[b] + r + 1] += block_col_sizes_[b];
                    }
                }
                std::partial_sum(sparse_data_.h_row_offsets.begin(), sparse_data_.h_row_offsets.end(),
                                 sparse_data_.h_row_offsets.begin());

                // Start of each block's rows in csr_value_offsets_
                std::vector<int> block_row_base(num_blocks_ + 1, 0);
                for (int b = 0; b < num_blocks_; ++b)
                {
                    block_row_base[b + 1] = block_row_base[b] + block_row_sizes_[b];
                }
                csr_value_offsets_.resize(block_row_base[num_blocks_]);

                // Visit blocks in column order so column indices are sorted within each row
                std::vector<int> order(num_blocks_);
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(),
                                 [&](int a, int b)
                                 {
                                     return block_col_positions_[a] < block_col_positions_[b];
                                 });

                std::vector<int> row_cursor(sparse_data_.h_row_offsets.begin(), sparse_data_.h_row_offsets.end() - 1);
                for (int b : order)
                {
                    for (int r = 0; r < block_row_sizes_[b]; ++r)
                    {
                        int row = block_row_positions_[b] + r;
                        int offset = row_cursor[row];
                        row_cursor[row] += block_col_sizes_[b];
                        csr_value_offsets_[block_row_base[b] + r] = offset;
                        for (int jj = 0; jj < block_col_sizes_[b]; ++jj)
                        {
                            sparse_data_.h_col_indices[offset + jj] = block_col_positions_[b] + jj;
                        }
                    }
                }

                pack_values(block_matrix);

                // Allocate and copy sparse matrix to the GPU
                get_cuda_error(cudaMalloc(&sparse_data_.d_values, sparse_data_.h_values.size() * sizeof(T)));
                get_cuda_error(cudaMalloc(&sparse_data_.d_row_offsets, sparse_data_.h_row_offsets.size() * sizeof(int)));
                get_cuda_error(cudaMalloc(&sparse_data_.d_col_indices, sparse_data_.h_col_indices.size() * sizeof(int)));
                copy2device(cudart);
                cudart.synchronize();
            }

            /// @brief Pack block values into sparse_data_.h_values using the current sparsity structure
            /// @param[in] block_matrix source block matrix with the same block structure as at construction
            void pack_values(const gpu::BlockMatrix_<T> &block_matrix)
            {
                int row_base = 0;
                for (int b = 0; b < num_blocks_; ++b)
                {
                    int block_rows = block_row_sizes_[b];
                    int block_cols = block_col_sizes_[b];
                    const T *block_data = block_matrix.get_block_data_at(block_row_positions_[b], block_col_positions_[b]);
                    if (block_data == nullptr)
                    {
                        throw std::logic_error("SparseMatrix::pack_values: block missing from source matrix");
                    }

                    if (format_ == SparseFormat::CSR)
                    {
                        // block_data is column-major: element (r, jj) at jj * block_rows + r
                        for (int r = 0; r < block_rows; ++r)
                        {
                            T *dst = sparse_data_.h_values.data() + csr_value_offsets_[row_base + r];
                            for (int jj = 0; jj < block_cols; ++jj)
                            {
                                dst[jj] = block_data[jj * block_rows + r];
                            }
                        }
                        row_base += block_rows;
                    }
                    else
                    {
                        // bsrValues is ROW-MAJOR within each block for CUSPARSE_ORDER_ROW
                        T *dst = sparse_data_.h_values.data() + sparse_data_.block_value_offsets[b];
                        for (int ii = 0; ii < block_rows; ++ii)
                            for (int jj = 0; jj < block_cols; ++jj)
                                dst[ii * block_cols + jj] = block_data[jj * block_rows + ii];
                    }
                }
            }

            /// @brief Convert GPU BlockMatrix to Block Sparse Row (BSR) format
            /// @param[in] cudart CUDA runtime for device operations
            /// @param[in] block_matrix source block matrix to convert
            /// @note Packs each block in ROW-MAJOR order internally
            /// @note BSR format requires uniform block sizes; falls back to CSR if blocks differ
            void convert_to_bsr_gpu(const CudaRuntime &cudart,
                                    const gpu::BlockMatrix_<T> &block_matrix)
            {
                int total_blocks = static_cast<int>(num_blocks_);

                int blockDim_rows = 0, blockDim_cols = 0;
                if (num_blocks_ > 0)
                {
                    Shape first_block_shape = block_matrix.get_block_shape(0);
                    blockDim_rows = static_cast<int>(first_block_shape.first);
                    blockDim_cols = static_cast<int>(first_block_shape.second);
                }

                int mb = (blockDim_rows > 0) ? static_cast<int>(m_) / blockDim_rows : 0;

                sparse_data_.format = format_;
                sparse_data_.h_values.resize(nnz_);
                sparse_data_.h_row_offsets.resize(mb + 1);       // size mb+1 (block rows), not total_rows+1
                sparse_data_.h_col_indices.resize(total_blocks); // one entry per BLOCK, not per row-match
                sparse_data_.block_value_offsets.resize(num_blocks_, -1); // Track offset for each original block

                // Group each block under its BLOCK-grid row: element_row / blockDim_rows
                std::vector<std::vector<size_t>> blocks_by_block_row(mb);
                for (size_t block_idx = 0; block_idx < block_matrix.num_blocks(); ++block_idx)
                {
                    int block_row_idx = block_row_positions_[block_idx] / blockDim_rows;
                    blocks_by_block_row[block_row_idx].push_back(block_idx);
                }

                sparse_data_.h_row_offsets[0] = 0;
                int current_block_idx = 0;
                int current_value_idx = 0;

                for (int br = 0; br < mb; ++br)
                {
                    auto &row_blocks = blocks_by_block_row[br];

                    // BSR requires column indices sorted within each block-row
                    std::sort(row_blocks.begin(), row_blocks.end(),
                              [&](size_t a, size_t b)
                              {
                                  return block_col_positions_[a] < block_col_positions_[b];
                              });

                    for (size_t block_idx : row_blocks)
                    {
                        // Track where this block's data starts in the values array
                        sparse_data_.block_value_offsets[block_idx] = current_value_idx;

                        // BLOCK-GRID column index (not element-space position)
                        int block_col_idx = block_col_positions_[block_idx] / blockDim_cols;
                        sparse_data_.h_col_indices[current_block_idx] = block_col_idx;

                        current_value_idx += block_row_sizes_[block_idx] * block_col_sizes_[block_idx];
                        current_block_idx++; // once per BLOCK, not once per row
                    }

                    sparse_data_.h_row_offsets[br + 1] = current_block_idx;
                }

                pack_values(block_matrix);

                // Allocate and copy sparse matrix to the GPU
                get_cuda_error(cudaMalloc(&sparse_data_.d_values, sparse_data_.h_values.size() * sizeof(T)));
                get_cuda_error(cudaMalloc(&sparse_data_.d_row_offsets, sparse_data_.h_row_offsets.size() * sizeof(int)));
                get_cuda_error(cudaMalloc(&sparse_data_.d_col_indices, sparse_data_.h_col_indices.size() * sizeof(int)));
                copy2device(cudart);
                cudart.synchronize();
            }

            /// @brief Implement GPU tensor interface: copy data to GPU device
            /// @param[in] cudart CUDA runtime instance
            void copy2device(const CudaRuntime &cudart) const override
            {
                SparseMatrix<T> *self = const_cast<SparseMatrix<T> *>(this);

                // Copy all data to device using async or sync based on runtime configuration
                if (cudart.asyncCopy())
                {
                    get_cuda_error(cudaMemcpyAsync(self->sparse_data_.d_values, self->sparse_data_.h_values.data(),
                                                   self->sparse_data_.h_values.size() * sizeof(T),
                                                   cudaMemcpyHostToDevice, cudart.getStream()));
                    get_cuda_error(cudaMemcpyAsync(self->sparse_data_.d_row_offsets, self->sparse_data_.h_row_offsets.data(),
                                                   self->sparse_data_.h_row_offsets.size() * sizeof(int),
                                                   cudaMemcpyHostToDevice, cudart.getStream()));
                    get_cuda_error(cudaMemcpyAsync(self->sparse_data_.d_col_indices, self->sparse_data_.h_col_indices.data(),
                                                   self->sparse_data_.h_col_indices.size() * sizeof(int),
                                                   cudaMemcpyHostToDevice, cudart.getStream()));
                }
                else
                {
                    get_cuda_error(cudaMemcpy(self->sparse_data_.d_values, self->sparse_data_.h_values.data(),
                                              self->sparse_data_.h_values.size() * sizeof(T),
                                              cudaMemcpyHostToDevice));
                    get_cuda_error(cudaMemcpy(self->sparse_data_.d_row_offsets, self->sparse_data_.h_row_offsets.data(),
                                              self->sparse_data_.h_row_offsets.size() * sizeof(int),
                                              cudaMemcpyHostToDevice));
                    get_cuda_error(cudaMemcpy(self->sparse_data_.d_col_indices, self->sparse_data_.h_col_indices.data(),
                                              self->sparse_data_.h_col_indices.size() * sizeof(int),
                                              cudaMemcpyHostToDevice));
                }
            }

            /// @brief Implement GPU tensor interface: copy data from GPU to host
            /// @param[in] cudart CUDA runtime instance
            void copy2host(const CudaRuntime &cudart) override
            {
                // Copy all data from device back to host using async or sync based on runtime configuration
                if (cudart.asyncCopy())
                {
                    get_cuda_error(cudaMemcpyAsync(sparse_data_.h_values.data(), sparse_data_.d_values,
                                                   sparse_data_.h_values.size() * sizeof(T),
                                                   cudaMemcpyDeviceToHost, cudart.getStream()));
                    get_cuda_error(cudaMemcpyAsync(sparse_data_.h_row_offsets.data(), sparse_data_.d_row_offsets,
                                                   sparse_data_.h_row_offsets.size() * sizeof(int),
                                                   cudaMemcpyDeviceToHost, cudart.getStream()));
                    get_cuda_error(cudaMemcpyAsync(sparse_data_.h_col_indices.data(), sparse_data_.d_col_indices,
                                                   sparse_data_.h_col_indices.size() * sizeof(int),
                                                   cudaMemcpyDeviceToHost, cudart.getStream()));
                }
                else
                {
                    get_cuda_error(cudaMemcpy(sparse_data_.h_values.data(), sparse_data_.d_values,
                                              sparse_data_.h_values.size() * sizeof(T),
                                              cudaMemcpyDeviceToHost));
                    get_cuda_error(cudaMemcpy(sparse_data_.h_row_offsets.data(), sparse_data_.d_row_offsets,
                                              sparse_data_.h_row_offsets.size() * sizeof(int),
                                              cudaMemcpyDeviceToHost));
                    get_cuda_error(cudaMemcpy(sparse_data_.h_col_indices.data(), sparse_data_.d_col_indices,
                                              sparse_data_.h_col_indices.size() * sizeof(int),
                                              cudaMemcpyDeviceToHost));
                }
            }

            /// @brief Implement GPU tensor interface: check if data is allocated on device
            /// @return true if sparse data is on GPU device
            bool alloc_on_device() const override
            {
                return initialized_ && (sparse_data_.d_values != nullptr);
            }

            /// @brief Implement GPU tensor interface: get pointer to GPU data (const)
            /// @return pointer to GPU device data
            T *gpu_data() const override
            {
                return sparse_data_.d_values;
            }

            /// @brief Implement GPU tensor interface: get pointer to GPU data (non-const)
            /// @return pointer to GPU device data
            T *gpu_data() override
            {
                return sparse_data_.d_values;
            }

            /// @brief Implement Tensor interface: get total number of elements (sparse: nnz)
            /// @return number of non-zero elements
            size_t size() override { return nnz_; }

            /// @brief Implement Tensor interface: get total number of elements (const)
            /// @return number of non-zero elements
            size_t size() const override { return nnz_; }

            /// @brief Implement Tensor interface: get pointer to host data
            /// @return pointer to sparse values array
            T *data() override { return sparse_data_.h_values.data(); }

            /// @brief Implement Tensor interface: get const pointer to host data
            /// @return const pointer to sparse values array
            T *data() const override { return const_cast<T *>(sparse_data_.h_values.data()); }

        private:
            /// @brief Values version of the source block matrix at the last sync (see BlockMatrix_::get_sparse)
            uint64_t source_values_version_ = 0;

            friend class BlockMatrix_<T>;
        };

        template <typename T>
        const SparseMatrix<T> &BlockMatrix_<T>::get_sparse(const CudaRuntime &cudart, SparseFormat format,
                                                           cudaDataType_t precision) const
        {
            std::shared_ptr<SparseMatrix<T>> &slot = sparse_cache_[static_cast<size_t>(format)];

            // Device data is only usable on the device it was uploaded to
            if (slot && slot->device_id() != cudart.device_id())
            {
                slot.reset();
            }

            if (!slot)
            {
                slot = std::make_shared<SparseMatrix<T>>(cudart, *this, format);
                ++sparse_build_count_;
            }
            else if (slot->source_values_version_ != values_version_)
            {
                slot->refresh_values(cudart, *this);
                ++sparse_values_refresh_count_;
            }
            slot->source_values_version_ = values_version_;

            slot->create_descriptor(cudart, precision);
            return *slot;
        }

    } // namespace gpu
} // namespace lahva
