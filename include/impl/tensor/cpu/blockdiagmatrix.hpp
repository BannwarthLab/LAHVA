/// @file blockdiagmatrix.hpp
/// @brief GPU-based block-diagonal matrix tensor implementation.
///
/// Provides the BlockDiagMatrix class for storing and manipulating block-diagonal matrices
/// on NVIDIA GPUs. Supports batched GEMM and GEMV operations via cuBLAS, automatic GPU memory
/// management with caching, and efficient workspace buffer allocation for performance.

#pragma once
#include "runtime.hpp"
#include "impl/tensor/gpu/gputensor.hpp"
#include "impl/tensor/gpu/sparse_format.hpp"
#include "impl/tensor/cpu/blockdiagmatrix.hpp"
#include <omp.h>
#include <algorithm>
#include <cstring>

namespace lahva
{
    namespace gpu
    {
        template <typename T>
        class BlockDiagMatrixSparse;

        /// @brief Helper struct for GPU block diagonal matrix data packing
        template<typename T>
        struct GPUBlockDiagData {
            T *d_data;           ///< Device pointer to padded packed blocks
            size_t padded_stride; ///< Stride for padded blocks
            size_t max_m;        ///< Maximum block row dimension
            size_t max_k;        ///< Maximum block column dimension
            int num_blocks;      ///< Number of blocks
            T *h_packed;         ///< Host pinned memory (for cleanup)
        };

        /// @brief Abstract base class for GPU block-diagonal matrices
        /// @tparam T data type for matrix elements
        template<typename T>
        class BlockDiagMatrix_ : public virtual GPUTensor_<T>
        {
        public:
            /// @brief Get total shape (rows, columns) of the block-diagonal matrix
            /// @return Shape object with total dimensions
            virtual Shape shape() const = 0;

            /// @brief Get number of blocks in the block-diagonal matrix
            /// @return Number of diagonal blocks
            virtual size_t num_blocks() const = 0;

            /// @brief Get shape of a specific diagonal block
            /// @param[in] idx Block index
            /// @return Shape object with block dimensions
            virtual Shape get_block_shape(size_t idx) const = 0;

            /// @brief Get raw data pointer for a specific diagonal block
            /// @param[in] idx Block index
            /// @return Const void pointer to block data in column-major format
            virtual const void* get_block_data(size_t idx) const = 0;

            /// @brief Get row offset array for all blocks
            /// @return Const reference to vector of cumulative row offsets
            virtual const std::vector<int>& get_row_offsets() const = 0;

            /// @brief Get column offset array for all blocks
            /// @return Const reference to vector of cumulative column offsets
            virtual const std::vector<int>& get_col_offsets() const = 0;

            /// @brief Element access operator (mutable)
            /// @param[in] i row index
            /// @param[in] j column index
            /// @return reference to element at (i,j)
            virtual T &operator()(size_t i, size_t j) = 0;

            /// @brief Element access operator (const)
            /// @param[in] i row index
            /// @param[in] j column index
            /// @return const reference to element at (i,j)
            virtual const T &operator()(size_t i, size_t j) const = 0;
        };

        /// @brief GPU-accelerated block-diagonal matrix with batched operations.
        ///
        /// Stores and manipulates block-diagonal matrices on NVIDIA GPUs with automatic
        /// GPU memory management and caching. The host representation maintains matrices
        /// in CPU memory, while an optional GPU cache stores padded/packed blocks for
        /// efficient cuBLAS batched operations. GPU cache is invalidated when host data
        /// is modified, ensuring correctness while maximizing GPU utilization.
        ///
        /// @tparam T Numeric element type (double, float, complex types)
        /// @tparam Allocator Host (pinned) memory allocator (default: CudaHostAllocator<T>)
        /// @tparam GPUAllocator Device memory allocator (default: CudaDeviceAllocator<T>)
        template <class T, class Allocator = CudaHostAllocator<T>, class GPUAllocator = CudaDeviceAllocator<T>>
        class BlockDiagMatrix : virtual public GPUTensor<T, Allocator, GPUAllocator>, virtual public BlockDiagMatrix_<T>
        {
            using alloc_ptr = CPUAllocator<T>;
            using gpualloc_ptr = GPUAllocator_<T>;
        protected:
            /// @brief Total number of rows (sum of all block rows)
            size_t n_rows_ = 0;

            /// @brief Total number of columns (sum of all block columns)
            size_t n_cols_ = 0;

            /// @brief Collection of diagonal block matrices (host memory)
            std::vector<Matrix<T, Allocator>> matrices;

            /// @brief Row dimensions of each diagonal block
            std::vector<size_t> block_rows_;

            /// @brief Column dimensions of each diagonal block
            std::vector<size_t> block_cols_;

            /// @brief Cumulative row offsets for each block (allows O(log n) block lookup)
            std::vector<int> row_offsets_;

            /// @brief Cumulative column offsets for each block (allows O(log n) block lookup)
            std::vector<int> col_offsets_;

            /// @brief Cached GPU block-diagonal data (padded and packed format)
            /// @note nullptr if GPU cache is invalid; managed via free_gpu_cache()
            mutable GPUBlockDiagData<T> *gpu_data_ = nullptr;

            /// @brief Whether the GPU cache (gpu_data_) is valid and up-to-date with host
            mutable bool gpu_data_valid_ = false;

            /// @brief GPU workspace buffer for batch operations (device memory)
            mutable T *gpu_workspace_ = nullptr;

            /// @brief Host workspace buffer pinned for efficient GPU transfers
            mutable T *host_workspace_ = nullptr;

            /// @brief Current size of allocated workspace in number of elements
            mutable size_t workspace_size_ = 0;

        public:
            /// @brief Default constructor - creates empty block-diagonal matrix
            BlockDiagMatrix() {};

            /// @brief Destructor - frees GPU memory and workspace buffers
            virtual ~BlockDiagMatrix() {
                if (gpu_data_ != nullptr) {
                    free_gpu_data(*gpu_data_);
                    delete gpu_data_;
                }
                if (gpu_workspace_ != nullptr) {
                    get_cuda_error(cudaFree(gpu_workspace_));
                }
                if (host_workspace_ != nullptr) {
                    get_cuda_error(cudaFreeHost(host_workspace_));
                }
            }

            /// @brief Copy constructor
            /// @param[in] other block-diagonal matrix to copy
            /// @note does not copy GPU cache; GPU cache is marked invalid
            BlockDiagMatrix(const BlockDiagMatrix &other) :
                n_rows_{other.n_rows_}, n_cols_{other.n_cols_},
                matrices{other.matrices},
                block_rows_{other.block_rows_},
                block_cols_{other.block_cols_},
                row_offsets_{other.row_offsets_},
                col_offsets_{other.col_offsets_}
            {}

            /// @brief Copy assignment operator
            /// @param[in] other block-diagonal matrix to copy from
            /// @return reference to this matrix
            /// @note invalidates GPU cache on assignment
            BlockDiagMatrix &operator=(const BlockDiagMatrix &other)
            {
                if (this != &other)
                {
                    free_gpu_cache();
                    n_rows_ = other.n_rows_;
                    n_cols_ = other.n_cols_;
                    matrices = other.matrices;
                    block_rows_ = other.block_rows_;
                    block_cols_ = other.block_cols_;
                    row_offsets_ = other.row_offsets_;
                    col_offsets_ = other.col_offsets_;
                }
                return *this;
            }

            /// @brief Move constructor
            /// @param[in] other block-diagonal matrix to move from
            /// @note transfers GPU cache ownership to this matrix
            BlockDiagMatrix(BlockDiagMatrix &&other) noexcept :
                n_rows_{other.n_rows_}, n_cols_{other.n_cols_},
                matrices{std::move(other.matrices)},
                block_rows_{std::move(other.block_rows_)},
                block_cols_{std::move(other.block_cols_)},
                row_offsets_{std::move(other.row_offsets_)},
                col_offsets_{std::move(other.col_offsets_)},
                gpu_data_{other.gpu_data_},
                gpu_data_valid_{other.gpu_data_valid_}
            {
                other.n_rows_ = 0;
                other.n_cols_ = 0;
                other.gpu_data_ = nullptr;
                other.gpu_data_valid_ = false;
            }

            /// @brief Move assignment operator
            /// @param[in] other block-diagonal matrix to move from
            /// @return reference to this matrix
            /// @note invalidates existing GPU cache; transfers GPU cache from other
            BlockDiagMatrix &operator=(BlockDiagMatrix &&other) noexcept
            {
                if (this != &other)
                {
                    free_gpu_cache();
                    n_rows_ = other.n_rows_;
                    n_cols_ = other.n_cols_;
                    matrices = std::move(other.matrices);
                    block_rows_ = std::move(other.block_rows_);
                    block_cols_ = std::move(other.block_cols_);
                    row_offsets_ = std::move(other.row_offsets_);
                    col_offsets_ = std::move(other.col_offsets_);
                    gpu_data_ = other.gpu_data_;
                    gpu_data_valid_ = other.gpu_data_valid_;
                    other.n_rows_ = 0;
                    other.n_cols_ = 0;
                    other.gpu_data_ = nullptr;
                    other.gpu_data_valid_ = false;
                }
                return *this;
            }

            /// @brief Create block-diagonal matrix with uniform blocks
            /// @param[in] n_blocks number of diagonal blocks to create
            /// @param[in] block_shape shape (rows, cols) for each block (all blocks have same shape)
            BlockDiagMatrix(size_t n_blocks, Shape block_shape)
            {
                matrices.reserve(n_blocks);
                block_rows_.reserve(n_blocks);
                block_cols_.reserve(n_blocks);
                for (size_t i = 0; i < n_blocks; ++i)
                    matrices.emplace_back(block_shape);
                build_offsets_();
            }

            /// @brief Create block-diagonal matrix with uniform blocks initialized to a value
            /// @param[in] n_blocks number of diagonal blocks to create
            /// @param[in] block_shape shape (rows, cols) for each block
            /// @param[in] val initialization value for all elements
            BlockDiagMatrix(size_t n_blocks, Shape block_shape, T val)
            {
                matrices.reserve(n_blocks);
                block_rows_.reserve(n_blocks);
                block_cols_.reserve(n_blocks);
                for (size_t i = 0; i < n_blocks; ++i)
                    matrices.emplace_back(block_shape, val);
                build_offsets_();
            }

            /// @brief Create block-diagonal matrix with varying block shapes
            /// @param[in] shapes vector of (rows, cols) pairs for each diagonal block
            /// @note blocks are uninitialized; use constructor with value parameter for initialization
            explicit BlockDiagMatrix(const std::vector<Shape> &shapes)
            {
                matrices.reserve(shapes.size());
                block_rows_.reserve(shapes.size());
                block_cols_.reserve(shapes.size());
                for (const auto &s : shapes)
                    matrices.emplace_back(s);
                build_offsets_();
            }

            /// @brief Create block-diagonal matrix with varying block shapes and uniform initialization
            /// @param[in] shapes vector of (rows, cols) pairs for each diagonal block
            /// @param[in] val initialization value for all block elements
            BlockDiagMatrix(const std::vector<Shape> &shapes, T val)
            {
                matrices.reserve(shapes.size());
                block_rows_.reserve(shapes.size());
                block_cols_.reserve(shapes.size());
                for (const auto &s : shapes)
                    matrices.emplace_back(s, val);
                build_offsets_();
            }

            /// @brief Create block-diagonal matrix from existing block matrices
            /// @param[in] blocks vector of Matrix objects to use as diagonal blocks
            /// @note takes a copy of the input vector; blocks are not modified
            explicit BlockDiagMatrix(const std::vector<Matrix<T, Allocator>> &blocks)
                : matrices(blocks)
            {
                block_rows_.reserve(matrices.size());
                block_cols_.reserve(matrices.size());
                build_offsets_();
            }

            /// @brief Create block-diagonal matrix from existing block matrices (move semantics)
            /// @param[in] blocks vector of Matrix objects to move as diagonal blocks
            /// @note blocks vector is moved into this matrix; contents of input vector are transferred
            explicit BlockDiagMatrix(std::vector<Matrix<T, Allocator>> &&blocks)
                : matrices(std::move(blocks))
            {
                block_rows_.reserve(matrices.size());
                block_cols_.reserve(matrices.size());
                build_offsets_();
            }

            /// @brief Add scalar to all block elements (in-place)
            /// @param[in] val value to add to each element
            /// @return reference to this matrix
            /// @note invalidates GPU cache
            BlockDiagMatrix &operator+=(T val)
            {
                free_gpu_cache();
                for (auto &m : matrices)
                    m += val;
                return *this;
            }

            /// @brief Get total shape of block-diagonal matrix
            /// @return Shape with total rows and columns across all blocks
            virtual Shape shape() const override { return Shape{n_rows_, n_cols_}; }

            /// @brief Get shapes of all blocks as vector of vectors
            /// @return Vector of [rows, cols] pairs for each diagonal block
            std::vector<std::vector<size_t>> block_shapes() const
            {
                std::vector<std::vector<size_t>> block_shapes;
                for (size_t i = 0; i < matrices.size(); i++) {
                    Shape s = matrices[i].shape();
                    block_shapes.push_back({s.first, s.second});
                }
                return block_shapes;
            }

            /// @brief Get number of diagonal blocks
            /// @return Count of blocks in matrix
            virtual size_t num_blocks() const override { return matrices.size(); }

            /// @brief Get const reference to a specific block matrix
            /// @param[in] idx block index
            /// @return const reference to the block matrix
            const Matrix<T, Allocator>& get_block(size_t idx) const { return matrices[idx]; }

            /// @brief Get cumulative row offsets for block access
            /// @return const reference to vector of row offsets (element i+1 is where block i ends)
            virtual const std::vector<int>& get_row_offsets() const override { return row_offsets_; }

            /// @brief Get cumulative column offsets for block access
            /// @return const reference to vector of column offsets (element i+1 is where block i ends)
            virtual const std::vector<int>& get_col_offsets() const override { return col_offsets_; }

            /// @brief Get shape of a specific block
            /// @param[in] idx block index
            /// @return Shape object with block dimensions
            virtual Shape get_block_shape(size_t idx) const override {
                return matrices[idx].shape();
            }

            /// @brief Get raw data pointer for a specific block
            /// @param[in] idx block index
            /// @return const void pointer to block data in column-major format
            virtual const void* get_block_data(size_t idx) const override {
                return static_cast<const void*>(matrices[idx].data());
            }

            /// @brief Get row dimensions of all blocks
            /// @return const reference to vector of block row dimensions
            const std::vector<size_t>& get_block_rows() const { return block_rows_; }

            /// @brief Get column dimensions of all blocks
            /// @return const reference to vector of block column dimensions
            const std::vector<size_t>& get_block_cols() const { return block_cols_; }

            /// @brief Extract main diagonals from all blocks as concatenated vector
            /// @return Vector containing diagonals of each block concatenated sequentially
            /// @note vector size equals sum of min(block_rows, block_cols) over all blocks
            cpu::Vector<T, Allocator> get_diagonal() const
            {
                size_t total = 0;
                for (const auto &m : matrices)
                {
                    Shape s = m.shape();
                    total += std::min(s.first, s.second);
                }
                cpu::Vector<T, Allocator> diag(total);
                size_t offset = 0;
                for (const auto &m : matrices)
                {
                    Shape s = m.shape();
                    size_t min_dim = std::min(s.first, s.second);
                    for (size_t i = 0; i < min_dim; i++)
                        diag[offset + i] = m(i, i);
                    offset += min_dim;
                }
                return diag;
            }

            /// @brief Set main diagonals of all blocks from concatenated vector
            /// @param[in] diag vector of concatenated diagonal values (must match total diagonal size)
            /// @note uses OMP parallelization for block-wise diagonal updates; invalidates GPU cache
            void set_diagonal(const cpu::Vector<T, Allocator> &diag)
            {
                free_gpu_cache();
                size_t offset = 0;
                for (auto &m : matrices)
                {
                    Shape s = m.shape();
                    size_t min_dim = std::min(s.first, s.second);
#pragma omp for
                    for (size_t i = 0; i < min_dim; i++)
                        m(i, i) = diag[offset + i];
                    offset += min_dim;
                }
            }

            /// @brief Symmetrize each block in-place: block = (block + block^T) / 2
            /// @note applies symmetrization to each diagonal block independently; invalidates GPU cache
            void symmetrize()
            {
                free_gpu_cache();
                for (auto &m : matrices)
                    m.symmetrize();
            }

            /// @brief Element access operator (mutable)
            /// @param[in] i row index
            /// @param[in] j column index
            /// @return reference to element at (i,j); returns zero for off-diagonal positions
            virtual T &operator()(size_t i, size_t j) override {
                for (size_t b = 0; b < matrices.size(); ++b) {
                    if (i >= row_offsets_[b] && i < row_offsets_[b] + (int)matrices[b].shape().first &&
                        j >= col_offsets_[b] && j < col_offsets_[b] + (int)matrices[b].shape().second) {
                        return matrices[b](i - row_offsets_[b], j - col_offsets_[b]);
                    }
                }
                static T zero{};
                return zero;
            }

            /// @brief Element access operator (const)
            /// @param[in] i row index
            /// @param[in] j column index
            /// @return const reference to element at (i,j); returns zero for off-diagonal positions
            virtual const T &operator()(size_t i, size_t j) const override {
                for (size_t b = 0; b < matrices.size(); ++b) {
                    if (i >= row_offsets_[b] && i < row_offsets_[b] + (int)matrices[b].shape().first &&
                        j >= col_offsets_[b] && j < col_offsets_[b] + (int)matrices[b].shape().second) {
                        return matrices[b](i - row_offsets_[b], j - col_offsets_[b]);
                    }
                }
                static const T zero{};
                return zero;
            }

            /// @brief Add a new block as the next diagonal block
            /// @param[in] block matrix block to append
            /// @note updates matrix dimensions and offset arrays for block lookup
            void add_block(const Matrix<T, Allocator> &block) {
                matrices.push_back(block);
                block_rows_.push_back(block.shape().first);
                block_cols_.push_back(block.shape().second);
                n_rows_ += block.shape().first;
                n_cols_ += block.shape().second;

                if (col_offsets_.empty()) {
                    col_offsets_.push_back(0);
                    row_offsets_.push_back(0);
                }
                col_offsets_.push_back(col_offsets_.back() + block.shape().second);
                row_offsets_.push_back(row_offsets_.back() + block.shape().first);
            }

            /// @brief Print all blocks to standard output
            /// @note calls print() on each block matrix sequentially
            void print() const {
                for (size_t i=0; i < matrices.size(); i++) {
                    matrices[i].print();
                }
            }

            /// @brief Print all blocks to a file
            /// @param[in] file filename to write block data to
            /// @note calls print(file) on each block matrix sequentially
            void print(const char* file) const
            {
                for (size_t i=0; i < matrices.size(); i++) {
                    matrices[i].print(file);
                }
            }

            /// @brief Prepare block-diagonal matrix for GPU operations (copy and allocate workspace)
            /// @param[in] cudart CUDA runtime for device selection and memory allocation
            /// @note copies host blocks to padded GPU format and allocates workspace buffers
            void to_gpu(const CudaRuntime &cudart) const {
                ensure_on_gpu();
                size_t estimated_workspace = static_cast<size_t>(max_block_size()) * 10;
                ensure_workspace(cudart, estimated_workspace);
            }

            /// @brief Get estimated GPU memory requirement for packed blocks
            /// @return Total size in elements: (max_row * max_col * num_blocks)
            size_t max_block_size() const {
                if (matrices.empty()) return 0;
                size_t max_m = *std::max_element(block_rows_.begin(), block_rows_.end());
                size_t max_k = *std::max_element(block_cols_.begin(), block_cols_.end());
                return max_m * max_k * num_blocks();
            }

            /// @brief Ensure block-diagonal matrix data is copied to GPU
            /// @return const reference to GPU block data structure
            /// @note lazily copies host blocks to padded GPU format on first call
            const GPUBlockDiagData<T>& ensure_on_gpu() const {
                if (!gpu_data_valid_) {
                    copy_to_gpu_impl();
                }
                return *gpu_data_;
            }

            /// @brief Check whether GPU cache is valid and up-to-date
            /// @return true if GPU data matches host data, false if cache is stale
            bool is_on_gpu() const {
                return gpu_data_valid_;
            }

            /// @brief Free GPU cache memory and mark cache as invalid
            /// @note GPU data will be regenerated on next ensure_on_gpu() call
            void free_gpu_cache() {
                if (gpu_data_ != nullptr) {
                    free_gpu_data(*gpu_data_);
                    delete gpu_data_;
                    gpu_data_ = nullptr;
                }
                gpu_data_valid_ = false;
            }

            /// @brief Ensure GPU and host workspace buffers are allocated
            /// @param[in] cudart CUDA runtime for device selection
            /// @param[in] num_elems number of elements to allocate (both GPU and host)
            /// @return pair of (gpu_ptr, host_pinned_ptr) workspace buffers
            /// @note reallocates if requested size exceeds current allocation
            std::pair<T*, T*> ensure_workspace(const CudaRuntime &cudart, size_t num_elems) const;

            /// @brief Pack dense matrix blocks into padded workspace for batch GEMM
            /// @param[in] cudart CUDA runtime for device selection
            /// @param[in] src source dense matrix to extract blocks from
            /// @param[in] offsets block row/column starting positions in src
            /// @param[in] block_sizes size of each block
            /// @param[in] padded_dim padded dimension (row or column) for alignment
            /// @param[in] transpose if true, extract column-major blocks; otherwise row-major
            /// @return pair of (gpu_workspace, host_workspace) with packed data
            /// @note allocates workspace via ensure_workspace(); data copied to GPU
            std::pair<T*, T*> pack_batch_buffers(
                const CudaRuntime &cudart,
                const Matrix_<T> &src,
                const std::vector<int> &offsets,
                const std::vector<size_t> &block_sizes,
                int padded_dim,
                bool transpose = false) const;

            /// @brief Pack dense matrix columns into padded workspace for batch operations
            /// @param[in] cudart CUDA runtime for device selection
            /// @param[in] src source dense matrix to extract column blocks from
            /// @param[in] offsets column starting positions in src
            /// @param[in] block_sizes column count for each block
            /// @param[in] padded_cols padded column dimension for alignment
            /// @return pair of (gpu_workspace, host_workspace) with packed column data
            std::pair<T*, T*> pack_batch_buffers_cols(
                const CudaRuntime &cudart,
                const Matrix_<T> &src,
                const std::vector<int> &offsets,
                const std::vector<size_t> &block_sizes,
                int padded_cols) const;

            /// @brief Unpack batch GEMM result from padded workspace into dense matrix
            /// @param[in,out] dst destination dense matrix to scatter blocks into
            /// @param[in] dst_rows row dimension of dst
            /// @param[in] n number of columns in result
            /// @param[in] src padded result blocks from batch operation
            /// @param[in] offsets block row starting positions in dst
            /// @param[in] block_sizes row count for each block
            /// @param[in] padded_rows padded row dimension of src blocks
            void unpack_batch_result(
                Matrix_<T> &dst,
                long long dst_rows,
                int n,
                const T *src,
                const std::vector<int> &offsets,
                const std::vector<size_t> &block_sizes,
                int padded_rows) const;

            /// @brief Unpack batch result from padded column blocks into dense matrix
            /// @param[in,out] dst destination dense matrix to scatter column blocks into
            /// @param[in] dst_rows row dimension of dst
            /// @param[in] src padded column blocks from batch operation
            /// @param[in] offsets column starting positions in dst
            /// @param[in] block_sizes column count for each block
            /// @param[in] padded_cols padded column dimension of src blocks
            void unpack_batch_result_cols(
                Matrix_<T> &dst,
                long long dst_rows,
                const T *src,
                const std::vector<int> &offsets,
                const std::vector<size_t> &block_sizes,
                int padded_cols) const;

            /// @brief Pack vector blocks into GPU-ready format with padding
            /// @param[in] cudart CUDA runtime for device selection
            /// @param[in] src source vector with scattered block elements
            /// @param[in,out] dst destination padded vector in GPU memory
            /// @param[in] offsets block starting positions in src
            /// @param[in] block_sizes size of each block
            /// @note padded to max block size for efficient batch operations
            void pack_vector_to_gpu(
                const CudaRuntime &cudart,
                const Vector<T> &src,
                Vector<T> &dst,
                const std::vector<int> &offsets,
                const std::vector<size_t> &block_sizes) const;

            /// @brief Unpack padded GPU vector blocks back to scattered vector
            /// @param[in] cudart CUDA runtime for device selection
            /// @param[in] src padded vector from GPU operations
            /// @param[in,out] dst destination vector with scattered block elements
            /// @param[in] offsets block starting positions in dst
            /// @param[in] block_sizes size of each block
            void unpack_vector_from_gpu(
                const CudaRuntime &cudart,
                Vector<T> &src,
                Vector<T> &dst,
                const std::vector<int> &offsets,
                const std::vector<size_t> &block_sizes) const;

        private:
            /// @brief Validate block dimensions do not exceed maximum representable size
            /// @param[in] n_rows total number of rows
            /// @param[in] n_cols total number of columns
            /// @throws std::out_of_range if dimensions would overflow size_t
            static void check_size_(size_t n_rows, size_t n_cols)
            {
                if (n_rows != 0 && n_cols > SIZE_MAX / n_rows)
                    throw std::out_of_range("Block size exceeds maximum representable size.");
            }

            /// @brief Rebuild cumulative offset arrays from block shapes
            /// @note called after adding blocks; computes row_offsets_, col_offsets_, block_rows_, block_cols_, and total dimensions
            void build_offsets_()
            {
                row_offsets_.reserve(matrices.size() + 1);
                col_offsets_.reserve(matrices.size() + 1);
                row_offsets_.push_back(0);
                col_offsets_.push_back(0);
                for (const auto &m : matrices)
                {
                    Shape s = m.shape();
                    n_rows_ += s.first;
                    n_cols_ += s.second;
                    block_rows_.push_back(s.first);
                    block_cols_.push_back(s.second);
                    row_offsets_.push_back(row_offsets_.back() + static_cast<int>(s.first));
                    col_offsets_.push_back(col_offsets_.back() + static_cast<int>(s.second));
                }
            }

            /// @brief Copy host blocks to GPU in padded format for efficient batch operations
            /// @note private implementation; called lazily by ensure_on_gpu()
            /// @details pads all blocks to max_m x max_k and copies to device memory with pinned host buffer
            void copy_to_gpu_impl() const {
                int num_blocks = matrices.size();

                size_t max_m = *std::max_element(block_rows_.begin(), block_rows_.end());
                size_t max_k = *std::max_element(block_cols_.begin(), block_cols_.end());

                long long padded_stride = (long long)max_m * max_k;

                T *h_packed;
                get_cuda_error(cudaHostAlloc(&h_packed, padded_stride * num_blocks * sizeof(T), cudaHostAllocDefault));

                std::memset(h_packed, 0, padded_stride * num_blocks * sizeof(T));

                for (int i = 0; i < num_blocks; ++i) {
                    const Matrix<T, Allocator>& block = matrices[i];
                    size_t block_m = block.shape().first;
                    size_t block_k = block.shape().second;

                    for (size_t j = 0; j < block_k; ++j) {
                        std::memcpy(
                            h_packed + i * padded_stride + j * max_m,
                            block.data() + j * block_m,
                            block_m * sizeof(T)
                        );
                    }
                }

                T *d_data;
                get_cuda_error(cudaMalloc(&d_data, padded_stride * num_blocks * sizeof(T)));
                get_cuda_error(cudaMemcpy(d_data, h_packed, padded_stride * num_blocks * sizeof(T), cudaMemcpyHostToDevice));

                gpu_data_ = new GPUBlockDiagData<T>{d_data, (size_t)padded_stride, max_m, max_k, num_blocks, h_packed};
                gpu_data_valid_ = true;
            }

        public:
            /// @brief Free GPU block data structures and associated device memory
            /// @param[in] gpu_data GPU data structure with device and host pinned pointers
            /// @note called automatically by destructor and free_gpu_cache()
            static void free_gpu_data(GPUBlockDiagData<T>& gpu_data) {
                get_cuda_error(cudaFree(gpu_data.d_data));
                get_cuda_error(cudaFreeHost(gpu_data.h_packed));
            }
    };

    template<typename T, typename Allocator, typename GPUAllocator>
    inline std::pair<T*, T*> BlockDiagMatrix<T, Allocator, GPUAllocator>::ensure_workspace(
        const CudaRuntime &cudart, size_t num_elems) const
    {
        if (workspace_size_ < num_elems) {
            if (gpu_workspace_ != nullptr) {
                get_cuda_error(cudaFree(gpu_workspace_));
            }
            if (host_workspace_ != nullptr) {
                get_cuda_error(cudaFreeHost(host_workspace_));
            }

            get_cuda_error(cudaSetDevice(cudart.device_id()));
            get_cuda_error(cudaMalloc(&gpu_workspace_, num_elems * sizeof(T)));
            get_cuda_error(cudaHostAlloc(&host_workspace_, num_elems * sizeof(T), cudaHostAllocDefault));
            workspace_size_ = num_elems;
        }
        return {gpu_workspace_, host_workspace_};
    }

    template<typename T, typename Allocator, typename GPUAllocator>
    inline std::pair<T*, T*> BlockDiagMatrix<T, Allocator, GPUAllocator>::pack_batch_buffers(
        const CudaRuntime &cudart,
        const Matrix_<T> &src,
        const std::vector<int> &offsets,
        const std::vector<size_t> &block_sizes,
        int padded_dim,
        bool transpose) const
    {
        int num_blocks = static_cast<int>(block_sizes.size());
        long long src_rows = static_cast<long long>(src.shape().first);
        long long src_cols = static_cast<long long>(src.shape().second);

        size_t total_elems;

        if (!transpose) {
            long long stride = static_cast<long long>(padded_dim) * src_cols;
            total_elems = static_cast<size_t>(stride) * num_blocks;

            auto [d_packed, h_packed] = ensure_workspace(cudart, total_elems);
            std::memset(h_packed, 0, total_elems * sizeof(T));

            for (int i = 0; i < num_blocks; ++i) {
                long long row_off = offsets[i];
                size_t rows = block_sizes[i];
                for (long long j = 0; j < src_cols; ++j) {
                    std::memcpy(
                        h_packed + i * stride + j * padded_dim,
                        src.data() + j * src_rows + row_off,
                        rows * sizeof(T));
                }
            }

            get_cuda_error(cudaSetDevice(cudart.device_id()));
            get_cuda_error(cudaMemcpy(d_packed, h_packed, total_elems * sizeof(T), cudaMemcpyHostToDevice));
            return {d_packed, h_packed};
        } else {
            long long stride = src_rows * padded_dim;
            total_elems = static_cast<size_t>(stride) * num_blocks;

            auto [d_packed, h_packed] = ensure_workspace(cudart, total_elems);
            std::memset(h_packed, 0, total_elems * sizeof(T));

            for (int i = 0; i < num_blocks; ++i) {
                long long col_off = offsets[i];
                size_t cols = block_sizes[i];
                for (size_t j = 0; j < cols; ++j) {
                    std::memcpy(
                        h_packed + i * stride + j * src_rows,
                        src.data() + (col_off + j) * src_rows,
                        src_rows * sizeof(T));
                }
            }

            get_cuda_error(cudaSetDevice(cudart.device_id()));
            get_cuda_error(cudaMemcpy(d_packed, h_packed, total_elems * sizeof(T), cudaMemcpyHostToDevice));
            return {d_packed, h_packed};
        }
    }

    template<typename T, typename Allocator, typename GPUAllocator>
    inline std::pair<T*, T*> BlockDiagMatrix<T, Allocator, GPUAllocator>::pack_batch_buffers_cols(
        const CudaRuntime &cudart,
        const Matrix_<T> &src,
        const std::vector<int> &offsets,
        const std::vector<size_t> &block_sizes,
        int padded_cols) const
    {
        int num_blocks = static_cast<int>(block_sizes.size());
        long long src_rows = static_cast<long long>(src.shape().first);
        long long stride = src_rows * padded_cols;
        size_t total_elems = static_cast<size_t>(stride) * num_blocks;

        auto [d_packed, h_packed] = ensure_workspace(cudart, total_elems);
        std::memset(h_packed, 0, total_elems * sizeof(T));

        for (int i = 0; i < num_blocks; ++i) {
            long long col_off = offsets[i];
            size_t cols = block_sizes[i];
            for (size_t j = 0; j < cols; ++j) {
                std::memcpy(
                    h_packed + i * stride + j * src_rows,
                    src.data() + (col_off + j) * src_rows,
                    src_rows * sizeof(T));
            }
        }

        get_cuda_error(cudaSetDevice(cudart.device_id()));
        get_cuda_error(cudaMemcpy(d_packed, h_packed, total_elems * sizeof(T), cudaMemcpyHostToDevice));
        return {d_packed, h_packed};
    }

    template<typename T, typename Allocator, typename GPUAllocator>
    inline void BlockDiagMatrix<T, Allocator, GPUAllocator>::unpack_batch_result(
        Matrix_<T> &dst,
        long long dst_rows,
        int n,
        const T *src,
        const std::vector<int> &offsets,
        const std::vector<size_t> &block_sizes,
        int padded_rows) const
    {
        int num_blocks = static_cast<int>(block_sizes.size());
        long long padded_stride = static_cast<long long>(padded_rows) * n;

        for (int i = 0; i < num_blocks; ++i) {
            long long row_off = offsets[i];
            size_t rows = block_sizes[i];
            for (int j = 0; j < n; ++j) {
                std::memcpy(
                    dst.data() + j * dst_rows + row_off,
                    src + i * padded_stride + j * padded_rows,
                    rows * sizeof(T));
            }
        }
    }

    template<typename T, typename Allocator, typename GPUAllocator>
    inline void BlockDiagMatrix<T, Allocator, GPUAllocator>::unpack_batch_result_cols(
        Matrix_<T> &dst,
        long long dst_rows,
        const T *src,
        const std::vector<int> &offsets,
        const std::vector<size_t> &block_sizes,
        int padded_cols) const
    {
        int num_blocks = static_cast<int>(block_sizes.size());
        long long padded_stride = static_cast<long long>(dst_rows) * padded_cols;

        for (int i = 0; i < num_blocks; ++i) {
            long long col_off = offsets[i];
            size_t cols = block_sizes[i];
            for (size_t j = 0; j < cols; ++j) {
                std::memcpy(
                    dst.data() + (col_off + j) * dst_rows,
                    src + i * padded_stride + j * dst_rows,
                    dst_rows * sizeof(T));
            }
        }
    }

    template<typename T, typename Allocator, typename GPUAllocator>
    inline void BlockDiagMatrix<T, Allocator, GPUAllocator>::pack_vector_to_gpu(
        const CudaRuntime &cudart,
        const Vector<T> &src,
        Vector<T> &dst,
        const std::vector<int> &offsets,
        const std::vector<size_t> &block_sizes) const
    {
        int num_blocks = this->num_blocks();
        size_t max_size = *std::max_element(block_sizes.begin(), block_sizes.end());

        for (int i = 0; i < num_blocks; ++i) {
            long long offset = offsets[i];
            size_t block_size = block_sizes[i];

            std::memcpy(
                dst.data() + i * max_size,
                src.data() + offset,
                block_size * sizeof(T)
            );
        }

        dst.allocateGPU(cudart);
        dst.copy2device(cudart);
    }

    template<typename T, typename Allocator, typename GPUAllocator>
    inline void BlockDiagMatrix<T, Allocator, GPUAllocator>::unpack_vector_from_gpu(
        const CudaRuntime &cudart,
        Vector<T> &src,
        Vector<T> &dst,
        const std::vector<int> &offsets,
        const std::vector<size_t> &block_sizes) const
    {
        src.copy2host(cudart);

        int num_blocks = this->num_blocks();
        size_t max_size = *std::max_element(block_sizes.begin(), block_sizes.end());

        for (int i = 0; i < num_blocks; ++i) {
            long long offset = offsets[i];
            size_t block_size = block_sizes[i];

            std::memcpy(
                dst.data() + offset,
                src.data() + i * max_size,
                block_size * sizeof(T)
            );
        }
    }

    } // namespace gpu
} // namespace lahva
