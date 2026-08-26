/// @file block-diag-matrix.hpp
/// @brief GPU-based block-diagonal matrix tensor implementation.
///
/// Provides the BlockDiagMatrix class for storing and manipulating block-diagonal matrices
/// on NVIDIA GPUs. Supports batched GEMM and GEMV operations via cuBLAS and automatic GPU
/// memory management with caching.

#pragma once

#include "runtime.hpp"
#include "impl/tensor/gpu/matrix.hpp"
#include "impl/tensor/gpu/block-matrix-classes/block-matrix.hpp"
#include <cstring>
#include <map>

namespace lahva
{
    namespace gpu
    {
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

        /// @brief GPU-accelerated block-diagonal matrix with batched operations.
        ///
        /// Stores and manipulates block-diagonal matrices on NVIDIA GPUs with automatic
        /// GPU memory management and caching. The host representation maintains matrices
        /// in CPU memory, while an optional GPU cache stores padded/packed blocks for
        /// efficient cuBLAS batched operations. GPU cache is invalidated when host data
        /// is modified, ensuring correctness while maximizing GPU utilization.
        /// If the blocks are not uniform in size, cuSPARSE is used.
        ///
        /// @tparam T Numeric element type (double, float, complex types)
        /// @tparam Allocator Host (pinned) memory allocator (default: CudaHostAllocator<T>)
        /// @tparam GPUAllocator Device memory allocator (default: CudaDeviceAllocator<T>)
        template <class T, class Allocator = CudaHostAllocator<T>, class GPUAllocator = CudaDeviceAllocator<T>>
        class BlockDiagMatrix : virtual public Tensor<T, Allocator, GPUAllocator>, virtual public BlockMatrix_<T>
        {
            using alloc_ptr = CPUAllocator<T>;
            using gpualloc_ptr = GPUAllocator_<T>;
        protected:
            /// @brief Total number of rows (max row position + 1)
            size_t n_rows_ = 0;

            /// @brief Total number of columns (max column position + 1)
            size_t n_cols_ = 0;

            /// @brief Preferred sparse format for GPU operations (default: BSR)
            SparseFormat sparse_format_ = SparseFormat::BSR;

            /// @brief Blocks stored as map indexed by element-space position (row, col)
            /// Positions are element-space (i,j) where block's top-left corner is placed
            std::map<std::pair<size_t, size_t>, Matrix<T, Allocator>> blocks_;

            /// @brief Cached row offsets for blocks
            mutable std::vector<int> row_offsets_;

            /// @brief Cached column offsets for blocks
            mutable std::vector<int> col_offsets_;

            /// @brief Flag indicating if offsets cache is valid
            mutable bool row_offsets_valid_ = false;

            /// @brief Flag indicating if column offsets cache is valid
            mutable bool col_offsets_valid_ = false;

            /// @brief GPU vector for row offsets
            mutable Vector<int> d_row_offsets_;

            /// @brief GPU vector for block row sizes
            mutable Vector<int> d_block_row_sizes_;

            /// @brief Flag indicating if GPU offset vectors are valid
            mutable bool d_offsets_valid_ = false;

            /// @brief Cached GPU block-diagonal data (padded and packed format)
            /// @note nullptr if GPU cache is invalid; managed via free_gpu_cache()
            mutable GPUBlockDiagData<T> *gpu_data_ = nullptr;

            /// @brief Whether the GPU cache (gpu_data_) is valid and up-to-date with host
            mutable bool gpu_data_valid_ = false;

        public:
            /// @brief Lazily copy host blocks to GPU (internal optimization - caches result)
            /// @note Internal library method; prefer copy2device() for public API
            const GPUBlockDiagData<T>& ensure_on_gpu(const CudaRuntime &cudart) const {
                if (!gpu_data_valid_) {
                    copy2device(cudart);
                }
                return *gpu_data_;
            }

        private:
            /// @brief Check whether GPU data is current
            bool is_on_gpu() const { return gpu_data_valid_; }

            /// @brief Free GPU cache memory
            void free_gpu_cache() {
                if (gpu_data_ != nullptr) {
                    free_gpu_data(*gpu_data_);
                    delete gpu_data_;
                    gpu_data_ = nullptr;
                }
                gpu_data_valid_ = false;
            }

        public:
            /// @brief Default constructor - creates empty block-diagonal matrix
            BlockDiagMatrix() {};

            /// @brief Destructor - frees GPU memory
            virtual ~BlockDiagMatrix() {
                if (gpu_data_ != nullptr) {
                    free_gpu_data(*gpu_data_);
                    delete gpu_data_;
                }
            }

            /// @brief Copy constructor
            /// @param[in] other block-diagonal matrix to copy
            /// @note does not copy GPU cache; GPU cache is marked invalid
            BlockDiagMatrix(const BlockDiagMatrix &other) :
                n_rows_{other.n_rows_}, n_cols_{other.n_cols_},
                blocks_{other.blocks_}
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
                    blocks_ = other.blocks_;
                    row_offsets_valid_ = false;
                    col_offsets_valid_ = false;
                    d_offsets_valid_ = false;
                }
                return *this;
            }

            /// @brief Move constructor
            /// @param[in] other block-diagonal matrix to move from
            /// @note transfers GPU cache ownership to this matrix
            BlockDiagMatrix(BlockDiagMatrix &&other) noexcept :
                n_rows_{other.n_rows_}, n_cols_{other.n_cols_},
                blocks_{std::move(other.blocks_)},
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
                    Tensor<T, Allocator, GPUAllocator>::operator=(std::move(other));
                    BlockMatrix_<T>::operator=(std::move(other));
                    free_gpu_cache();
                    n_rows_ = other.n_rows_;
                    n_cols_ = other.n_cols_;
                    blocks_ = std::move(other.blocks_);
                    gpu_data_ = other.gpu_data_;
                    gpu_data_valid_ = other.gpu_data_valid_;
                    row_offsets_valid_ = false;
                    col_offsets_valid_ = false;
                    d_offsets_valid_ = false;
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
                size_t row_pos = 0, col_pos = 0;
                for (size_t i = 0; i < n_blocks; ++i) {
                    blocks_[{row_pos, col_pos}] = Matrix<T, Allocator>(block_shape);
                    row_pos += block_shape.first;
                    col_pos += block_shape.second;
                }
                update_dimensions();
            }

            /// @brief Create block-diagonal matrix with uniform blocks initialized to a value
            /// @param[in] n_blocks number of diagonal blocks to create
            /// @param[in] block_shape shape (rows, cols) for each block
            /// @param[in] val initialization value for all elements
            BlockDiagMatrix(size_t n_blocks, Shape block_shape, T val)
            {
                size_t row_pos = 0, col_pos = 0;
                for (size_t i = 0; i < n_blocks; ++i) {
                    blocks_[{row_pos, col_pos}] = Matrix<T, Allocator>(block_shape, val);
                    row_pos += block_shape.first;
                    col_pos += block_shape.second;
                }
                update_dimensions();
            }

            /// @brief Create block-diagonal matrix with varying block shapes
            /// @param[in] shapes vector of (rows, cols) pairs for each diagonal block
            /// @note blocks are uninitialized; use constructor with value parameter for initialization
            explicit BlockDiagMatrix(const std::vector<Shape> &shapes)
            {
                size_t row_pos = 0, col_pos = 0;
                for (const auto &s : shapes) {
                    blocks_[{row_pos, col_pos}] = Matrix<T, Allocator>(s);
                    row_pos += s.first;
                    col_pos += s.second;
                }
                update_dimensions();
            }

            /// @brief Create block-diagonal matrix with varying block shapes and uniform initialization
            /// @param[in] shapes vector of (rows, cols) pairs for each diagonal block
            /// @param[in] val initialization value for all block elements
            BlockDiagMatrix(const std::vector<Shape> &shapes, T val)
            {
                size_t row_pos = 0, col_pos = 0;
                for (const auto &s : shapes) {
                    blocks_[{row_pos, col_pos}] = Matrix<T, Allocator>(s, val);
                    row_pos += s.first;
                    col_pos += s.second;
                }
                update_dimensions();
            }

            /// @brief Create block-diagonal matrix from existing block matrices
            /// @param[in] blocks vector of Matrix objects to use as diagonal blocks
            /// @note takes a copy of the input vector; blocks are not modified
            BlockDiagMatrix(const std::vector<Matrix<T, Allocator>> &blocks)
            {
                size_t row_pos = 0, col_pos = 0;
                Shape first_shape;
                bool all_same_size = true;

                for (size_t i = 0; i < blocks.size(); ++i) {
                    const auto &block = blocks[i];
                    if (i == 0) {
                        first_shape = block.shape();
                    } else if (block.shape() != first_shape) {
                        all_same_size = false;
                    }

                    blocks_[{row_pos, col_pos}] = block;
                    Shape s = block.shape();
                    row_pos += s.first;
                    col_pos += s.second;
                }

                if (!all_same_size) {
                    sparse_format_ = SparseFormat::CSR;
                }

                update_dimensions();
            }

            /// @brief Create block-diagonal matrix from existing block matrices (move semantics)
            /// @param[in] blocks vector of Matrix objects to move as diagonal blocks
            /// @note blocks vector is moved into this matrix; contents of input vector are transferred
            BlockDiagMatrix(std::vector<Matrix<T, Allocator>> &&blocks)
            {
                size_t row_pos = 0, col_pos = 0;
                Shape first_shape;
                bool all_same_size = true;

                for (size_t i = 0; i < blocks.size(); ++i) {
                    auto &block = blocks[i];
                    if (i == 0) {
                        first_shape = block.shape();
                    } else if (block.shape() != first_shape) {
                        all_same_size = false;
                    }

                    blocks_[{row_pos, col_pos}] = std::move(block);
                    Shape s = blocks_[{row_pos, col_pos}].shape();
                    row_pos += s.first;
                    col_pos += s.second;
                }

                if (!all_same_size) {
                    sparse_format_ = SparseFormat::CSR;
                }

                update_dimensions();
            }

            /// @brief Add scalar to all block elements (in-place)
            /// @param[in] val value to add to each element
            /// @return reference to this matrix
            /// @note invalidates GPU cache
            BlockDiagMatrix &operator+=(T val)
            {
                free_gpu_cache();
                for (auto &[pos, block] : blocks_)
                    block += val;
                return *this;
            }

            /// @brief Get total shape of block-diagonal matrix
            /// @return Shape with total rows and columns across all blocks
            virtual Shape shape() const override { return Shape{n_rows_, n_cols_}; }

            /// @brief Get shapes of all blocks as vector of vectors
            /// @return Vector where each element [rows, cols] is the shape of a block in order
            std::vector<std::vector<size_t>> block_shapes() const
            {
                std::vector<std::vector<size_t>> result;
                for (const auto& [pos, block] : blocks_) {
                    Shape s = block.shape();
                    result.push_back({s.first, s.second});
                }
                return result;
            }

            /// @brief Get number of diagonal blocks
            /// @return Count of blocks in matrix
            virtual size_t num_blocks() const override { return blocks_.size(); }

            /// @brief Get const reference to a specific block matrix by linear index
            /// @param[in] idx block index
            /// @return const reference to the block matrix
            const Matrix<T, Allocator>& get_block(size_t idx) const {
                if (idx >= blocks_.size()) {
                    throw std::out_of_range("Block index out of range");
                }
                auto it = blocks_.begin();
                std::advance(it, idx);
                return it->second;
            }

            /// @brief Get cumulative row offsets for block access
            /// @return const reference to vector of row offsets (element i+1 is where block i ends)
            virtual const std::vector<int>& get_row_offsets() const override {
                if (!row_offsets_valid_) {
                    compute_offsets();
                }
                return row_offsets_;
            }

            /// @brief Get cumulative column offsets for block access
            /// @return const reference to vector of column offsets (element i+1 is where block i ends)
            virtual const std::vector<int>& get_col_offsets() const override {
                if (!col_offsets_valid_) {
                    compute_offsets();
                }
                return col_offsets_;
            }

            /// @brief Get shape of a specific block
            /// @param[in] idx block index
            /// @return Shape object with block dimensions
            virtual Shape get_block_shape(size_t idx) const override {
                if (idx >= blocks_.size()) {
                    throw std::out_of_range("Block index out of range");
                }
                auto it = blocks_.begin();
                std::advance(it, idx);
                return it->second.shape();
            }

            /// @brief Get raw data pointer for a specific block
            /// @param[in] idx block index
            /// @return const pointer to block data
            const T* get_block_data(size_t idx) const {
                if (idx >= blocks_.size()) {
                    throw std::out_of_range("Block index out of range");
                }
                auto it = blocks_.begin();
                std::advance(it, idx);
                return it->second.data();
            }

            /// @brief Get block data by element-space row and column position
            /// @param[in] block_row Element-space row position of block
            /// @param[in] block_col Element-space column position of block
            /// @return Pointer to block data, or nullptr if block doesn't exist at that position
            virtual const T* get_block_data_at(size_t block_row, size_t block_col) const override {
                auto it = blocks_.find({block_row, block_col});
                if (it != blocks_.end()) {
                    return it->second.data();
                }
                return nullptr;
            }

            /// @brief Get row dimensions of all blocks
            /// @return vector of row dimensions (m_i) for each block in order
            std::vector<size_t> get_block_rows() const {
                std::vector<size_t> result;
                for (const auto &[pos, block] : blocks_) {
                    result.push_back(block.shape().first);
                }
                return result;
            }

            /// @brief Get column dimensions of all blocks
            /// @return vector of column dimensions (k_i) for each block in order
            std::vector<size_t> get_block_cols() const {
                std::vector<size_t> result;
                for (const auto &[pos, block] : blocks_) {
                    result.push_back(block.shape().second);
                }
                return result;
            }

            /// @brief Get GPU pointer to row offsets
            /// @param[in] cudart CUDA runtime for device allocation
            /// @return Device pointer to row offsets array
            const int* get_d_row_offsets(const CudaRuntime &cudart) const {
                if (!d_offsets_valid_) {
                    init_gpu_offsets(cudart);
                }
                return d_row_offsets_.gpu_data();
            }

            /// @brief Get GPU pointer to block row sizes
            /// @param[in] cudart CUDA runtime for device allocation
            /// @return Device pointer to block row sizes array
            const int* get_d_block_row_sizes(const CudaRuntime &cudart) const {
                if (!d_offsets_valid_) {
                    init_gpu_offsets(cudart);
                }
                return d_block_row_sizes_.gpu_data();
            }

            /// @brief Extract main diagonals from all blocks as concatenated vector
            /// @return Vector containing diagonals of each block concatenated sequentially
            /// @note vector size equals sum of min(block_rows, block_cols) over all blocks
            cpu::Vector<T, Allocator> get_diagonal() const
            {
                size_t total = 0;
                for (const auto &[pos, block] : blocks_)
                {
                    Shape s = block.shape();
                    total += std::min(s.first, s.second);
                }
                cpu::Vector<T, Allocator> diag(total);
                size_t offset = 0;
                for (const auto &[pos, block] : blocks_)
                {
                    Shape s = block.shape();
                    size_t min_dim = std::min(s.first, s.second);
                    for (size_t i = 0; i < min_dim; i++)
                        diag[offset + i] = block(i, i);
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
                for (std::pair<const std::pair<size_t, size_t>, Matrix<T, Allocator>> &entry : blocks_)
                {
                    Matrix<T, Allocator> &block = entry.second;
                    Shape s = block.shape();
                    size_t min_dim = std::min(s.first, s.second);
#pragma omp for
                    for (size_t i = 0; i < min_dim; i++)
                        block(i, i) = diag[offset + i];
                    offset += min_dim;
                }
            }

            /// @brief Symmetrize each block in-place: block = (block + block^T) / 2
            /// @note applies symmetrization to each diagonal block independently; invalidates GPU cache
            void symmetrize()
            {
                free_gpu_cache();
                for (auto &[pos, block] : blocks_)
                    block.symmetrize();
            }

            /// @brief Element access operator (mutable)
            /// @param[in] i row index
            /// @param[in] j column index
            /// @return reference to element at (i,j); returns zero for off-diagonal positions
            /// @note GPU cache must be manually invalidated after modifications via free_gpu_cache()
            virtual T &operator()(size_t i, size_t j) override {
                for (auto &[pos, block] : blocks_) {
                    size_t row_start = pos.first;
                    size_t col_start = pos.second;
                    Shape s = block.shape();
                    size_t row_end = row_start + s.first;
                    size_t col_end = col_start + s.second;

                    if (i >= row_start && i < row_end && j >= col_start && j < col_end) {
                        return block(i - row_start, j - col_start);
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
                for (const auto &[pos, block] : blocks_) {
                    size_t row_start = pos.first;
                    size_t col_start = pos.second;
                    Shape s = block.shape();
                    size_t row_end = row_start + s.first;
                    size_t col_end = col_start + s.second;

                    if (i >= row_start && i < row_end && j >= col_start && j < col_end) {
                        return block(i - row_start, j - col_start);
                    }
                }
                static const T zero{};
                return zero;
            }

            /// @brief Get element-space row position for a block by linear index
            size_t get_block_row(size_t idx) const override {
                if (idx >= blocks_.size()) {
                    throw std::out_of_range("Block index out of range");
                }
                auto it = blocks_.begin();
                std::advance(it, idx);
                return it->first.first;
            }

            /// @brief Get element-space column position for a block by linear index
            size_t get_block_col(size_t idx) const override {
                if (idx >= blocks_.size()) {
                    throw std::out_of_range("Block index out of range");
                }
                auto it = blocks_.begin();
                std::advance(it, idx);
                return it->first.second;
            }

            /// @brief Get preferred sparse format for GPU operations
            SparseFormat get_sparse_format() const override {
                return sparse_format_;
            }

            /// @brief Set preferred sparse format for GPU operations
            /// @param[in] format SparseFormat::CSR or SparseFormat::BSR
            /// @note Invalidates GPU cache when changed
            void set_sparse_format(SparseFormat format) {
                if (format != sparse_format_) {
                    sparse_format_ = format;
                    free_gpu_cache();
                }
            }

            /// @brief Add a new block at specified index
            /// @param[in] block matrix block to insert
            /// @param[in] index position to insert (default: append at end)
            /// @note blocks at index >= the insertion index are shifted; invalidates GPU cache
            /// @throws std::out_of_range if index is beyond current block count
            void add_block(const Matrix<T, Allocator> &block, size_t index = std::numeric_limits<size_t>::max()) {
                free_gpu_cache();

                // Default to append at end
                if (index == std::numeric_limits<size_t>::max()) {
                    index = blocks_.size();
                }

                if (index > blocks_.size()) {
                    throw std::out_of_range("Block insertion index out of range");
                }

                Shape new_shape = block.shape();
                size_t insert_row_pos, insert_col_pos;

                // Check if block size is compatible with current sparse format
                if (!blocks_.empty() && get_sparse_format() == SparseFormat::BSR)
                {
                    auto first_block_shape = blocks_.begin()->second.shape();
                    if (new_shape.first != first_block_shape.first || new_shape.second != first_block_shape.second)
                    {
                        set_sparse_format(SparseFormat::CSR);
                    }
                }

                // Find insertion position by iterating to index-th block
                if (index == 0) {
                    insert_row_pos = 0;
                    insert_col_pos = 0;
                } else {
                    auto it = blocks_.begin();
                    std::advance(it, index - 1);
                    const auto &[pos, prev_block] = *it;
                    Shape prev_shape = prev_block.shape();
                    insert_row_pos = pos.first + prev_shape.first;
                    insert_col_pos = pos.second + prev_shape.second;
                }

                // Shift all blocks at index >= insertion index
                std::vector<std::pair<std::pair<size_t, size_t>, Matrix<T, Allocator>>> to_shift;
                auto it = blocks_.begin();
                size_t current_idx = 0;
                while (it != blocks_.end()) {
                    if (current_idx >= index) {
                        to_shift.push_back(*it);
                        it = blocks_.erase(it);
                    } else {
                        ++it;
                        ++current_idx;
                    }
                }

                // Re-insert shifted blocks with updated positions
                size_t shift_row = new_shape.first;
                size_t shift_col = new_shape.second;
                for (auto &[old_pos, block_data] : to_shift) {
                    size_t new_row = old_pos.first + shift_row;
                    size_t new_col = old_pos.second + shift_col;
                    blocks_[{new_row, new_col}] = std::move(block_data);
                }

                // Insert the new block
                blocks_[{insert_row_pos, insert_col_pos}] = block;
                update_dimensions();
                row_offsets_valid_ = false;
                col_offsets_valid_ = false;
            }

            /// @brief Print all blocks to standard output
            /// @note Outputs header with total dimensions, then each block with position and data
            void print() const {
                std::cout << "BlockDiagMatrix (" << n_rows_ << " x " << n_cols_ << ")" << std::endl;
                for (const auto &[pos, block] : blocks_) {
                    std::cout << "Block at (" << pos.first << ", " << pos.second << ")" << std::endl;
                    block.print();
                }
            }

            /// @brief Print all blocks to a file
            /// @param[in] file filename to write block data to
            /// @note Appends block data to file; each block's data written sequentially
            void print(const char* file) const
            {
                for (const auto &[pos, block] : blocks_) {
                    block.print(file);
                }
            }

            /// @brief Get estimated GPU memory requirement for packed blocks
            size_t max_block_size() const {
                if (blocks_.empty()) return 0;
                size_t max_m = 0, max_k = 0;
                for (const auto &[pos, block] : blocks_) {
                    Shape s = block.shape();
                    max_m = std::max(max_m, static_cast<size_t>(s.first));
                    max_k = std::max(max_k, static_cast<size_t>(s.second));
                }
                return max_m * max_k * num_blocks();
            }

            /// @brief Copy host blocks to GPU with automatic packing into padded format
            /// @param[in] cudart CUDA runtime for device selection
            /// @note Pads blocks to max dimensions for efficient batched operations; invalidates stale GPU cache
            void copy2device(const CudaRuntime &cudart) const override {
                (void)cudart;
                int num_blocks = blocks_.size();

                if (num_blocks == 0) {
                    gpu_data_valid_ = true;
                    return;
                }

                size_t max_m = 0, max_k = 0;
                for (const auto &[pos, block] : blocks_) {
                    Shape s = block.shape();
                    max_m = std::max(max_m, static_cast<size_t>(s.first));
                    max_k = std::max(max_k, static_cast<size_t>(s.second));
                }

                long long padded_stride = (long long)max_m * max_k;

                T *h_packed;
                get_cuda_error(cudaHostAlloc(&h_packed, padded_stride * num_blocks * sizeof(T), cudaHostAllocDefault));

                std::memset(h_packed, 0, padded_stride * num_blocks * sizeof(T));

                int i = 0;
                for (const auto &[pos, block] : blocks_) {
                    size_t block_m = block.shape().first;
                    size_t block_k = block.shape().second;

                    for (size_t j = 0; j < block_k; ++j) {
                        std::memcpy(
                            h_packed + i * padded_stride + j * max_m,
                            block.data() + j * block_m,
                            block_m * sizeof(T)
                        );
                    }
                    i++;
                }

                T *d_data;
                get_cuda_error(cudaMalloc(&d_data, padded_stride * num_blocks * sizeof(T)));
                get_cuda_error(cudaMemcpy(d_data, h_packed, padded_stride * num_blocks * sizeof(T), cudaMemcpyHostToDevice));

                gpu_data_ = new GPUBlockDiagData<T>{d_data, (size_t)padded_stride, max_m, max_k, num_blocks, h_packed};
                gpu_data_valid_ = true;
            }

            /// @brief Copy GPU blocks back to host with automatic unpacking from padded format
            /// @param[in] cudart CUDA runtime for device operations
            /// @note Unpacks padded blocks back to original block layouts; no-op if GPU cache invalid
            void copy2host(const CudaRuntime &cudart) override {
                (void)cudart;
                if (!gpu_data_valid_ || gpu_data_ == nullptr) {
                    return;
                }

                const GPUBlockDiagData<T>& gpu_data = *gpu_data_;

                get_cuda_error(cudaMemcpy(gpu_data.h_packed, gpu_data.d_data,
                                         gpu_data.padded_stride * gpu_data.num_blocks * sizeof(T),
                                         cudaMemcpyDeviceToHost));

                int block_idx = 0;
                for (auto &[pos, block] : blocks_) {
                    size_t block_m = block.shape().first;
                    size_t block_k = block.shape().second;

                    for (size_t j = 0; j < block_k; ++j) {
                        std::memcpy(
                            block.data() + j * block_m,
                            gpu_data.h_packed + block_idx * gpu_data.padded_stride + j * gpu_data.max_m,
                            block_m * sizeof(T)
                        );
                    }
                    block_idx++;
                }
            }

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

        private:
            /// @brief Update total matrix dimensions based on stored blocks
            void update_dimensions() {
                n_rows_ = 0;
                n_cols_ = 0;

                for (const auto &[pos, block] : blocks_) {
                    size_t row_start = pos.first;
                    size_t col_start = pos.second;
                    Shape s = block.shape();

                    size_t row_end = row_start + s.first;
                    size_t col_end = col_start + s.second;

                    n_rows_ = std::max(n_rows_, row_end);
                    n_cols_ = std::max(n_cols_, col_end);
                }
            }

            /// @brief Compute and cache block offsets
            void compute_offsets() const {
                row_offsets_.clear();
                col_offsets_.clear();

                int row_offset = 0;
                int col_offset = 0;

                for (const auto &[pos, block] : blocks_) {
                    Shape s = block.shape();
                    row_offsets_.push_back(row_offset);
                    col_offsets_.push_back(col_offset);
                    row_offset += static_cast<int>(s.first);
                    col_offset += static_cast<int>(s.second);
                }

                row_offsets_.push_back(row_offset);
                col_offsets_.push_back(col_offset);

                row_offsets_valid_ = true;
                col_offsets_valid_ = true;
            }

            /// @brief Initialize GPU offset vectors (lazy initialization)
            /// @param[in] cudart CUDA runtime for device allocation
            void init_gpu_offsets(const CudaRuntime &cudart) const {
                // Ensure host offsets are computed
                if (!row_offsets_valid_) {
                    compute_offsets();
                }

                // Create GPU vectors and copy host data
                d_row_offsets_ = Vector<int>(row_offsets_.size());
                std::copy(row_offsets_.begin(), row_offsets_.end(), d_row_offsets_.data());
                d_row_offsets_.copy2device(cudart);

                // Get block row sizes (differences between consecutive offsets)
                std::vector<int> block_row_sizes;
                for (size_t i = 0; i < row_offsets_.size() - 1; ++i) {
                    block_row_sizes.push_back(row_offsets_[i + 1] - row_offsets_[i]);
                }

                d_block_row_sizes_ = Vector<int>(block_row_sizes.size());
                std::copy(block_row_sizes.begin(), block_row_sizes.end(), d_block_row_sizes_.data());
                d_block_row_sizes_.copy2device(cudart);

                d_offsets_valid_ = true;
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

    } // namespace gpu
} // namespace lahva
