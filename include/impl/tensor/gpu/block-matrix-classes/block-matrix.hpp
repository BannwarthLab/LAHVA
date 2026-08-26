/// @file block-matrix.hpp
/// @brief GPU block matrix base and concrete classes.
///
/// Provides the BlockMatrix_ abstract base class and BlockMatrix concrete class
/// for GPU-accelerated block matrix operations at arbitrary positions.

#pragma once

#include "impl/tensor/gpu/matrix.hpp"
#include <map>

namespace lahva
{
    namespace gpu
    {
        /// @brief Enumeration for sparse matrix format selection
        enum class SparseFormat
        {
            CSR,  ///< Compressed Sparse Row format
            BSR   ///< Block Sparse Row format
        };

        /// @brief Forward declaration of SparseMatrix
        template<typename T>
        class SparseMatrix;

        /// @brief Abstract base class for GPU block-diagonal matrices
        /// @tparam T data type for matrix elements
        template<typename T>
        class BlockMatrix_ : public virtual Tensor_<T>
        {
        public:
            BlockMatrix_() = default;
            BlockMatrix_(const BlockMatrix_ &) = default;
            BlockMatrix_ &operator=(const BlockMatrix_ &) = default;
            BlockMatrix_(BlockMatrix_ &&) noexcept = default;
            BlockMatrix_ &operator=(BlockMatrix_ &&) noexcept { return *this; }
            virtual ~BlockMatrix_() = default;

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

            /// @brief Get grid row index for a block by linear index
            /// @param[in] idx Block linear index
            /// @return Grid row index
            virtual size_t get_block_row(size_t idx) const = 0;

            /// @brief Get grid column index for a block by linear index
            /// @param[in] idx Block linear index
            /// @return Grid column index
            virtual size_t get_block_col(size_t idx) const = 0;

            /// @brief Get preferred sparse format for GPU operations (CSR or BSR)
            /// @return SparseFormat enum value
            virtual SparseFormat get_sparse_format() const {
                return SparseFormat::BSR;
            }

            /// @brief Get a pointer to block data by element-space row and column position
            /// @param[in] block_row Element-space row position of block
            /// @param[in] block_col Element-space column position of block
            /// @return Pointer to block data, or nullptr if block doesn't exist at that position
            virtual const T* get_block_data_at(size_t block_row, size_t block_col) const = 0;

        };

        /// @brief GPU-accelerated block matrix with blocks at arbitrary positions
        ///
        /// A generalization of BlockDiagMatrix that allows blocks at any (i,j) position,
        /// not just along the diagonal. Useful for general sparse or structured matrices.
        ///
        /// @tparam T Numeric element type (double, float, complex types)
        /// @tparam Allocator Host (pinned) memory allocator (default: CudaHostAllocator<T>)
        /// @tparam GPUAllocator Device memory allocator (default: CudaDeviceAllocator<T>)
        template <class T, class Allocator = CudaHostAllocator<T>, class GPUAllocator = CudaDeviceAllocator<T>>
        class BlockMatrix : virtual public Tensor<T, Allocator, GPUAllocator>, virtual public BlockMatrix_<T>
        {
            using alloc_ptr = CPUAllocator<T>;
            using gpualloc_ptr = GPUAllocator_<T>;
        public:
            /// @brief Default constructor
            BlockMatrix()
                : Tensor<T, Allocator, GPUAllocator>(), BlockMatrix_<T>(),
                  n_rows_(0), n_cols_(0), uniform_block_size_(true), cached_block_shape_(0, 0)
            {
            }

            /// @brief Constructor from shape - creates zero-initialized matrix
            /// @param[in] shape total matrix dimensions (rows, columns)
            /// @param[in] alloc host memory allocator (default: Allocator())
            /// @param[in] gpualloc device memory allocator (default: GPUAllocator())
            explicit BlockMatrix(const Shape &shape, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator())
                : Tensor<T, Allocator, GPUAllocator>(shape.first * shape.second, alloc, gpualloc),
                  BlockMatrix_<T>(),
                  n_rows_(shape.first), n_cols_(shape.second), uniform_block_size_(true), cached_block_shape_(0, 0)
            {
            }

            /// @brief Copy constructor
            BlockMatrix(const BlockMatrix &other) = default;

            /// @brief Move constructor
            BlockMatrix(BlockMatrix &&other) noexcept = default;

            /// @brief Copy assignment operator
            BlockMatrix &operator=(const BlockMatrix &other) = default;

            /// @brief Move assignment operator
            BlockMatrix &operator=(BlockMatrix &&other) noexcept {
                Tensor<T, Allocator, GPUAllocator>::operator=(std::move(other));
                BlockMatrix_<T>::operator=(std::move(other));
                n_rows_ = other.n_rows_;
                n_cols_ = other.n_cols_;
                blocks_ = std::move(other.blocks_);
                block_order_ = std::move(other.block_order_);
                row_offsets_ = std::move(other.row_offsets_);
                col_offsets_ = std::move(other.col_offsets_);
                row_offsets_valid_ = other.row_offsets_valid_;
                col_offsets_valid_ = other.col_offsets_valid_;
                cached_num_block_rows_ = other.cached_num_block_rows_;
                cached_num_block_cols_ = other.cached_num_block_cols_;
                block_dims_valid_ = other.block_dims_valid_;
                uniform_block_size_ = other.uniform_block_size_;
                cached_block_shape_ = other.cached_block_shape_;
                d_row_offsets_ = std::move(other.d_row_offsets_);
                d_block_row_sizes_ = std::move(other.d_block_row_sizes_);
                d_offsets_valid_ = other.d_offsets_valid_;
                other.n_rows_ = 0;
                other.n_cols_ = 0;
                other.row_offsets_valid_ = false;
                other.col_offsets_valid_ = false;
                other.cached_num_block_rows_ = 0;
                other.cached_num_block_cols_ = 0;
                other.block_dims_valid_ = false;
                other.uniform_block_size_ = true;
                other.cached_block_shape_ = Shape(0, 0);
                other.d_offsets_valid_ = false;
                return *this;
            }

            /// @brief Get total shape (rows, columns) of the block matrix
            Shape shape() const override { return Shape(n_rows_, n_cols_); }

            /// @brief Get total number of blocks stored
            size_t num_blocks() const override { return blocks_.size(); }

            /// @brief Get shape of a specific block by linear index
            Shape get_block_shape(size_t idx) const override {
                if (idx >= block_order_.size()) {
                    throw std::out_of_range("Block index " + std::to_string(idx) + " out of range");
                }
                return block_order_[idx]->second.shape();
            }

            /// @brief Get row offsets for all blocks
            const std::vector<int>& get_row_offsets() const override {
                if (!row_offsets_valid_) {
                    compute_offsets();
                }
                return row_offsets_;
            }

            /// @brief Get column offsets for all blocks
            const std::vector<int>& get_col_offsets() const override {
                if (!col_offsets_valid_) {
                    compute_offsets();
                }
                return col_offsets_;
            }

            /// @brief Element access operator (mutable)
            /// @param[in] i row index
            /// @param[in] j column index
            /// @return reference to element at (i,j); returns zero for positions outside all blocks
            T &operator()(size_t i, size_t j) override {
                // Find block containing element (i, j)
                // (i, j) in blocks_ map is the element-space position of block top-left corner
                for (auto& [pos, block] : blocks_) {
                    size_t row_start = pos.first;
                    size_t col_start = pos.second;
                    Shape s = block.shape();
                    size_t row_end = row_start + s.first;
                    size_t col_end = col_start + s.second;

                    if (i >= row_start && i < row_end && j >= col_start && j < col_end) {
                        return block(i - row_start, j - col_start);
                    }
                }
                static T zero_value = (T)0;
                return zero_value;
            }

            /// @brief Element access operator (const)
            /// @param[in] i row index
            /// @param[in] j column index
            /// @return const reference to element at (i,j); returns zero for positions outside all blocks
            const T &operator()(size_t i, size_t j) const override {
                // Find block containing element (i, j)
                // (i, j) in blocks_ map is the element-space position of block top-left corner
                for (const auto& [pos, block] : blocks_) {
                    size_t row_start = pos.first;
                    size_t col_start = pos.second;
                    Shape s = block.shape();
                    size_t row_end = row_start + s.first;
                    size_t col_end = col_start + s.second;

                    if (i >= row_start && i < row_end && j >= col_start && j < col_end) {
                        return block(i - row_start, j - col_start);
                    }
                }
                static const T zero_value = (T)0;
                return zero_value;
            }

            /// @brief Get element row position for a block by linear index
            size_t get_block_row(size_t idx) const override {
                if (idx >= block_order_.size()) {
                    throw std::out_of_range("Block index out of range");
                }
                return block_order_[idx]->first.first;
            }

            /// @brief Get element column position for a block by linear index
            size_t get_block_col(size_t idx) const override {
                if (idx >= block_order_.size()) {
                    throw std::out_of_range("Block index out of range");
                }
                return block_order_[idx]->first.second;
            }

            /// @brief Get block data by element-space row and column position
            /// @param[in] block_row Element-space row position of block
            /// @param[in] block_col Element-space column position of block
            /// @return Pointer to block data, or nullptr if block doesn't exist at that position
            const T* get_block_data_at(size_t block_row, size_t block_col) const override {
                auto it = blocks_.find({block_row, block_col});
                if (it != blocks_.end()) {
                    return it->second.data();
                }
                return nullptr;
            }

            /// @brief Get preferred sparse format for GPU operations
            /// @return SparseFormat::BSR if all blocks have uniform size, CSR otherwise
            SparseFormat get_sparse_format() const override {
                return uniform_block_size_ ? SparseFormat::BSR : SparseFormat::CSR;
            }

            /// @brief Get number of block rows in the block grid
            /// @return maximum row position + 1
            size_t num_block_rows() const {
                if (!block_dims_valid_) {
                    compute_block_dims();
                }
                return cached_num_block_rows_;
            }

            /// @brief Get number of block columns in the block grid
            /// @return maximum column position + 1
            size_t num_block_cols() const {
                if (!block_dims_valid_) {
                    compute_block_dims();
                }
                return cached_num_block_cols_;
            }

            /// @brief Check if a block exists at position (i,j)
            /// @param[in] i element row where block starts
            /// @param[in] j element column where block starts
            /// @return true if block exists at (i,j), false otherwise
            bool has_block(size_t i, size_t j) const {
                return blocks_.find({i, j}) != blocks_.end();
            }

            /// @brief Add or update a block at position (i,j)
            /// @param[in] i element row where block starts
            /// @param[in] j element column where block starts
            /// @param[in] block the block matrix to store
            /// @throws std::runtime_error if block overlaps with existing blocks
            void set_block(size_t i, size_t j, const Matrix<T, Allocator>& block) {
                Shape new_shape = block.shape();
                size_t new_row_end = i + new_shape.first;
                size_t new_col_end = j + new_shape.second;

                // Check for overlaps with existing blocks
                for (const auto& [pos, existing_block] : blocks_) {
                    // Skip if replacing same block
                    if (pos == std::make_pair(i, j)) {
                        continue;
                    }

                    size_t exist_row_start = pos.first;
                    size_t exist_col_start = pos.second;
                    Shape exist_shape = existing_block.shape();
                    size_t exist_row_end = exist_row_start + exist_shape.first;
                    size_t exist_col_end = exist_col_start + exist_shape.second;

                    // Check for row and column overlap
                    bool rows_overlap = (i < exist_row_end) && (exist_row_start < new_row_end);
                    bool cols_overlap = (j < exist_col_end) && (exist_col_start < new_col_end);

                    if (rows_overlap && cols_overlap) {
                        throw std::runtime_error("Block at (" + std::to_string(i) + "," + std::to_string(j) +
                                               ") size " + std::to_string(new_shape.first) + "x" +
                                               std::to_string(new_shape.second) +
                                               " overlaps with existing block at (" +
                                               std::to_string(exist_row_start) + "," + std::to_string(exist_col_start) +
                                               ") size " + std::to_string(exist_shape.first) + "x" +
                                               std::to_string(exist_shape.second));
                    }
                }

                bool is_new_block = blocks_.find({i, j}) == blocks_.end();
                blocks_[{i, j}] = block;

                if (is_new_block) {
                    block_order_.push_back(blocks_.find({i, j}));
                }

                update_dimensions();
                check_uniform_block_size(new_shape);

                // Invalidate offset and dimension caches since block configuration changed
                row_offsets_valid_ = false;
                col_offsets_valid_ = false;
                block_dims_valid_ = false;
                d_offsets_valid_ = false;
            }

            /// @brief Get block at position (i,j)
            /// @param[in] i element row where block starts
            /// @param[in] j element column where block starts
            /// @return reference to block matrix at (i,j)
            /// @throws std::out_of_range if block doesn't exist
            Matrix<T, Allocator>& get_block(size_t i, size_t j) {
                auto key = std::make_pair(i, j);
                auto it = blocks_.find(key);
                if (it == blocks_.end()) {
                    throw std::out_of_range("Block at (" + std::to_string(i) + "," + std::to_string(j) + ") does not exist");
                }
                return it->second;
            }

            /// @brief Get block at position (i,j) (const)
            /// @param[in] i element row where block starts
            /// @param[in] j element column where block starts
            /// @return const reference to block matrix at (i,j)
            /// @throws std::out_of_range if block doesn't exist
            const Matrix<T, Allocator>& get_block(size_t i, size_t j) const {
                auto key = std::make_pair(i, j);
                auto it = blocks_.find(key);
                if (it == blocks_.end()) {
                    throw std::out_of_range("Block at (" + std::to_string(i) + "," + std::to_string(j) + ") does not exist");
                }
                return it->second;
            }

            /// @brief Get GPU pointer to row offsets (lazy initialized)
            /// @param[in] cudart CUDA runtime for device allocation
            /// @return Device pointer to row offsets array
            const int* get_d_row_offsets(const CudaRuntime &cudart) const {
                if (!d_offsets_valid_) {
                    init_gpu_offsets(cudart);
                }
                return d_row_offsets_.gpu_data();
            }

            /// @brief Get GPU pointer to block row sizes (lazy initialized)
            /// @param[in] cudart CUDA runtime for device allocation
            /// @return Device pointer to block row sizes array
            const int* get_d_block_row_sizes(const CudaRuntime &cudart) const {
                if (!d_offsets_valid_) {
                    init_gpu_offsets(cudart);
                }
                return d_block_row_sizes_.gpu_data();
            }

            /// @brief Print the block matrix structure to standard output
            /// @note Outputs total dimensions, then each block with position and data
            void print() const {
                std::cout << "BlockMatrix (" << n_rows_ << " x " << n_cols_ << ")" << std::endl;
                for (const auto& [pos, block] : blocks_) {
                    std::cout << "Block at (" << pos.first << ", " << pos.second << ") size "
                              << block.shape().first << " x " << block.shape().second << ":" << std::endl;
                    for (size_t i = 0; i < block.shape().first; i++) {
                        for (size_t j = 0; j < block.shape().second; j++) {
                            std::cout << std::setw(5) << (int)block(i, j) << " ";
                        }
                        std::cout << std::endl;
                    }
                }
            }

        private:
            /// @brief Total number of rows (sum of all block rows)
            size_t n_rows_;

            /// @brief Total number of columns (sum of all block columns)
            size_t n_cols_;

            /// @brief Blocks stored as a map indexed by (i,j) block coordinates
            std::map<std::pair<size_t, size_t>, Matrix<T, Allocator>> blocks_;

            /// @brief Iterators to blocks in insertion order
            std::vector<typename std::map<std::pair<size_t, size_t>, Matrix<T, Allocator>>::iterator> block_order_;

            /// @brief Cached row offsets for blocks
            mutable std::vector<int> row_offsets_;

            /// @brief Cached column offsets for blocks
            mutable std::vector<int> col_offsets_;

            /// @brief Flag indicating if offsets cache is valid
            mutable bool row_offsets_valid_ = false;

            /// @brief Flag indicating if column offsets cache is valid
            mutable bool col_offsets_valid_ = false;

            /// @brief Flag indicating if all blocks have uniform size
            bool uniform_block_size_;

            /// @brief Cached shape for uniform block size check
            Shape cached_block_shape_;

            /// @brief Cached number of block rows
            mutable size_t cached_num_block_rows_ = 0;

            /// @brief Cached number of block columns
            mutable size_t cached_num_block_cols_ = 0;

            /// @brief Flag indicating if block dimensions cache is valid
            mutable bool block_dims_valid_ = false;

            /// @brief GPU vector for row offsets
            mutable Vector<int> d_row_offsets_;

            /// @brief GPU vector for block row sizes
            mutable Vector<int> d_block_row_sizes_;

            /// @brief Flag indicating if GPU offset vectors are valid
            mutable bool d_offsets_valid_ = false;

            /// @brief Compute and cache block grid dimensions
            void compute_block_dims() const {
                cached_num_block_rows_ = 0;
                cached_num_block_cols_ = 0;

                for (const auto& block_it : block_order_) {
                    cached_num_block_rows_ = std::max(cached_num_block_rows_, block_it->first.first + 1);
                    cached_num_block_cols_ = std::max(cached_num_block_cols_, block_it->first.second + 1);
                }

                block_dims_valid_ = true;
            }

            /// @brief Compute and cache block offsets
            void compute_offsets() const {
                row_offsets_.clear();
                col_offsets_.clear();

                int row_offset = 0;
                int col_offset = 0;

                for (const auto& block_it : block_order_) {
                    Shape s = block_it->second.shape();
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

            /// @brief Initialize GPU offset vectors
            void init_gpu_offsets(const CudaRuntime &cudart) const {

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

            /// @brief Update total matrix dimensions based on stored blocks
            void update_dimensions() {
                n_rows_ = 0;
                n_cols_ = 0;

                // (i, j) are now element-space positions, so compute max bounds
                for (const auto& [pos, block] : blocks_) {
                    size_t row_start = pos.first;
                    size_t col_start = pos.second;
                    Shape s = block.shape();

                    size_t row_end = row_start + s.first;
                    size_t col_end = col_start + s.second;

                    n_rows_ = std::max(n_rows_, row_end);
                    n_cols_ = std::max(n_cols_, col_end);
                }
            }

            /// @brief Check if the new block maintains uniform block size
            void check_uniform_block_size(const Shape& new_shape) {
                if (blocks_.empty()) {
                    uniform_block_size_ = true;
                    cached_block_shape_ = new_shape;
                    return;
                }

                if (!uniform_block_size_) {
                    return;  // Already non-uniform, stay non-uniform
                }

                if (new_shape != cached_block_shape_) {
                    uniform_block_size_ = false;
                }
            }

        };

    } // namespace gpu
} // namespace lahva
