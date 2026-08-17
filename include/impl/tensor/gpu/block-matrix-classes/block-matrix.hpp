/// @file block-matrix.hpp
/// @brief GPU block matrix base and concrete classes.
///
/// Provides the BlockMatrix_ abstract base class and BlockMatrix concrete class
/// for GPU-accelerated block matrix operations at arbitrary positions.

#pragma once

#include "impl/tensor/gpu/gputensor.hpp"
#include "impl/tensor/gpu/matrix.hpp"
#include <vector>
#include <map>
#include <cstddef>
#include <iostream>
#include <iomanip>

namespace lahva
{
    namespace gpu
    {
        /// @brief Enumeration for sparse matrix format selection
        enum class SparseFormat
        {
            CSR,        ///< Compressed Sparse Row format
            BSR         ///< Block Sparse Row format
        };

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

            /// @brief Get grid row index for a block by linear index
            /// @param[in] idx Block linear index
            /// @return Grid row index
            virtual size_t get_block_row(size_t idx) const = 0;

            /// @brief Get grid column index for a block by linear index
            /// @param[in] idx Block linear index
            /// @return Grid column index
            virtual size_t get_block_col(size_t idx) const = 0;

            /// @brief Get preferred sparse format for GPU operations (CSR or BSR)
            /// @return SparseFormat enum value (default: CSR)
            virtual SparseFormat get_sparse_format() const {
                return SparseFormat::CSR;
            }
        };

        //! @brief GPU-accelerated block matrix with blocks at arbitrary positions
        //!
        //! A generalization of BlockDiagMatrix that allows blocks at any (i,j) position,
        //! not just along the diagonal. Useful for general sparse or structured matrices.
        //!
        //! @tparam T Numeric element type (double, float, complex types)
        //! @tparam Allocator Host (pinned) memory allocator (default: CudaHostAllocator<T>)
        //! @tparam GPUAllocator Device memory allocator (default: CudaDeviceAllocator<T>)
        template <class T, class Allocator = CudaHostAllocator<T>, class GPUAllocator = CudaDeviceAllocator<T>>
        class BlockMatrix : virtual public Tensor<T, Allocator, GPUAllocator>, virtual public BlockMatrix_<T>
        {
            using alloc_ptr = CPUAllocator<T>;
            using gpualloc_ptr = GPUAllocator_<T>;
        public:
            //! @brief Default constructor
            BlockMatrix()
                : n_rows_(0), n_cols_(0)
            {
            }

            //! @brief Move assignment operator
            BlockMatrix &operator=(BlockMatrix &&other) noexcept {
                Tensor<T, Allocator, GPUAllocator>::operator=(std::move(other));
                BlockMatrix_<T>::operator=(std::move(other));
                n_rows_ = other.n_rows_;
                n_cols_ = other.n_cols_;
                blocks_ = std::move(other.blocks_);
                row_offsets_ = std::move(other.row_offsets_);
                col_offsets_ = std::move(other.col_offsets_);
                row_offsets_valid_ = other.row_offsets_valid_;
                col_offsets_valid_ = other.col_offsets_valid_;
                other.n_rows_ = 0;
                other.n_cols_ = 0;
                other.row_offsets_valid_ = false;
                other.col_offsets_valid_ = false;
                return *this;
            }

            //! @brief Get total shape (rows, columns) of the block matrix
            Shape shape() const override { return Shape(n_rows_, n_cols_); }

            //! @brief Get total number of blocks stored
            size_t num_blocks() const override { return blocks_.size(); }

            //! @brief Get shape of a specific block by linear index
            Shape get_block_shape(size_t idx) const override {
                if (idx >= blocks_.size()) {
                    throw std::out_of_range("Block index " + std::to_string(idx) + " out of range");
                }
                auto it = blocks_.begin();
                std::advance(it, idx);
                return it->second.shape();
            }

            //! @brief Get raw data pointer for a specific block by linear index
            const void* get_block_data(size_t idx) const override {
                if (idx >= blocks_.size()) {
                    throw std::out_of_range("Block index " + std::to_string(idx) + " out of range");
                }
                auto it = blocks_.begin();
                std::advance(it, idx);
                return (const void*)it->second.data();
            }

            //! @brief Get row offsets for all blocks
            const std::vector<int>& get_row_offsets() const override {
                if (!row_offsets_valid_) {
                    compute_offsets();
                }
                return row_offsets_;
            }

            //! @brief Get column offsets for all blocks
            const std::vector<int>& get_col_offsets() const override {
                if (!col_offsets_valid_) {
                    compute_offsets();
                }
                return col_offsets_;
            }

            //! @brief Element access operator (mutable)
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
                // Element is outside all blocks; return cached zero
                return zero_value_;
            }

            //! @brief Get element row position for a block by linear index
            size_t get_block_row(size_t idx) const override {
                size_t count = 0;
                for (const auto& [pos, block] : blocks_) {
                    if (count == idx) {
                        return pos.first;  // Element-space row position
                    }
                    count++;
                }
                throw std::out_of_range("Block index out of range");
            }

            //! @brief Get element column position for a block by linear index
            size_t get_block_col(size_t idx) const override {
                size_t count = 0;
                for (const auto& [pos, block] : blocks_) {
                    if (count == idx) {
                        return pos.second;  // Element-space column position
                    }
                    count++;
                }
                throw std::out_of_range("Block index out of range");
            }

            //! @brief Element access operator (const)
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
                // Element is outside all blocks; return cached zero
                return zero_value_;
            }

            // ========== BlockMatrix-Specific Methods ==========

            //! @brief Get number of block rows (max row position + 1)
            size_t num_block_rows() const {
                size_t max_row = 0;
                for (const auto& [pos, block] : blocks_) {
                    max_row = std::max(max_row, pos.first + 1);
                }
                return max_row;
            }

            //! @brief Get number of block columns (max column position + 1)
            size_t num_block_cols() const {
                size_t max_col = 0;
                for (const auto& [pos, block] : blocks_) {
                    max_col = std::max(max_col, pos.second + 1);
                }
                return max_col;
            }

            //! @brief Check if a block exists at position (i,j)
            bool has_block(size_t i, size_t j) const {
                return blocks_.find({i, j}) != blocks_.end();
            }

            //! @brief Add or update a block at position (i,j)
            //! @param[in] i element row where block starts
            //! @param[in] j element column where block starts
            //! @param[in] block the block matrix to store
            //! @throws std::runtime_error if block overlaps with existing blocks
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

                blocks_[{i, j}] = block;
                update_dimensions();

                // Invalidate offset caches since block configuration changed
                row_offsets_valid_ = false;
                col_offsets_valid_ = false;
            }

            //! @brief Get block at position (i,j)
            //! @throws std::out_of_range if block doesn't exist
            Matrix<T, Allocator>& get_block(size_t i, size_t j) {
                auto key = std::make_pair(i, j);
                auto it = blocks_.find(key);
                if (it == blocks_.end()) {
                    throw std::out_of_range("Block at (" + std::to_string(i) + "," + std::to_string(j) + ") does not exist");
                }
                return it->second;
            }

            //! @brief Get block at position (i,j) (const)
            //! @throws std::out_of_range if block doesn't exist
            const Matrix<T, Allocator>& get_block(size_t i, size_t j) const {
                auto key = std::make_pair(i, j);
                auto it = blocks_.find(key);
                if (it == blocks_.end()) {
                    throw std::out_of_range("Block at (" + std::to_string(i) + "," + std::to_string(j) + ") does not exist");
                }
                return it->second;
            }

            //! @brief Print the block matrix structure
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
            //! @brief Total number of rows (sum of all block rows)
            size_t n_rows_;

            //! @brief Total number of columns (sum of all block columns)
            size_t n_cols_;

            //! @brief Blocks stored as a map indexed by (i,j) block coordinates
            std::map<std::pair<size_t, size_t>, Matrix<T, Allocator>> blocks_;

            //! @brief Cached row offsets for blocks (mutable for lazy computation)
            mutable std::vector<int> row_offsets_;

            //! @brief Cached column offsets for blocks (mutable for lazy computation)
            mutable std::vector<int> col_offsets_;

            //! @brief Flag indicating if offsets cache is valid
            mutable bool row_offsets_valid_ = false;

            //! @brief Flag indicating if column offsets cache is valid
            mutable bool col_offsets_valid_ = false;

            //! @brief Zero value for structural zeros outside blocks (mutable for const access)
            mutable T zero_value_ = (T)0;

            //! @brief Compute and cache block offsets
            void compute_offsets() const {
                row_offsets_.clear();
                col_offsets_.clear();

                int row_offset = 0;
                int col_offset = 0;

                for (const auto& [pos, block] : blocks_) {
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

            //! @brief Update total matrix dimensions based on stored blocks
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
        };

    } // namespace gpu
} // namespace lahva
