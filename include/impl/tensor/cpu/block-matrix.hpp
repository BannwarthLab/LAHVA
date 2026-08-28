#pragma once

#include "impl/tensor/cpu/matrix.hpp"
#include <iomanip>

namespace lahva
{
    namespace cpu
    {
        /// @brief Abstract base class for block matrix operations
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

            /// @brief Get number of blocks in the block matrix
            /// @return Number of blocks
            virtual size_t num_blocks() const = 0;

            /// @brief Get shape of a specific block
            /// @param[in] idx Block index
            /// @return Shape object with block dimensions
            virtual Shape get_block_shape(size_t idx) const = 0;

            /// @brief Get row offset array for all blocks
            /// @return Const reference to vector of cumulative row offsets
            virtual const std::vector<int>& get_row_offsets() const = 0;

            /// @brief Get column offset array for all blocks
            /// @return Const reference to vector of cumulative column offsets
            virtual const std::vector<int>& get_col_offsets() const = 0;

            /// @brief Get element row position for a block by linear index
            /// @param[in] idx Block linear index
            /// @return Element-space row position
            virtual size_t get_block_row(size_t idx) const = 0;

            /// @brief Get element column position for a block by linear index
            /// @param[in] idx Block linear index
            /// @return Element-space column position
            virtual size_t get_block_col(size_t idx) const = 0;
        };

        /// @brief CPU block matrix with blocks at arbitrary positions
        ///
        /// Inherits from Matrix and stores blocks in a single large dense matrix.
        /// All BLAS operations are inherited from Matrix. Block metadata is tracked
        /// separately for block-level access.
        ///
        /// @tparam T Numeric element type (double, float, complex types)
        /// @tparam Allocator Host memory allocator (default: StdAllocator<T>)
        template <class T, class Allocator = StdAllocator<T>>
        class BlockMatrix : public Matrix<T, Allocator>, virtual public BlockMatrix_<T>
        {
        public:
            /// @brief Default constructor - creates empty 0x0 matrix
            BlockMatrix() : BlockMatrix_<T>(), Matrix<T, Allocator>(Shape{0, 0}, (T)0)
            {
            }

            /// @brief Constructor from shape - allocates dense backing matrix for given dimensions
            explicit BlockMatrix(const Shape &shape, const Allocator &alloc = Allocator())
                : Tensor<T, Allocator>(shape.first * shape.second, alloc), BlockMatrix_<T>(), Matrix<T, Allocator>(shape, (T)0, alloc)
            {
            }

            /// @brief Copy constructor
            BlockMatrix(const BlockMatrix &other) = default;

            /// @brief Move constructor
            BlockMatrix(BlockMatrix &&other) noexcept = default;

            /// @brief Copy assignment operator
            BlockMatrix &operator=(const BlockMatrix &other) = default;

            /// @brief Move assignment operator
            BlockMatrix &operator=(BlockMatrix &&other) noexcept = default;

            /// @brief Get total number of blocks stored
            size_t num_blocks() const override { return block_info_.size(); }

            /// @brief Get shape of a specific block by linear index
            Shape get_block_shape(size_t idx) const override {
                if (idx >= block_info_.size()) {
                    throw std::out_of_range("Block index " + std::to_string(idx) + " out of range");
                }
                return block_info_[idx].shape;
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

            /// @brief Get element row position for a block by linear index
            size_t get_block_row(size_t idx) const override {
                if (idx >= block_info_.size()) {
                    throw std::out_of_range("Block index out of range");
                }
                return block_info_[idx].row;
            }

            /// @brief Get element column position for a block by linear index
            size_t get_block_col(size_t idx) const override {
                if (idx >= block_info_.size()) {
                    throw std::out_of_range("Block index out of range");
                }
                return block_info_[idx].col;
            }

            /// @brief Print the full block matrix
            void print() const {
                // Call parent Matrix print if data is allocated
                if (this->data() != nullptr && this->n_rows_ > 0 && this->n_cols_ > 0) {
                    Matrix<T, Allocator>::print();
                } else {
                    std::cout << "BlockMatrix: uninitialized or empty (shape=(" << this->n_rows_ << ", "
                             << this->n_cols_ << "))" << std::endl;
                }
            }

            // ========== BlockMatrix-Specific Methods ==========

            /// @brief Get number of block rows (max row position + 1)
            size_t num_block_rows() const {
                size_t max_row = 0;
                for (const auto& info : block_info_) {
                    max_row = std::max(max_row, info.row + 1);
                }
                return max_row;
            }

            /// @brief Get number of block columns (max column position + 1)
            size_t num_block_cols() const {
                size_t max_col = 0;
                for (const auto& info : block_info_) {
                    max_col = std::max(max_col, info.col + 1);
                }
                return max_col;
            }

            /// @brief Check if a block exists at position (i,j)
            bool has_block(size_t i, size_t j) const {
                for (const auto& info : block_info_) {
                    if (info.row == i && info.col == j) {
                        return true;
                    }
                }
                return false;
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

                for (const auto& info : block_info_) {
                    if (info.row == i && info.col == j) {
                        continue;
                    }

                    size_t exist_row_start = info.row;
                    size_t exist_col_start = info.col;
                    Shape exist_shape = info.shape;
                    size_t exist_row_end = exist_row_start + exist_shape.first;
                    size_t exist_col_end = exist_col_start + exist_shape.second;

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

                size_t matrix_rows = std::max(static_cast<size_t>(this->shape().first), new_row_end);
                size_t matrix_cols = std::max(static_cast<size_t>(this->shape().second), new_col_end);

                if (this->data() == nullptr ||
                    matrix_rows > static_cast<size_t>(this->shape().first) ||
                    matrix_cols > static_cast<size_t>(this->shape().second)) {
                    Matrix<T, Allocator> new_matrix(Shape{matrix_rows, matrix_cols}, (T)0);
                    if (this->shape().first > 0 && this->shape().second > 0) {
                        for (size_t col = 0; col < static_cast<size_t>(this->shape().second); ++col) {
                            for (size_t row = 0; row < static_cast<size_t>(this->shape().first); ++row) {
                                new_matrix(row, col) = (*this)(row, col);
                            }
                        }
                    }
                    Matrix<T, Allocator>::operator=(std::move(new_matrix));
                }

                for (size_t col = 0; col < new_shape.second; ++col) {
                    for (size_t row = 0; row < new_shape.first; ++row) {
                        (*this)(i + row, j + col) = block(row, col);
                    }
                }

                bool found = false;
                for (auto& info : block_info_) {
                    if (info.row == i && info.col == j) {
                        info.shape = new_shape;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    block_info_.push_back({i, j, new_shape});
                }

                row_offsets_valid_ = false;
                col_offsets_valid_ = false;
            }

        private:
            struct BlockInfo {
                size_t row;
                size_t col;
                Shape shape;
            };

            /// @brief Block metadata (position and shape)
            std::vector<BlockInfo> block_info_;

            /// @brief Cached row offsets for blocks (mutable for lazy computation)
            mutable std::vector<int> row_offsets_;

            /// @brief Cached column offsets for blocks (mutable for lazy computation)
            mutable std::vector<int> col_offsets_;

            /// @brief Flag indicating if row offsets cache is valid
            mutable bool row_offsets_valid_ = false;

            /// @brief Flag indicating if column offsets cache is valid
            mutable bool col_offsets_valid_ = false;

            /// @brief Compute and cache block offsets
            void compute_offsets() const {
                row_offsets_.clear();
                col_offsets_.clear();

                int row_offset = 0;
                int col_offset = 0;

                for (const auto& info : block_info_) {
                    row_offsets_.push_back(row_offset);
                    col_offsets_.push_back(col_offset);
                    row_offset += static_cast<int>(info.shape.first);
                    col_offset += static_cast<int>(info.shape.second);
                }

                row_offsets_.push_back(row_offset);
                col_offsets_.push_back(col_offset);

                row_offsets_valid_ = true;
                col_offsets_valid_ = true;
            }
        };

    } // namespace cpu
} // namespace lahva
