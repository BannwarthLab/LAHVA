#pragma once
#include "impl/tensor/allocators.hpp"
#include "impl/tensor/cpu/matrix.hpp"
#include "impl/tensor/cpu/tensor.hpp"
#include <omp.h>
#include <algorithm>
#include <vector>

namespace lahva
{
    namespace cpu
    {



    template<typename T>
    class BlockDiagMatrix_ : public virtual Tensor<T>
    {
        public:
            BlockDiagMatrix_() = default;
            BlockDiagMatrix_(const BlockDiagMatrix_ &) = default;
            BlockDiagMatrix_ &operator=(const BlockDiagMatrix_ &) = default;
            BlockDiagMatrix_(BlockDiagMatrix_ &&) noexcept = default;
            BlockDiagMatrix_ &operator=(BlockDiagMatrix_ &&) noexcept { return *this; }
            virtual ~BlockDiagMatrix_() = default;

            //! @brief Get the shape of the block diagonal matrix.
            //! @return Shape object with (n_rows, n_cols) representing total matrix dimensions
            virtual Shape shape() const  = 0;

            //! @brief Get the number of blocks in the block diagonal matrix.
            //! @return Number of diagonal blocks
            virtual size_t num_blocks() const = 0;

            //! @brief Get the shape of a specific block.
            //! @param[in] idx Block index
            //! @return Shape of the block at index idx
            virtual Shape get_block_shape(size_t idx) const = 0;

            //! @brief Get raw data pointer to a block.
            //! @param[in] idx Block index
            //! @return Const void pointer to block data
            virtual const void* get_block_data(size_t idx) const = 0;

            //! @brief Get cumulative row offsets of all blocks.
            //! @return Const reference to vector of row offsets where offsets[i] is cumulative row position of block i
            virtual const std::vector<int>& get_row_offsets() const = 0;

            //! @brief Get cumulative column offsets of all blocks.
            //! @return Const reference to vector of column offsets where offsets[i] is cumulative column position of block i
            virtual const std::vector<int>& get_col_offsets() const = 0;
    };

    template <class T, class Allocator = StdAllocator<T>>
    class BlockDiagMatrix : virtual public Tensor<T, Allocator>, virtual public BlockDiagMatrix_<T>
    {
        using alloc_ptr = CPUAllocator<T>;
    protected:
        //! @brief Total number of rows in the block diagonal matrix
        size_t n_rows_ = 0;

        //! @brief Total number of columns in the block diagonal matrix
        size_t n_cols_ = 0;

        //! @brief Vector of matrices representing the diagonal blocks
        std::vector<Matrix<T, Allocator>> matrices;

        //! @brief Vector of block sizes (row dimension of each block)
        std::vector<size_t> block_sizes_;

        //! @brief Cumulative row offsets for each block (row_offsets_[i] is the starting row of block i)
        std::vector<int> row_offsets_;

        //! @brief Cumulative column offsets for each block (col_offsets_[i] is the starting column of block i)
        std::vector<int> col_offsets_;

    public:
        //! @brief Default constructor creates an empty block diagonal matrix
        BlockDiagMatrix() = default;

        //! @brief Virtual destructor for proper polymorphic cleanup
        virtual ~BlockDiagMatrix() = default;

        //! @brief Copy constructor creates a deep copy of another block diagonal matrix
        //! @param[in] other The block diagonal matrix to copy
        BlockDiagMatrix(const BlockDiagMatrix &other) :
            n_rows_{other.n_rows_}, n_cols_{other.n_cols_},
            matrices{other.matrices},
            block_sizes_{other.block_sizes_},
            row_offsets_{other.row_offsets_},
            col_offsets_{other.col_offsets_}
        {}

        //! @brief Copy assignment operator
        //! @param[in] other The block diagonal matrix to copy
        //! @return Reference to this matrix
        BlockDiagMatrix &operator=(const BlockDiagMatrix &other)
        {
            if (this != &other)
            {
                n_rows_ = other.n_rows_;
                n_cols_ = other.n_cols_;
                matrices = other.matrices;
                block_sizes_ = other.block_sizes_;
                row_offsets_ = other.row_offsets_;
                col_offsets_ = other.col_offsets_;
            }
            return *this;
        }

        //! @brief Move constructor transfers ownership of another block diagonal matrix
        //! @param[in] other The block diagonal matrix to move from (will be emptied)
        BlockDiagMatrix(BlockDiagMatrix &&other) noexcept :
            n_rows_{other.n_rows_}, n_cols_{other.n_cols_},
            matrices{std::move(other.matrices)},
            block_sizes_{std::move(other.block_sizes_)},
            row_offsets_{std::move(other.row_offsets_)},
            col_offsets_{std::move(other.col_offsets_)}
        {
            other.n_rows_ = 0;
            other.n_cols_ = 0;
        }

        //! @brief Move assignment operator
        //! @param[in] other The block diagonal matrix to move from (will be emptied)
        //! @return Reference to this matrix
        BlockDiagMatrix &operator=(BlockDiagMatrix &&other) noexcept
        {
            if (this != &other)
            {
                Tensor<T, Allocator>::operator=(std::move(other));
                BlockDiagMatrix_<T>::operator=(std::move(other));
                n_rows_ = other.n_rows_;
                n_cols_ = other.n_cols_;
                matrices = std::move(other.matrices);
                block_sizes_ = std::move(other.block_sizes_);
                row_offsets_ = std::move(other.row_offsets_);
                col_offsets_ = std::move(other.col_offsets_);
                other.n_rows_ = 0;
                other.n_cols_ = 0;
            }
            return *this;
        }

        //! @brief Constructor creates n_blocks identical blocks with given shape
        //! @param[in] n_blocks Number of diagonal blocks
        //! @param[in] block_shape Shape of each block (n_rows, n_cols)
        BlockDiagMatrix(size_t n_blocks, Shape block_shape)
        {
            matrices.reserve(n_blocks);
            block_sizes_.reserve(n_blocks);
            for (size_t i = 0; i < n_blocks; ++i)
                matrices.emplace_back(block_shape);
            build_offsets_();
        }

        //! @brief Constructor creates n_blocks identical blocks with given shape initialized to value
        //! @param[in] n_blocks Number of diagonal blocks
        //! @param[in] block_shape Shape of each block (n_rows, n_cols)
        //! @param[in] val Initial value for all block elements
        BlockDiagMatrix(size_t n_blocks, Shape block_shape, T val)
        {
            matrices.reserve(n_blocks);
            block_sizes_.reserve(n_blocks);
            for (size_t i = 0; i < n_blocks; ++i)
                matrices.emplace_back(block_shape, val);
            build_offsets_();
        }

        //! @brief Constructor creates blocks from vector of shapes
        //! @param[in] shapes Vector of Shape objects, one per block
        explicit BlockDiagMatrix(const std::vector<Shape> &shapes)
        {
            matrices.reserve(shapes.size());
            block_sizes_.reserve(shapes.size());
            for (const auto &s : shapes)
                matrices.emplace_back(s);
            build_offsets_();
        }

        //! @brief Constructor creates blocks from vector of shapes, initialized to value
        //! @param[in] shapes Vector of Shape objects, one per block
        //! @param[in] val Initial value for all block elements
        BlockDiagMatrix(const std::vector<Shape> &shapes, T val)
        {
            matrices.reserve(shapes.size());
            block_sizes_.reserve(shapes.size());
            for (const auto &s : shapes)
                matrices.emplace_back(s, val);
            build_offsets_();
        }

        //! @brief Constructor from vector of block matrices (copy)
        //! @param[in] blocks Vector of matrices to use as blocks
        explicit BlockDiagMatrix(const std::vector<Matrix<T, Allocator>> &blocks)
            : matrices(blocks)
        {
            block_sizes_.reserve(matrices.size());
            build_offsets_();
        }

        //! @brief Constructor from vector of block matrices (move)
        //! @param[in] blocks Vector of matrices to move as blocks (will be emptied)
        explicit BlockDiagMatrix(std::vector<Matrix<T, Allocator>> &&blocks)
            : matrices(std::move(blocks))
        {
            block_sizes_.reserve(matrices.size());
            build_offsets_();
        }

        //! @brief Non-const element access operator
        //! @param[in] i Row index
        //! @param[in] j Column index
        //! @return Reference to element at (i, j), or reference to static zero if out of block bounds
        T &operator()(size_t i, size_t j)
        {
            assert(i < n_rows_ && j < n_cols_);
            auto it = std::upper_bound(row_offsets_.begin(), row_offsets_.end(), static_cast<int>(i));
            size_t b = static_cast<size_t>(std::distance(row_offsets_.begin(), it)) - 1;
            if (j >= static_cast<size_t>(col_offsets_[b]) && j < static_cast<size_t>(col_offsets_[b + 1]))
                return matrices[b](i - row_offsets_[b], j - col_offsets_[b]);
            static T zero{};
            return zero;
        }

        //! @brief Const element access operator
        //! @param[in] i Row index
        //! @param[in] j Column index
        //! @return Const reference to element at (i, j), or reference to static zero if out of block bounds
        const T &operator()(size_t i, size_t j) const
        {
            assert(i < n_rows_ && j < n_cols_);
            auto it = std::upper_bound(row_offsets_.begin(), row_offsets_.end(), static_cast<int>(i));
            size_t b = static_cast<size_t>(std::distance(row_offsets_.begin(), it)) - 1;
            if (j >= static_cast<size_t>(col_offsets_[b]) && j < static_cast<size_t>(col_offsets_[b + 1]))
                return matrices[b](i - row_offsets_[b], j - col_offsets_[b]);
            static const T zero{};
            return zero;
        }

        //! @brief In-place scalar addition to all block elements
        //! @param[in] val Value to add to all elements
        //! @return Reference to this matrix after addition
        BlockDiagMatrix &operator+=(T val)
        {
            for (auto &m : matrices)
                m += val;
            return *this;
        }

        //! @brief Get the total shape of the block diagonal matrix
        //! @return Shape object with (total_rows, total_cols)
        Shape shape() const override { return Shape{n_rows_, n_cols_}; }

        //! @brief Get the number of diagonal blocks
        //! @return Total number of blocks in the block diagonal matrix
        size_t num_blocks() const override { return matrices.size(); }

        //! @brief Get the shape of a specific block (override from base class)
        //! @param[in] idx Block index
        //! @return Shape of block at index idx
        Shape get_block_shape(size_t idx) const override {
            return matrices[idx].shape();
        }

        //! @brief Get shapes of all blocks
        //! @return Vector of shape vectors, one pair per block
        std::vector<std::vector<size_t>> block_shapes() const
            {
                std::vector<std::vector<size_t>> block_shapes;
                for (size_t i = 0; i < matrices.size(); i++) {
                    Shape s = matrices[i].shape();
                    block_shapes.push_back({s.first, s.second});
                }
                return block_shapes;
            }

        //! @brief Extract concatenated diagonals of all blocks
        //! @return Vector containing diagonal elements from all blocks in order
        Vector<T, Allocator> get_diagonal() const
        {
            size_t total = 0;
            for (const auto &m : matrices)
            {
                Shape s = m.shape();
                total += std::min(s.first, s.second);
            }
            Vector<T, Allocator> diag(total);
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

        //! @brief Set diagonal of each block from a flat vector of concatenated block diagonals
        //! @param[in] diag Vector containing diagonal elements for all blocks in order
        void set_diagonal(const Vector<T, Allocator> &diag)
        {
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

        //! @brief Symmetrize each block in-place: block = (block + block^T) / 2
        void symmetrize()
        {
            for (auto &m : matrices)
                m.symmetrize();
        }

        template <typename... Args>
        void symmetrize(const CPURuntime &rt_, Args &&...args)
        {
            (symmetrize(args...));
        }

        template <typename... Args>
        Vector<T, Allocator> get_diagonal(const CPURuntime &rt_, Args &&...args)
        {
            return get_diagonal(args...);
        }

        template <typename... Args>
        void set_diagonal(const CPURuntime &rt_, Args &&...args)
        {
            (set_diagonal(args...));
        }

        //! @brief Get raw data pointer to a block (override from base class)
        //! @param[in] idx Block index
        //! @return Const void pointer to block data at index idx
        const void* get_block_data(size_t idx) const override {
            return static_cast<const void*>(matrices[idx].data());
        }

        //! @brief Get const reference to a block matrix
        //! @param[in] idx Block index
        //! @return Const reference to the block at index idx
        const Matrix<T, Allocator>& get_block(size_t idx) const { return matrices[idx]; }

        //! @brief Get cumulative row offsets of all blocks (override from base class)
        //! @return Const reference to vector where offsets[i] is cumulative row position of block i
        const std::vector<int>& get_row_offsets() const override { return row_offsets_; }

        //! @brief Get cumulative column offsets of all blocks (override from base class)
        //! @return Const reference to vector where offsets[i] is cumulative column position of block i
        const std::vector<int>& get_col_offsets() const override { return col_offsets_; }

        //! @brief Add a new block to the block diagonal matrix
        //! @param[in] block The matrix to add as a new diagonal block
        //! @note Updates total dimensions and all offset arrays
        void add_block(const Matrix<T, Allocator> &block) {
            matrices.push_back(block);
            block_sizes_.push_back(block.shape().first);
            n_rows_ += block.shape().first;
            n_cols_ += block.shape().second;

            // Update offsets
            if (col_offsets_.empty()) {
                col_offsets_.push_back(0);
                row_offsets_.push_back(0);
            }
            col_offsets_.push_back(col_offsets_.back() + block.shape().second);
            row_offsets_.push_back(row_offsets_.back() + block.shape().first);
        }

        //! @brief Print all blocks to console
        void print() const {
            for (size_t i=0; i < matrices.size(); i++) {
                matrices[i].print();
            }
        }

        //! @brief Print all blocks to file
        //! @param[in] file File path to write block data to
        void print(const char* file) const
        {
            for (size_t i=0; i < matrices.size(); i++) {
                matrices[i].print(file);
            }
        }

    private:
        //! @brief Validate that block dimensions don't exceed size limits
        //! @param[in] n_rows Number of rows
        //! @param[in] n_cols Number of columns
        //! @throws std::out_of_range if size exceeds SIZE_MAX / n_rows
        static void check_size_(size_t n_rows, size_t n_cols);

        //! @brief Build cumulative offset arrays from block matrices
        //! @note Called after blocks are added to initialize row_offsets_ and col_offsets_
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
                block_sizes_.push_back(s.first);
                row_offsets_.push_back(row_offsets_.back() + static_cast<int>(s.first));
                col_offsets_.push_back(col_offsets_.back() + static_cast<int>(s.second));
            }
        }
    };    

    //! @brief Check that total block size is within valid bounds
    //! @param[in] n_rows Number of rows in block diagonal matrix
    //! @param[in] n_cols Number of columns in block diagonal matrix
    //! @throws std::out_of_range if total size exceeds SIZE_MAX / n_rows
    template <typename T, class Allocator>
    void BlockDiagMatrix<T, Allocator>::check_size_(size_t n_rows, size_t n_cols)
    {
        if (n_rows != 0 && n_cols > SIZE_MAX / n_rows)
        {
            throw std::out_of_range("Block size exceeds maximum representable size.");
        }
    }

    } // namespace cpu
} // namespace lahva
