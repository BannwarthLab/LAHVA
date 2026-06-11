#pragma once
#include "impl/tensor/allocators.hpp"
#include "impl/tensor/cpu/tensor.hpp"
#include "impl/tensor/cpu/matrix.hpp"
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
            virtual Shape shape() const  = 0;
            virtual size_t num_blocks() const = 0;
    };

    template <class T, class Allocator = StdAllocator<T>>
    class BlockDiagMatrix : virtual public CPUTensor<T, Allocator>, virtual public BlockDiagMatrix_<T>
    {
        using alloc_ptr = CPUAllocator<T>;
    protected:
        // Total matrix shape in each dimension
        size_t n_rows_ = 0;
        size_t n_cols_ = 0;
        
        std::vector<Matrix<T, Allocator>> matrices;
        std::vector<size_t> block_sizes_;
        std::vector<int> row_offsets_;  // Cumulative row offsets for each block
        std::vector<int> col_offsets_;  // Cumulative column offsets for each block

    public:
        BlockDiagMatrix() = default;
        virtual ~BlockDiagMatrix() = default;

        BlockDiagMatrix(const BlockDiagMatrix &other) :
            n_rows_{other.n_rows_}, n_cols_{other.n_cols_},
            matrices{other.matrices},
            block_sizes_{other.block_sizes_},
            row_offsets_{other.row_offsets_},
            col_offsets_{other.col_offsets_}
        {}

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

        BlockDiagMatrix &operator=(BlockDiagMatrix &&other) noexcept
        {
            if (this != &other)
            {
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

        BlockDiagMatrix(size_t n_blocks, Shape block_shape)
        {
            matrices.reserve(n_blocks);
            block_sizes_.reserve(n_blocks);
            for (size_t i = 0; i < n_blocks; ++i)
                matrices.emplace_back(block_shape);
            build_offsets_();
        }

        BlockDiagMatrix(size_t n_blocks, Shape block_shape, T val)
        {
            matrices.reserve(n_blocks);
            block_sizes_.reserve(n_blocks);
            for (size_t i = 0; i < n_blocks; ++i)
                matrices.emplace_back(block_shape, val);
            build_offsets_();
        }

        explicit BlockDiagMatrix(const std::vector<Shape> &shapes)
        {
            matrices.reserve(shapes.size());
            block_sizes_.reserve(shapes.size());
            for (const auto &s : shapes)
                matrices.emplace_back(s);
            build_offsets_();
        }

        BlockDiagMatrix(const std::vector<Shape> &shapes, T val)
        {
            matrices.reserve(shapes.size());
            block_sizes_.reserve(shapes.size());
            for (const auto &s : shapes)
                matrices.emplace_back(s, val);
            build_offsets_();
        }

        explicit BlockDiagMatrix(const std::vector<Matrix<T, Allocator>> &blocks)
            : matrices(blocks)
        {
            block_sizes_.reserve(matrices.size());
            build_offsets_();
        }

        explicit BlockDiagMatrix(std::vector<Matrix<T, Allocator>> &&blocks)
            : matrices(std::move(blocks))
        {
            block_sizes_.reserve(matrices.size());
            build_offsets_();
        }

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

        //! @brief in-place, scalar addition
        BlockDiagMatrix &operator+=(T val)
        {
            for (auto &m : matrices)
                m += val;
            return *this;
        }

        //! @return number of rows/columns of the Matrix
        Shape shape() const override { return Shape{n_rows_, n_cols_}; }

        //! @return number of blocks
        size_t num_blocks() const override { return matrices.size(); }

        //! @return number of rows/columns of the submatrices
        std::vector<std::vector<size_t>> block_shapes() const
            {
                std::vector<std::vector<size_t>> block_shapes;
                for (size_t i = 0; i < matrices.size(); i++) {
                    Shape s = matrices[i].shape();
                    block_shapes.push_back({s.first, s.second});
                }
                return block_shapes;
            }

        //! @return concatenated diagonals of all blocks
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

        //! @brief set diagonal of each block from a flat vector of concatenated block diagonals
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

        //! @brief symmetrize each block in-place: block = (block + block^T) / 2
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

        //! @return const void pointer to block data (for compatibility with abstract interface)
        const void* get_block_data(size_t idx) const {
            return static_cast<const void*>(matrices[idx].data());
        }

        //! @return const reference to a block matrix
        const Matrix<T, Allocator>& get_block(size_t idx) const { return matrices[idx]; }

        //! @return const reference to row offsets (cumulative row position of each block)
        const std::vector<int>& get_row_offsets() const { return row_offsets_; }

        //! @return const reference to column offsets (cumulative column position of each block)
        const std::vector<int>& get_col_offsets() const { return col_offsets_; }

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

        //! prints the Matrix as string
        void print() const {
            for (size_t i=0; i < matrices.size(); i++) {
                matrices[i].print();
            }
        }

        void print(const char* file) const
        {
            for (size_t i=0; i < matrices.size(); i++) {
                matrices[i].print(file);
            }
        }

    private:
        static void check_size_(size_t n_rows, size_t n_cols);

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
