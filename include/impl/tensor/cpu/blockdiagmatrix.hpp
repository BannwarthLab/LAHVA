#pragma once
#include "impl/tensor/allocators.hpp"
#include "impl/tensor/cpu/tensor.hpp"
#include <omp.h>
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

    //! @brief slim wrapper around a float or double array to allow easy acces with
    //!        two indeces using the () operator.
    //!
    //! If NDEBUG is **not** defined, range checks are performed.  d
    //! The data is stored in column-major order in a 1D array.
    //!
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
        BlockDiagMatrix() {};
        virtual ~BlockDiagMatrix() {}

        // //! @param[in] i row index
        // //! @param[in] j column index
        // //! @return reference to Matrix element i,j
        // T &operator()(size_t i, size_t j);
        // //! @param[in] i row index
        // //! @param[in] j column index
        // //! @return reference to Matrix element i,j
        // const T &operator()(size_t i, size_t j) const;

        // //! @brief in-place, scalar addition
        // BlockDiagMatrix &operator+=(T val);

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
    };    //     void print(const char* file) const
    //     {
    //         for (size_t i=0; i < matrices.size(); i++) {
    //             matrices[i].print(file);
    //         }

    //     }
    // };

    //  template <typename T, class Allocator>
    // void BlockDiagMatrix<T, Allocator>::check_size_(size_t n_rows,
    //                             size_t n_cols)
    // {
    //     if (n_cols > SIZE_MAX)
    //     {
    //         throw std::out_of_range("Number of columns exceeds maximum Matrix size.");
    //     }
    //     else if (n_rows > SIZE_MAX)
    //     {
    //         throw std::out_of_range("Number of rows exceeds maximum Matrix size.");
    //     }
    //     else if (data_size_(n_rows, n_cols) > SIZE_MAX)
    //     {
    //         throw std::out_of_range("Vector exceeds maximum Matrix size.");
    //     }
    // }

    // // copy operations
    //  template <typename T, class Allocator>
    // BlockDiagMatrix<T, Allocator>::BlockDiagMatrix(const BlockDiagMatrix<T, Allocator> &other) :
    // CPUTensor<T, Allocator>{other},
    // n_rows_{other.n_rows_}, n_cols_{other.n_cols_}
    // {

    // }

    // template <typename T, class Allocator>
    // BlockDiagMatrix<T, Allocator> &BlockDiagMatrix<T, Allocator>::operator=(const BlockDiagMatrix<T, Allocator> &other)
    // {
    //     if (this != &other)
    //     {
    //         CPUTensor<T, Allocator>::operator=(other);

    //         n_rows_ = other.n_rows_;
    //         n_cols_ = other.n_cols_;
    //     }

    //     return *this;
    // }

    // // move operations
    // template <typename T, class Allocator>
    // BlockDiagMatrix<T, Allocator>::BlockDiagMatrix(BlockDiagMatrix<T, Allocator> &&other) :
    // CPUTensor<T, Allocator>{other},
    // n_rows_{other.n_rows_}, n_cols_{other.n_cols_}
    // {

    //     other.n_rows_ = 0;
    //     other.n_cols_ = 0;

    // }

    //  template <typename T, class Allocator>
    // BlockDiagMatrix<T, Allocator> &BlockDiagMatrix<T, Allocator>::operator=(BlockDiagMatrix<T, Allocator> &&other)
    // {
    //     if (this != &other)
    //     {
    //         CPUTensor<T, Allocator>::operator=(std::move(other));

    //         n_rows_ = other.n_rows_;
    //         n_cols_ = other.n_cols_;

    //         other.n_rows_ = 0;
    //         other.n_cols_ = 0;
    //     }
    //     return *this;
    // }

    //  template <typename T, class Allocator>
    // T &BlockDiagMatrix<T, Allocator>::operator()(size_t i, size_t j)
    // {
    //     return this->data_[this->data_id_(i, j)];
    // }

    //  template <typename T, class Allocator>
    // const T &BlockDiagMatrix<T, Allocator>::operator()(size_t i, size_t j) const
    // {
    //     return this->data_[this->data_id_(i, j)];
    // }

    //  template <typename T, class Allocator>
    // BlockDiagMatrix<T, Allocator> &BlockDiagMatrix<T, Allocator>::operator+=(T val)
    // {
    // #pragma omp for
    //     for (size_t i = 0; i < data_size_(n_rows_, n_cols_); i++)
    //     {
    //         this->data_[i] += val;
    //     }

    //     return *this;
    // }

    //  template <typename T, class Allocator>
    // void BlockDiagMatrix<T, Allocator>::print() const
    // {
    //     for (size_t i=0; i < matrices.size(); i++) {
    //         matrices[i].print();
    //     }

    } // namespace cpu
} // namespace lahva
