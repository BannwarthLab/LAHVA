#pragma once

#include "impl/tensor/allocators.hpp"
#include "impl/tensor/cpu/tensor.hpp"
#include "impl/tensor/cpu/vector.hpp"
#include "impl/blas/cpu/level1.h"
#include <initializer_list>
#include <omp.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cassert>

namespace lahva
{
    namespace cpu
    {
        
    
    
        template<typename T>
        class Matrix_ : public virtual Tensor<T>
        {
            public:
                virtual Shape shape() const  = 0;

                /// @param[in] i row index
                /// @param[in] j column index
                /// @return reference to Matrix element i,j
                virtual T &operator()(size_t i, size_t j) = 0;

                /// @param[in] i row index
                /// @param[in] j column index
                /// @return reference to Matrix element i,j
                virtual const T &operator()(size_t i, size_t j) const = 0;

        };


        /// @brief Matrix wrapper
        /// If NDEBUG is **not** defined, range checks are performed.
        /// The data is stored in column-major order in a 1D array.
        ///
        template <class T, class Allocator = StdAllocator<T>>
        class Matrix : virtual public CPUTensor<T, Allocator>, virtual public Matrix_<T>
        {
            using alloc_ptr = CPUAllocator<T>;
        protected:
            /// @brief Number of rows in the matrix
            size_t n_rows_;

            /// @brief Number of columns in the matrix
            size_t n_cols_;

            /// @brief Calculate linear index for matrix element in column-major storage
            /// @param[in] i row index
            /// @param[in] j column index
            /// @return linear index in 1D storage array
            /// @note Range checks deactivated if NDEBUG is defined
            inline size_t data_id_(size_t i, size_t j) const
            {
                assert(i < this->n_rows_ && j < this->n_cols_);
                return i + j * n_rows_;
            }

            /// @brief Calculate total size needed for matrix storage
            /// @param[in] n_rows number of rows
            /// @param[in] n_cols number of columns
            /// @return number of elements needed: n_rows * n_cols
            inline size_t data_size_(size_t n_rows, size_t n_cols)
            {
                return n_rows * n_cols;
            }

            /// @brief Validate matrix dimensions
            /// @param[in] rows number of rows to validate
            /// @param[in] cols number of columns to validate
            /// @throws std::out_of_range if dimensions exceed maximum size
            inline void check_size_(size_t, size_t);

        public:
            Matrix() {};

            /// @brief Allocate square matrix without initialization
            /// @param[in] n matrix dimension (n x n)
            /// @param[in] alloc allocator instance for memory management
            Matrix(size_t n, const alloc_ptr &alloc = Allocator());
            template<typename U>
            Matrix(size_t n, const std::shared_ptr<CPUAllocator<U>> &alloc)
            : Matrix<T, Allocator>{n, Allocator(*alloc)} {};

            /// @brief Allocate square matrix initialized with uniform value
            /// @param[in] n matrix dimension (n x n)
            /// @param[in] val initialization value for all elements
            /// @param[in] alloc allocator instance for memory management
            Matrix(size_t n, T val, const alloc_ptr &alloc = Allocator());
            template<typename U>
            Matrix(size_t n, T val, const std::shared_ptr<CPUAllocator<U>> &alloc)
            : Matrix<T, Allocator>{n, val, Allocator(*alloc)} {};

            /// @brief Allocate matrix with specified shape without initialization
            /// @param[in] shape dimensions (rows, columns)
            /// @param[in] alloc allocator instance for memory management
            Matrix(const Shape &shape, const alloc_ptr &alloc = Allocator());

            /// @brief Allocate and initialize matrix from initializer list
            /// @param[in] shape dimensions (rows, columns)
            /// @param[in] init initializer list with element values
            /// @param[in] row_major if true, init is in row-major order; column-major if false
            /// @param[in] alloc allocator instance for memory management
            Matrix(const Shape &shape, std::initializer_list<T> init, bool row_major = false, const alloc_ptr &alloc = Allocator());
            template<typename U>
            Matrix(const Shape& shape, const std::shared_ptr<CPUAllocator<U>> &alloc)
            : Matrix<T, Allocator>{shape, Allocator(*alloc)} {};

            /// @brief Allocate matrix with specified shape initialized with uniform value
            /// @param[in] shape dimensions (rows, columns)
            /// @param[in] val initialization value for all elements
            /// @param[in] alloc allocator instance for memory management
            Matrix(const Shape &shape, T val, const alloc_ptr &alloc = Allocator());
            template<typename U>
            Matrix(const Shape &shape, T val, const std::shared_ptr<CPUAllocator<U>> &alloc)
            : Matrix<T, Allocator>{shape, val, Allocator(*alloc)} {};

            /// @brief Wrap existing data pointer with optional ownership
            /// @param[in] shape dimensions (rows, columns)
            /// @param[in] data pointer to matrix data in column-major storage format
            /// @param[in] take_ownership if true, matrix will free data on destruction; if false, external code is responsible
            /// @param[in] alloc allocator instance for memory management
            /// @note data must be in column-major format as defined by data_id_()
            Matrix(const Shape &shape, T *data, bool take_ownership = true, const alloc_ptr &alloc = Allocator());
            template<typename U>
            Matrix(const Shape &shape, T* data, bool take_ownership = true, const std::shared_ptr<CPUAllocator<U>> &alloc = Allocator())
            : Matrix<T, Allocator>{shape, data, take_ownership, Allocator(*alloc)} {};

            /// @brief Copy data from const pointer into newly allocated matrix
            /// @param[in] shape dimensions (rows, columns)
            /// @param[in] data pointer to read-only matrix data
            /// @param[in] alloc allocator instance for memory management
            Matrix(const Shape &shape, const T *data, const alloc_ptr &alloc = Allocator());
            template<typename U>
            Matrix(const Shape &shape, T* data, const std::shared_ptr<CPUAllocator<U>> &alloc)
            : Matrix<T, Allocator>{shape, data, Allocator(*alloc)} {};

            /// @brief Copy constructor
            Matrix(const Matrix &);

            /// @brief Move constructor
            Matrix(Matrix &&);

            /// @brief Copy assignment operator
            Matrix &operator=(const Matrix &);

            /// @brief Move assignment operator
            Matrix &operator=(Matrix &&);

            /// @brief Destructor - deallocates matrix data if owned
            virtual ~Matrix();

            /// @param[in] i row index
            /// @param[in] j column index
            /// @return reference to Matrix element i,j
            T &operator()(size_t i, size_t j);
            /// @param[in] i row index
            /// @param[in] j column index
            /// @return reference to Matrix element i,j
            const T &operator()(size_t i, size_t j) const;

            /// @brief in-place, scalar addition
            Matrix &operator+=(T val);

            /// @return number of rows/columns of the Matrix
            Shape shape() const { return Shape{n_rows_, n_cols_}; }

            /// prints the Matrix as string
            void print() const;

            void print(const char* file) const
            {
                std::ofstream os(file);
                os.precision(10);
                if (os.is_open())
                {
                    for (size_t i = 0; i < n_rows_; i++)
                {
                    for (size_t j = 0; j < n_cols_; j++)
                    {
                        os << this->data_[this->data_id_(i, j)] << ", ";
                    }
                    os << std::endl;
                }
                }
                else
                {
                    std::cerr << "Unable to open file";
                    return;
                }
                os.close();
                
            }

            Vector<T, Allocator> get_diagonal() const;

            void set_diagonal(const Vector<T, Allocator> &diag);

            void symmetrize();

            template <typename... Args>
            void symmetrize(const CPURuntime &rt_, Args &&...args)
            {
                (symmetrize(args...));
            } 

            template <typename... Args>
            void get_diagonal(const CPURuntime &rt_, Args &&...args)
            {
                (get_diagonal(args...));
            } 

            template <typename... Args>
            void set_diagonal(const CPURuntime &rt_, Args &&...args)
            {
                (set_diagonal(args...));
            } 

            bool ownsData() { return this->is_owner_; };
        };

        /// @brief Implementation: Validate matrix dimensions against SIZE_MAX
        template <typename T, class Allocator>
        void Matrix<T, Allocator>::check_size_(size_t n_rows,
                                    size_t n_cols)
        {
            if (n_cols > SIZE_MAX)
            {
                throw std::out_of_range("Number of columns exceeds maximum Matrix size.");
            }
            else if (n_rows > SIZE_MAX)
            {
                throw std::out_of_range("Number of rows exceeds maximum Matrix size.");
            }
            else if (data_size_(n_rows, n_cols) > SIZE_MAX)
            {
                throw std::out_of_range("Vector exceeds maximum Matrix size.");
            }
        }

        /// @brief Implementation: Allocate matrix storage without initialization
        template <typename T, class Allocator>
        Matrix<T, Allocator>::Matrix(const Shape &shape, const alloc_ptr &alloc) :
        CPUTensor<T, Allocator>{shape.first*shape.second, alloc}, n_rows_{shape.first}, n_cols_{shape.second}
        {
            check_size_(shape.first, shape.second);
        }

        /// @brief Implementation: Wrap existing data pointer with optional ownership
        template <typename T, class Allocator>
        Matrix<T, Allocator>::Matrix(const Shape &shape, T *data, bool take_ownership, const alloc_ptr &alloc) :
        CPUTensor<T, Allocator>{alloc}, n_rows_{shape.first}, n_cols_{shape.second}
        {
            this->data_ = data;
            this->count_ = n_rows_*n_cols_;
            this->is_owner_ = take_ownership;
        }

        /// @brief Implementation: Allocate and fill matrix with uniform value
        template <typename T, class Allocator>
        Matrix<T, Allocator>::Matrix(const Shape &shape, T val, const alloc_ptr &alloc) :
        Matrix(shape, alloc)
        {
            std::fill(this->data_, this->data_ + data_size_(n_rows_, n_cols_), val);
        }

        /// @brief Implementation: Initialize from initializer list with optional row/column major conversion
        template <typename T, class Allocator>
        Matrix<T, Allocator>::Matrix(const Shape &shape, std::initializer_list<T> init, bool row_major, const alloc_ptr &alloc) :
        Matrix(shape, alloc)
        {
            assert(init.size() == this->count_);

            if (row_major)
            {
#pragma omp parallel for ordered schedule(static)
                for (size_t i = 0; i < this->n_rows_; i++)
                {
                    for (size_t j = 0; j < this->n_cols_; j++)
                    {
                        this->data_[this->data_id_(i, j)] = init.begin()[j + i * n_rows_];
                    }
                }
            }
            else
            {
                std::copy(init.begin(), init.end(), this->data_);
            }


        };


        /// @brief Implementation: Create square matrix from dimension
        template <typename T, class Allocator>
        Matrix<T, Allocator>::Matrix(size_t n, const alloc_ptr &alloc) :
        Matrix<T, Allocator>::Matrix(Shape(n, n), alloc)
        {}

        /// @brief Implementation: Create square matrix with uniform value
        template <typename T, class Allocator>
        Matrix<T, Allocator>::Matrix(size_t n, T val, const alloc_ptr &alloc) :
        Matrix<T, Allocator>::Matrix(Shape(n, n), val, alloc) {}

        /// @brief Implementation: Copy from const data pointer
        template <typename T, class Allocator>
        Matrix<T, Allocator>::Matrix(const Shape &shape, const T *data, const alloc_ptr &alloc) :
        Matrix<T, Allocator>::Matrix(shape, alloc)
        {
            std::copy(data, data + this->data_size_(n_rows_, n_cols_), this->data_);
        };

        /// @brief Destructor implementation
        template <typename T, class Allocator>
        Matrix<T, Allocator>::~Matrix()
        {

        }

        /// @brief Copy constructor implementation
        template <typename T, class Allocator>
        Matrix<T, Allocator>::Matrix(const Matrix<T, Allocator> &other) :
        CPUTensor<T, Allocator>{other},
        n_rows_{other.n_rows_}, n_cols_{other.n_cols_}
        {

        }

        /// @brief Copy assignment operator implementation
        template <typename T, class Allocator>
        Matrix<T, Allocator> &Matrix<T, Allocator>::operator=(const Matrix<T, Allocator> &other)
        {
            if (this != &other)
            {
                CPUTensor<T, Allocator>::operator=(other);

                n_rows_ = other.n_rows_;
                n_cols_ = other.n_cols_;
            }

            return *this;
        }

        /// @brief Move constructor implementation
        template <typename T, class Allocator>
        Matrix<T, Allocator>::Matrix(Matrix<T, Allocator> &&other) :
        CPUTensor<T, Allocator>{other},
        n_rows_{other.n_rows_}, n_cols_{other.n_cols_}
        {

            other.n_rows_ = 0;
            other.n_cols_ = 0;

        }

        /// @brief Move assignment operator implementation
        template <typename T, class Allocator>
        Matrix<T, Allocator> &Matrix<T, Allocator>::operator=(Matrix<T, Allocator> &&other)
        {
            if (this != &other)
            {
                CPUTensor<T, Allocator>::operator=(std::move(other));

                n_rows_ = other.n_rows_;
                n_cols_ = other.n_cols_;

                other.n_rows_ = 0;
                other.n_cols_ = 0;
            }
            return *this;
        }

        /// @brief Implementation: Element access operator
        template <typename T, class Allocator>
        T &Matrix<T, Allocator>::operator()(size_t i, size_t j)
        {
            return this->data_[this->data_id_(i, j)];
        }

        /// @brief Implementation: Const element access operator
        template <typename T, class Allocator>
        const T &Matrix<T, Allocator>::operator()(size_t i, size_t j) const
        {
            return this->data_[this->data_id_(i, j)];
        }

        /// @brief Implementation: In-place scalar addition to all elements
        template <typename T, class Allocator>
        Matrix<T, Allocator> &Matrix<T, Allocator>::operator+=(T val)
        {
#pragma omp for
            for (size_t i = 0; i < data_size_(n_rows_, n_cols_); i++)
            {
                this->data_[i] += val;
            }

            return *this;
        }

        /// @brief Implementation: Print matrix to stdout in row-major order
        template <typename T, class Allocator>
        void Matrix<T, Allocator>::print() const
        {
            for (size_t i = 0; i < n_rows_; i++)
            {
                for (size_t j = 0; j < n_cols_; j++)
                {
                    std::cout << this->data_[this->data_id_(i, j)] << ", ";
                }
                std::cout << std::endl;
            }
        }

        /// @brief Implementation: Symmetrize matrix on CPU: M = (M + M^T)/2
        template <typename T, class Allocator>
        void Matrix<T, Allocator>::symmetrize()
        {
            assert(n_cols_ == n_rows_);

            Matrix<T, Allocator> copy = *this;
#pragma omp parallel for ordered schedule(static)
            for (size_t i = 0; i < n_cols_; i++)
            {
                for (size_t j = 0; j < n_cols_; j++)
                {
#pragma omp ordered
                    this->data_[this->data_id_(i, j)] = 0.5 * (copy(i, j) + copy(j, i));
                }
            }
        }

        /// @brief Implementation: Extract diagonal with stride stepping
        template <typename T, class Allocator>
        Vector<T, Allocator> Matrix<T, Allocator>::get_diagonal() const
        {
            size_t min_dim = std::min(n_cols_, n_rows_);
            size_t max_dim = std::max(n_cols_, n_rows_);
            Vector<T, Allocator> diag(min_dim);
            cpu::CopyVectors(diag.size(), this->data(), max_dim+1 ,diag.data(), 1);

            return diag;
        }

        /// @brief Implementation: Set diagonal with stride stepping
        template <typename T, class Allocator>
        void Matrix<T, Allocator>::set_diagonal(const Vector<T, Allocator> &diag)
        {
        size_t min_dim = std::min(n_cols_, n_rows_);
            if (diag.size() != min_dim)
            {
                throw std::runtime_error("The vector given to set diagonal doesn't correspond to the minimal dimension.");
            }
            size_t max_dim = std::max(n_cols_, n_rows_);
            cpu::CopyVectors(diag.size(), diag.data(), 1, this->data(), max_dim+1);
        }
    } // namespace cpu
} // namespace lahva

