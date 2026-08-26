/// @file matrix.hpp
/// @brief GPU-based matrix tensor implementation.
///
/// Provides the Matrix_ class for 2-dimensional tensor storage and operations on GPU.
/// Supports dense matrix operations with cuBLAS compatibility, column-major storage,
/// and mixed-precision computations with CPU matrices.

#pragma once

#include "impl/blas/cpu/level1.h"
#include "impl/tensor/allocators.hpp"
#include "impl/tensor/cpu/tensor.hpp"
#include "impl/tensor/gpu/vector.hpp"

namespace lahva
{
    namespace gpu
    {

        /// @brief Abstract GPU matrix base class
        /// @tparam T data type for matrix elements
        template <typename T>
        class Matrix_ : public virtual Tensor_<T>, public virtual cpu::Matrix_<T>
        {
        public:
            /// @brief Get matrix shape
            /// @return Shape object with dimensions (rows, cols)
            virtual Shape shape() const = 0;

            /// @brief Element access operator
            /// @param[in] i row index
            /// @param[in] j column index
            /// @return reference to matrix element at (i,j)
            virtual T &operator()(size_t i, size_t j) = 0;

            /// @brief Const element access operator
            /// @param[in] i row index
            /// @param[in] j column index
            /// @return const reference to matrix element at (i,j)
            virtual const T &operator()(size_t i, size_t j) const = 0;
        };

        /// @brief GPU kernel wrapper: symmetrize matrix in-place
        /// @tparam T data type for matrix elements
        /// @param[in] cudart CUDA runtime instance
        /// @param[in,out] m matrix to symmetrize, result is (M + M^T)/2
        template <typename T>
        void SymmetrizeMatrix(const CudaRuntime &cudart, Matrix_<T> &m);

        /// @brief GPU kernel wrapper: extract diagonal from matrix
        /// @tparam T data type for matrix elements
        /// @param[in] cudart CUDA runtime instance
        /// @param[in] m source matrix
        /// @param[out] diag output vector containing diagonal elements
        template <typename T>
        void GetDiagonal(const CudaRuntime &cudart, const Matrix_<T> &m, Tensor_<T> &diag);

        /// @brief GPU kernel wrapper: set diagonal of matrix
        /// @tparam T data type for matrix elements
        /// @param[in] cudart CUDA runtime instance
        /// @param[in] diag vector containing values to set on diagonal
        /// @param[in,out] m target matrix
        template <typename T>
        void SetDiagonal(const CudaRuntime &cudart, const Tensor_<T> &diag, Matrix_<T> &m);

        /// @brief GPU-based 2-dimensional matrix tensor with dual memory management
        ///
        /// If NDEBUG is **not** defined, range checks are performed.
        /// The data is stored in column-major order in a 1D array.
        /// Supports both host (CPU) and device (GPU) memory with automatic transfers.
        ///
        /// @tparam T data type for matrix elements
        /// @tparam Allocator host (CPU) memory allocator type (default: CudaHostAllocator)
        /// @tparam GPUAllocator device (GPU) memory allocator type (default: CudaDeviceAsyncAllocator)
        template <class T, class Allocator = CudaHostAllocator<T>, class GPUAllocator = CudaDeviceAsyncAllocator<T>>
        class Matrix : public Tensor<T, Allocator, GPUAllocator>, public virtual Matrix_<T>
        {
            using alloc_ptr = Allocator;
            using gpualloc_ptr = GPUAllocator;

        protected:
            /// @brief Number of rows in the matrix
            size_t n_rows_ = 0;

            /// @brief Number of columns in the matrix
            size_t n_cols_ = 0;

            /// @brief Calculate linear index for matrix element in column-major storage
            /// @param[in] i row index
            /// @param[in] j column index
            /// @return linear index in 1D storage array
            /// @note Range checks deactivated if NDEBUG is defined
            inline size_t data_id_(size_t i, size_t j) const
            {
                assert(i < n_rows_ && j < n_cols_);
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
            /// @brief Default constructor for GPU matrix
            Matrix() : Tensor<T, Allocator, GPUAllocator>{} {};

            /// @brief Construct a square n x n GPU matrix
            /// @param[in] n dimension of square matrix (n x n)
            /// @param[in] alloc host (CPU) memory allocator
            /// @param[in] gpualloc device (GPU) memory allocator
            /// @note values are not guaranteed to be initialized
            Matrix(size_t n, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());

            /// @brief Construct a square n x n GPU matrix initialized with a value
            /// @param[in] n dimension of square matrix (n x n)
            /// @param[in] val initial value for all elements
            /// @param[in] alloc host (CPU) memory allocator
            /// @param[in] gpualloc device (GPU) memory allocator
            Matrix(size_t n, T val, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());

            /// @brief Construct a GPU matrix with specified shape
            /// @param[in] shape matrix dimensions (rows, cols)
            /// @param[in] alloc host (CPU) memory allocator
            /// @param[in] gpualloc device (GPU) memory allocator
            /// @note values are not guaranteed to be initialized
            Matrix(const Shape &shape, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());

            /// @brief Construct a GPU matrix from initializer list
            /// @param[in] shape matrix dimensions (rows, cols)
            /// @param[in] init initializer list with matrix values
            /// @param[in] row_major if true, interpret init as row-major; else column-major
            /// @param[in] alloc host (CPU) memory allocator
            /// @param[in] gpualloc device (GPU) memory allocator
            Matrix(const Shape &shape, std::initializer_list<T> init, bool row_major = false, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());
            template <typename U>
            Matrix(const Shape &shape, const CudaRuntime &cudart, const U &alloc)
                : Matrix<T, Allocator, GPUAllocator>(shape, cudart, static_cast<GPUAllocator>(alloc)) {
                  };

            /// @brief Construct a GPU matrix using CUDA runtime for GPU memory
            /// @param[in] shape matrix dimensions (rows, cols)
            /// @param[in] cudart CUDA runtime instance for GPU operations
            /// @param[in] gpualloc device (GPU) memory allocator
            Matrix(const Shape &shape, const CudaRuntime &cudart, const gpualloc_ptr &gpualloc = GPUAllocator());

            /// @brief Construct a GPU matrix with specified shape initialized with a value
            /// @param[in] shape matrix dimensions (rows, cols)
            /// @param[in] val initial value for all elements
            /// @param[in] alloc host (CPU) memory allocator
            /// @param[in] gpualloc device (GPU) memory allocator
            Matrix(const Shape &shape, T val, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());

            /// @brief Construct a GPU matrix from raw data pointer
            /// @param[in] shape matrix dimensions (rows, cols)
            /// @param[in] data pointer to matrix data in column-major format
            /// @param[in] take_ownership if false, data is not freed when matrix is destructed
            /// @param[in] alloc host (CPU) memory allocator
            /// @param[in] gpualloc device (GPU) memory allocator
            /// @note pointer must hold the data in column-major format
            Matrix(const Shape &shape, T *data, bool take_ownership = true, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());

            /// @brief Construct a GPU matrix from const data pointer
            /// @param[in] shape matrix dimensions (rows, cols)
            /// @param[in] data pointer to constant matrix data to copy
            /// @param[in] alloc host (CPU) memory allocator
            /// @param[in] gpualloc device (GPU) memory allocator
            /// @note pointer must hold the data in column-major format
            Matrix(const Shape &shape, const T *data, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());

            /// @brief Copy constructor for GPU matrix
            /// @param[in] other source matrix to copy
            Matrix(const Matrix &);

            /// @brief Move constructor for GPU matrix
            /// @param[in] other source matrix to move from
            Matrix(Matrix &&);

            /// @brief Copy assignment operator
            /// @param[in] other source matrix
            /// @return reference to this matrix
            Matrix &operator=(const Matrix &);

            /// @brief Move assignment operator
            /// @param[in] other source matrix to move from
            /// @return reference to this matrix
            Matrix &operator=(Matrix &&);

            /// @brief Destructor for GPU matrix
            virtual ~Matrix();

            /// @brief Element access operator
            /// @param[in] i row index
            /// @param[in] j column index
            /// @return reference to matrix element at (i,j)
            T &operator()(size_t i, size_t j);

            /// @brief Const element access operator
            /// @param[in] i row index
            /// @param[in] j column index
            /// @return const reference to matrix element at (i,j)
            const T &operator()(size_t i, size_t j) const;

            /// @brief In-place scalar addition
            /// @param[in] val value to add to all elements
            /// @return reference to this matrix
            Matrix &operator+=(T val);

            /// @brief Get matrix shape
            /// @return Shape object with (rows, columns)
            Shape shape() const { return Shape{n_rows_, n_cols_}; }

            /// @brief Print matrix to standard output
            void print() const;

            /// @brief Print matrix to file
            /// @param[in] file output filename
            void print(const char *file) const
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

            /// @brief Extract diagonal from matrix (CPU operation)
            /// @return vector containing diagonal elements
            Vector<T, Allocator, GPUAllocator> get_diagonal() const;

            /// @brief Set diagonal of matrix (CPU operation)
            /// @param[in] diag vector containing values to set on diagonal
            void set_diagonal(const Vector<T, Allocator, GPUAllocator> &diag);

            /// @brief Symmetrize matrix in-place: M = (M + M^T)/2 (CPU operation)
            void symmetrize();

            /// @brief Extract diagonal from matrix (GPU operation)
            /// @param[in] cudart CUDA runtime instance
            /// @return vector containing diagonal elements
            Vector<T, Allocator, GPUAllocator> get_diagonal(const CudaRuntime &cudart) const;

            /// @brief Set diagonal of matrix (GPU operation)
            /// @param[in] cudart CUDA runtime instance
            /// @param[in] diag vector containing values to set on diagonal
            void set_diagonal(const CudaRuntime &cudart, const Vector<T, Allocator, GPUAllocator> &diag);

            /// @brief Symmetrize matrix in-place (GPU operation) M = (M + M^T)/2
            /// @param[in] cudart CUDA runtime instance
            void symmetrize(const CudaRuntime &cudart);

            /// @brief Check if matrix owns its data
            /// @return true if this matrix is responsible for freeing data
            bool ownsData() { return this->is_owner_; };
        };

        /// @brief Implementation: Validate matrix dimensions against SIZE_MAX
        template <typename T, class Allocator, class GPUAllocator>
        void Matrix<T, Allocator, GPUAllocator>::check_size_(size_t n_rows,
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
        template <typename T, class Allocator, class GPUAllocator>
        Matrix<T, Allocator, GPUAllocator>::Matrix(const Shape &shape, const alloc_ptr &alloc, const gpualloc_ptr &gpualloc) : Tensor<T, Allocator, GPUAllocator>{shape.first * shape.second, alloc, gpualloc},
                                                                                                                               n_rows_{shape.first}, n_cols_{shape.second}
        {
            this->check_size_(shape.first, shape.second);
        }

        /// @brief Implementation: Initialize from initializer list with optional row/column major conversion
        template <typename T, class Allocator, class GPUAllocator>
        Matrix<T, Allocator, GPUAllocator>::Matrix(const Shape &shape, std::initializer_list<T> init, bool row_major, const alloc_ptr &alloc, const gpualloc_ptr &gpualloc) : Matrix(shape, alloc, gpualloc)
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

        /// @brief Implementation: Create GPU-only matrix using CUDA runtime
        /// Data is not allocated on host, only on GPU device
        template <typename T, class Allocator, class GPUAllocator>
        Matrix<T, Allocator, GPUAllocator>::Matrix(const Shape &shape, const CudaRuntime &cudart, const gpualloc_ptr &gpualloc) : Tensor<T, Allocator, GPUAllocator>{gpualloc},
                                                                                                                                  n_rows_{shape.first}, n_cols_{shape.second}
        {
            check_size_(shape.first, shape.second);
            this->count_ = n_rows_ * n_cols_;
            this->is_owner_ = false;
            this->gpu_buffer = true;
        }

        /// @brief Implementation: Wrap existing data pointer with optional ownership
        template <typename T, class Allocator, class GPUAllocator>
        Matrix<T, Allocator, GPUAllocator>::Matrix(const Shape &shape, T *data, bool take_ownership, const alloc_ptr &alloc, const gpualloc_ptr &gpualloc) : Tensor<T, Allocator, GPUAllocator>{alloc, gpualloc}, n_rows_{shape.first}, n_cols_{shape.second}
        {
            this->data_ = data;
            this->count_ = n_rows_ * n_cols_;
            this->is_owner_ = take_ownership;
        }

        /// @brief Implementation: Allocate and fill matrix with uniform value
        template <typename T, class Allocator, class GPUAllocator>
        Matrix<T, Allocator, GPUAllocator>::Matrix(const Shape &shape, T val, const alloc_ptr &alloc, const gpualloc_ptr &gpualloc) : Matrix(shape, alloc, gpualloc)
        {
            std::fill(this->data_, this->data_ + data_size_(n_rows_, n_cols_), val);
        }

        /// @brief Implementation: Create square matrix from dimension
        template <typename T, class Allocator, class GPUAllocator>
        Matrix<T, Allocator, GPUAllocator>::Matrix(size_t n, const alloc_ptr &alloc, const gpualloc_ptr &gpualloc) : Matrix<T, Allocator, GPUAllocator>::Matrix(Shape(n, n), alloc, gpualloc)
        {
        }

        /// @brief Implementation: Create square matrix with uniform value
        template <typename T, class Allocator, class GPUAllocator>
        Matrix<T, Allocator, GPUAllocator>::Matrix(size_t n, T val, const alloc_ptr &alloc, const gpualloc_ptr &gpualloc) : Matrix<T, Allocator, GPUAllocator>::Matrix(Shape(n, n), val, alloc, gpualloc) {}

        /// @brief Implementation: Copy from const data pointer
        template <typename T, class Allocator, class GPUAllocator>
        Matrix<T, Allocator, GPUAllocator>::Matrix(const Shape &shape, const T *data, const alloc_ptr &alloc, const gpualloc_ptr &gpualloc) : Matrix<T, Allocator, GPUAllocator>::Matrix(shape, alloc, gpualloc)
        {
            std::copy(data, data + this->data_size_(n_rows_, n_cols_), this->data_);
        };

        /// @brief Destructor implementation - base class handles memory cleanup
        template <typename T, class Allocator, class GPUAllocator>
        Matrix<T, Allocator, GPUAllocator>::~Matrix()
        {
        }

        /// @brief Implementation: Copy constructor
        template <typename T, class Allocator, class GPUAllocator>
        Matrix<T, Allocator, GPUAllocator>::Matrix(const Matrix<T, Allocator, GPUAllocator> &other) : Tensor<T, Allocator, GPUAllocator>{other},
                                                                                                      n_rows_{other.n_rows_}, n_cols_{other.n_cols_}
        {
        }

        /// @brief Implementation: Copy assignment operator
        template <typename T, class Allocator, class GPUAllocator>
        Matrix<T, Allocator, GPUAllocator> &Matrix<T, Allocator, GPUAllocator>::operator=(const Matrix<T, Allocator, GPUAllocator> &other)
        {
            if (this != &other)
            {
                Tensor<T, Allocator, GPUAllocator>::operator=(other);

                n_rows_ = other.n_rows_;
                n_cols_ = other.n_cols_;
            }

            return *this;
        }

        /// @brief Implementation: Move constructor
        template <typename T, class Allocator, class GPUAllocator>
        Matrix<T, Allocator, GPUAllocator>::Matrix(Matrix<T, Allocator, GPUAllocator> &&other) : Tensor<T, Allocator, GPUAllocator>{other},
                                                                                                 n_rows_{other.n_rows_}, n_cols_{other.n_cols_}
        {

            other.n_rows_ = 0;
            other.n_cols_ = 0;
        }

        /// @brief Implementation: Move assignment operator
        template <typename T, class Allocator, class GPUAllocator>
        Matrix<T, Allocator, GPUAllocator> &Matrix<T, Allocator, GPUAllocator>::operator=(Matrix<T, Allocator, GPUAllocator> &&other)
        {
            if (this != &other)
            {
                Tensor<T, Allocator, GPUAllocator>::operator=(std::move(other));

                n_rows_ = other.n_rows_;
                n_cols_ = other.n_cols_;

                other.n_rows_ = 0;
                other.n_cols_ = 0;
            }
            return *this;
        }

        /// @brief Implementation: Element access operator
        template <typename T, class Allocator, class GPUAllocator>
        T &Matrix<T, Allocator, GPUAllocator>::operator()(size_t i, size_t j)
        {
            return this->data_[this->data_id_(i, j)];
        }

        /// @brief Implementation: Const element access operator
        template <typename T, class Allocator, class GPUAllocator>
        const T &Matrix<T, Allocator, GPUAllocator>::operator()(size_t i, size_t j) const
        {
            return this->data_[this->data_id_(i, j)];
        }

        /// @brief Implementation: In-place scalar addition to all elements
        template <typename T, class Allocator, class GPUAllocator>
        Matrix<T, Allocator, GPUAllocator> &Matrix<T, Allocator, GPUAllocator>::operator+=(T val)
        {
#pragma omp for
            for (size_t i = 0; i < data_size_(n_rows_, n_cols_); i++)
            {
                this->data_[i] += val;
            }

            return *this;
        }

        /// @brief Implementation: Print matrix to stdout in row-major order
        template <typename T, class Allocator, class GPUAllocator>
        void Matrix<T, Allocator, GPUAllocator>::print() const
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
        template <typename T, class Allocator, class GPUAllocator>
        void Matrix<T, Allocator, GPUAllocator>::symmetrize()
        {
            assert(n_cols_ == n_rows_);

            Matrix<T, Allocator, GPUAllocator> copy = *this;
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

        /// @brief Implementation: Symmetrize matrix on GPU using CUDA kernel
        template <typename T, class Allocator, class GPUAllocator>
        void Matrix<T, Allocator, GPUAllocator>::symmetrize(const CudaRuntime &cudart)
        {
            assert(n_cols_ == n_rows_);
            gpu::SymmetrizeMatrix<T>(cudart, *this);
        }

        /// @brief Implementation: Extract diagonal on CPU with stride stepping
        template <typename T, class Allocator, class GPUAllocator>
        Vector<T, Allocator, GPUAllocator> Matrix<T, Allocator, GPUAllocator>::get_diagonal() const
        {
            size_t min_dim = std::min(n_cols_, n_rows_);
            size_t max_dim = std::max(n_cols_, n_rows_);
            Vector<T, Allocator, GPUAllocator> diag(min_dim);
            cpu::CopyVectors(diag.size(), this->data(), max_dim + 1, diag.data(), 1);

            return diag;
        }

        /// @brief Implementation: Extract diagonal on GPU using CUDA kernel
        template <typename T, class Allocator, class GPUAllocator>
        Vector<T, Allocator, GPUAllocator> Matrix<T, Allocator, GPUAllocator>::get_diagonal(const CudaRuntime &cudart) const
        {
            size_t min_dim = std::min(n_cols_, n_rows_);
            Vector<T, Allocator, GPUAllocator> diag(min_dim);
            gpu::GetDiagonal<T>(cudart, *this, diag);
            return diag;
        }

        /// @brief Implementation: Set diagonal on CPU with stride stepping
        template <typename T, class Allocator, class GPUAllocator>
        void Matrix<T, Allocator, GPUAllocator>::set_diagonal(const Vector<T, Allocator, GPUAllocator> &diag)
        {
            size_t min_dim = std::min(n_cols_, n_rows_);
            if (diag.size() != min_dim)
            {
                throw std::runtime_error("The vector given to set diagonal doesn't correspond to the minimal dimension.");
            }
            size_t max_dim = std::max(n_cols_, n_rows_);
            cpu::CopyVectors(diag.size(), diag.data(), 1, this->data(), max_dim + 1);
        }

        /// @brief Implementation: Set diagonal on GPU using CUDA kernel
        template <typename T, class Allocator, class GPUAllocator>
        void Matrix<T, Allocator, GPUAllocator>::set_diagonal(const CudaRuntime &cudart, const Vector<T, Allocator, GPUAllocator> &diag)
        {
            size_t min_dim = std::min(n_cols_, n_rows_);
            if (diag.size() != min_dim)
            {
                throw std::runtime_error("The vector given to set diagonal doesn't correspond to the minimal dimension.");
            }
            gpu::SetDiagonal<T>(cudart, diag, *this);
        }
    } // namespace gpu

#ifdef _CUDA
    namespace cpu
    {
        // We need to have the get and set Routiunes for the CPU data of a GPU Tensor as well

        /// @brief Get the diagonal of a matrix
        /// @tparam T data type for matrix elements
        /// @tparam U CPU Allocator type of the matrix
        /// @tparam V GPU Allocator type of the matrix
        /// @param[in] cpurt CPU Runtime instance
        /// @param[in] m source matrix
        /// @param[out] diag output vector containing diagonal elements
        template <typename T, typename U, typename V>
        void GetDiagonal(const CPURuntime &cpurt, const gpu::Matrix<T, U, V> &mat, gpu::Vector<T, U, V> &vec)
        {
            size_t max_dim = std::max(mat.shape().first, mat.shape().second);
            cpu::CopyVectors(vec.size(), mat->data(), max_dim + 1, vec.data(), 1);
        };

        /// @brief Get the diagonal of a matrix
        /// @tparam T data type for matrix elements
        /// @tparam U CPU Allocator type of the matrix
        /// @tparam V GPU Allocator type of the matrix
        /// @param[in] m source matrix
        /// @param[out] diag output vector containing diagonal elements
        template <typename T, typename U, typename V>
        void GetDiagonal(const gpu::Matrix<T, U, V> &mat, gpu::Vector<T, U, V> &vec)
        {
            size_t max_dim = std::max(mat.shape().first, mat.shape().second);
            cpu::CopyVectors(vec.size(), mat.data(), max_dim + 1, vec.data(), 1);
        };

        /// @brief Set the diagonal of a matrix
        /// @tparam T data type for matrix elements
        /// @param cpurt CPU Runtime instance
        /// @param[in] vec vector containing values to set on diagonal
        /// @param[in,out] m target matrix
        template <typename T, typename U, typename V>
        void SetDiagonal(const CPURuntime &cpurt, const gpu::Vector<T, U, V> &vec, gpu::Matrix<T, U, V> &m)
        {
            m.set_diagonal(vec);
        };

    } // namespace cpu
#endif
} // namespace lahva
