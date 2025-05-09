#pragma once

#include "impl/tensor/allocators.hpp"
#include "impl/tensor/gpu/gputensor.hpp"
#include "impl/tensor/cpu/tensor.hpp"
#include "impl/tensor/cpu/matrix.hpp"
#include "impl/tensor/gpu/vector.hpp"
#include "impl/blas/cpu/level1.h"
#include <initializer_list>
#include <omp.h>
#include <iostream>
#include <string>
#include <fstream>
namespace lahva
{
    namespace gpu
    {
        
    
    
    template<typename T>
    class Matrix_ : public virtual GPUTensor_<T>, public virtual cpu::Matrix_<T>
    {
        public:
            virtual Shape shape() const  = 0;

            //! @param[in] i row index
            //! @param[in] j column index
            //! @return reference to Matrix element i,j
            virtual T &operator()(size_t i, size_t j) = 0;
            //! @param[in] i row index
            //! @param[in] j column index
            //! @return reference to Matrix element i,j
            virtual const T &operator()(size_t i, size_t j) const = 0;

    };


    //! @brief slim wrapper around a float or double array to allow easy acces with
    //!        two indeces using the () operator.
    //!
    //! If NDEBUG is **not** defined, range checks are performed.  d
    //! The data is stored in column-major order in a 1D array.
    //!
    template <class T, class Allocator = CudaHostAllocator<T>, class GPUAllocator = CudaDeviceAllocator<T>>
    class Matrix : public virtual GPUTensor<T, Allocator, GPUAllocator>, public virtual Matrix_<T>
    {
        using alloc_ptr = Allocator;
        using gpualloc_ptr = GPUAllocator;
    protected:
        // shape in each dimension, i.e. data_ has length n_rows_*n_cols
        size_t n_rows_ = 0;
        
        size_t n_cols_ = 0;
    
        // indicates whether the Matrix object owns the data and consequently is
        // responsible for freeing it

        inline size_t data_id_(size_t i, size_t j) const
        {
            // deactivated if NDEBUG is defined
            assert(i < n_rows_ && j < n_cols_);

            // range checks are perfomred in constructor and above (if in debug mode)
            return i + j * n_rows_;
        }

        // length of the array data_
        static size_t data_size_(size_t n_rows, size_t n_cols)
        {
            return n_rows * n_cols;
        }

        // raises an error, if the shape is not valid
        // Vector uses size_t as shape, so the check shape function has to be able
        // to deal with that
        static void check_size_(size_t, size_t);

    public:
        Matrix() : GPUTensor<T, Allocator, GPUAllocator>{} {};
        //! construct a square Matrix with dimensions n x n
        //! It is not guaranteed that the values will be initialized
        Matrix(size_t n, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());
        //! construct a square n x n Matrix initialized with value val
        Matrix(size_t n, T val, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());

        //! construct a Matrix with dimensions shape.first x shape.second
        //! It is not guaranteed that the values will be initialized!
        Matrix(const Shape &shape, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());

        Matrix(const Shape &shape, std::initializer_list<T> init, bool row_major = false, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());

        template <typename U>
        Matrix(const Shape &shape, const CudaRuntime &cudart, const U &alloc)
            : Matrix<T,Allocator,GPUAllocator>(shape, cudart)
            { 
                this->gpualloc_ = alloc;
            };

        Matrix(const Shape &shape, const CudaRuntime &cudart, const gpualloc_ptr &gpualloc = GPUAllocator());
        //! construct a Matrix of shape with initial values val
        Matrix(const Shape &shape, T val, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());
        //! construct a Matrix by giving ownership of the raw data
        //! note: the data needs to be in the format defined by data_id_()
        //! @param take_ownership if false, the data will not be freed, when the
        //!                       Matrix object is destructed. Use with care!
        Matrix(const Shape &shape, T *data, bool take_ownership = true, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());
        Matrix(const Shape &shape, const T *data, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());

        Matrix(const Matrix &);
        Matrix(Matrix &&);
        Matrix &operator=(const Matrix &);
        Matrix &operator=(Matrix &&);
        virtual ~Matrix();

        //! @param[in] i row index
        //! @param[in] j column index
        //! @return reference to Matrix element i,j
        T &operator()(size_t i, size_t j);
        //! @param[in] i row index
        //! @param[in] j column index
        //! @return reference to Matrix element i,j
        const T &operator()(size_t i, size_t j) const;

        //! @brief in-place, scalar addition
        Matrix &operator+=(T val);

        //! @return number of rows/columns of the Matrix
        Shape shape() const { return Shape{n_rows_, n_cols_}; }

        //! prints the Matrix as string
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
                    os << this->data_[data_id_(i, j)] << ", ";
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

        Vector<T, Allocator, GPUAllocator> get_diagonal() const;

        void set_diagonal(const Vector<T, Allocator, GPUAllocator> &diag);

        void symmetrize();

        bool ownsData() { return this->is_owner_; };
    };

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

    template <typename T, class Allocator, class GPUAllocator>
    Matrix<T, Allocator, GPUAllocator>::Matrix(const Shape &shape, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) : 
    GPUTensor<T, Allocator, GPUAllocator>{shape.first*shape.second, alloc, gpualloc}, 
    n_rows_{shape.first}, n_cols_{shape.second}
    {
        check_size_(shape.first, shape.second);
    }

    template <typename T, class Allocator, class GPUAllocator>
    Matrix<T, Allocator, GPUAllocator>::Matrix(const Shape &shape, std::initializer_list<T> init, bool row_major, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) : 
    Matrix(shape, alloc, gpualloc)
    {
        assert(init.size() == this->count_);
        if (row_major)
        {
            #pragma omp parallel for ordered schedule(static)
            for (size_t i = 0; i < this->n_rows_; i++)
            {
                for (size_t j = 0; j < this->n_cols_; j++)
                {
                    this->data_[data_id_(i, j)] = init.begin()[j + i * n_rows_];
                }
            }
        }
        else
        {
            std::copy(init.begin(), init.end(), this->data_);
        }
    };

    template <typename T, class Allocator, class GPUAllocator>
    Matrix<T, Allocator, GPUAllocator>::Matrix(const Shape &shape, const CudaRuntime &cudart, const gpualloc_ptr &gpualloc) : 
    GPUTensor<T, Allocator, GPUAllocator>{gpualloc},
    n_rows_{shape.first}, n_cols_{shape.second}
    {
        check_size_(shape.first, shape.second);
        this->count_ = n_rows_*n_cols_; 
        this->is_owner_ = false;
        this->is_on_device_ = true;
        this->gpu_buffer = true;
    }

    template <typename T, class Allocator, class GPUAllocator>
    Matrix<T, Allocator, GPUAllocator>::Matrix(const Shape &shape, T *data, bool take_ownership, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) : 
    GPUTensor<T, Allocator, GPUAllocator>{alloc, gpualloc}, n_rows_{shape.first}, n_cols_{shape.second}
    {
        this->data_ = data;
        this->count_ = n_rows_*n_cols_;
        this->is_owner_ = take_ownership;
    }


    template <typename T, class Allocator, class GPUAllocator>
    Matrix<T, Allocator, GPUAllocator>::Matrix(const Shape &shape, T val, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) : 
    Matrix(shape, alloc, gpualloc)
    {
        std::fill(this->data_, this->data_ + data_size_(n_rows_, n_cols_), val);
    }

    template <typename T, class Allocator, class GPUAllocator>
    Matrix<T, Allocator, GPUAllocator>::Matrix(size_t n, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) : 
    Matrix<T, Allocator, GPUAllocator>::Matrix(Shape(n, n), alloc, gpualloc) 
    {}

    template <typename T, class Allocator, class GPUAllocator>
    Matrix<T, Allocator, GPUAllocator>::Matrix(size_t n, T val, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) : 
    Matrix<T, Allocator, GPUAllocator>::Matrix(Shape(n, n), val, alloc, gpualloc) {}

    template <typename T, class Allocator, class GPUAllocator>
    Matrix<T, Allocator, GPUAllocator>::Matrix(const Shape &shape, const T *data, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) : 
    Matrix<T, Allocator, GPUAllocator>::Matrix(shape, alloc, gpualloc)
    {
        std::copy(data, data + data_size_(n_rows_, n_cols_), this->data_);
    };

    template <typename T, class Allocator, class GPUAllocator>
    Matrix<T, Allocator, GPUAllocator>::~Matrix()
    {
        
    }

    // copy operations
    template <typename T, class Allocator, class GPUAllocator>
    Matrix<T, Allocator, GPUAllocator>::Matrix(const Matrix<T, Allocator, GPUAllocator> &other) : 
    GPUTensor<T, Allocator, GPUAllocator>{other}, 
    n_rows_{other.n_rows_}, n_cols_{other.n_cols_}  
    {
        
    }

    template <typename T, class Allocator, class GPUAllocator>
    Matrix<T, Allocator, GPUAllocator> &Matrix<T, Allocator, GPUAllocator>::operator=(const Matrix<T, Allocator, GPUAllocator> &other)
    {
        if (this != &other)
        {
            GPUTensor<T, Allocator, GPUAllocator>::operator=(other);

            n_rows_ = other.n_rows_;
            n_cols_ = other.n_cols_;
        }

        return *this;
    }

    // move operations
    template <typename T, class Allocator, class GPUAllocator>
    Matrix<T, Allocator, GPUAllocator>::Matrix(Matrix<T, Allocator, GPUAllocator> &&other) : 
    GPUTensor<T, Allocator, GPUAllocator>{other},
    n_rows_{other.n_rows_}, n_cols_{other.n_cols_}
    {
   
        other.n_rows_ = 0;
        other.n_cols_ = 0;

    }

    template <typename T, class Allocator, class GPUAllocator>
    Matrix<T, Allocator, GPUAllocator> &Matrix<T, Allocator, GPUAllocator>::operator=(Matrix<T, Allocator, GPUAllocator> &&other)
    {
        if (this != &other)
        {
            GPUTensor<T, Allocator, GPUAllocator>::operator=(std::move(other));
            
            n_rows_ = other.n_rows_;
            n_cols_ = other.n_cols_;
            
            other.n_rows_ = 0;
            other.n_cols_ = 0;
        }
        return *this;
    }

    template <typename T, class Allocator, class GPUAllocator>
    T &Matrix<T, Allocator, GPUAllocator>::operator()(size_t i, size_t j)
    {
        return this->data_[data_id_(i, j)];
    }

    template <typename T, class Allocator, class GPUAllocator>
    const T &Matrix<T, Allocator, GPUAllocator>::operator()(size_t i, size_t j) const
    {
        return this->data_[data_id_(i, j)];
    }

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

    template <typename T, class Allocator, class GPUAllocator>
    void Matrix<T, Allocator, GPUAllocator>::print() const
    {
        for (size_t i = 0; i < n_rows_; i++)
        {
            for (size_t j = 0; j < n_cols_; j++)
            {
                std::cout << this->data_[data_id_(i, j)] << ", ";
            }
            std::cout << std::endl;
        }
    }

    //! @param A Matrix to symmetrize
    //! @return (A+A^T)/2
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
                this->data_[data_id_(i, j)] = 0.5 * (copy(i, j) + copy(j, i));
            }
        }
    }

    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator> Matrix<T, Allocator, GPUAllocator>::get_diagonal() const
    {
        size_t min_dim = std::min(n_cols_, n_rows_);
        size_t max_dim = std::max(n_cols_, n_rows_);
        Vector<T, Allocator, GPUAllocator> diag(min_dim);
        cpu::CopyVectors(diag.size(), this->data(), max_dim+1 ,diag.data(), 1);

        return diag;
    }

    template <typename T, class Allocator, class GPUAllocator>
    void Matrix<T, Allocator, GPUAllocator>::set_diagonal(const Vector<T, Allocator, GPUAllocator> &diag)
    {
        size_t min_dim = std::min(n_cols_, n_rows_);
        if (diag.size() != min_dim)
        {
            throw std::runtime_error("The vector given to set diagonal doesn't correspond to the minimal dimension.");
        }
        size_t max_dim = std::max(n_cols_, n_rows_);
        cpu::CopyVectors(diag.size(), diag.data(), 1, this->data(), max_dim+1);
    }
    } // namespace gpu
} // namespace lahva
