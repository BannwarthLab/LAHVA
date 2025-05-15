#pragma once
#include "runtime.hpp"
#include "impl/tensor/gpu/gputensor.hpp"
#include "impl/tensor/cpu/lowtrimatrix.hpp"
#include <omp.h>

namespace lahva
{
    namespace gpu
    {
        

    template<typename T>
    class LowTriMatrix_ : virtual public GPUTensor_<T>, virtual public cpu::LowTriMatrix_<T>
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

    template <class T, class Allocator = CudaHostAllocator<T>, class GPUAllocator = CudaDeviceAllocator<T>>
    class LowTriMatrix : virtual public GPUTensor<T, Allocator, GPUAllocator>, virtual public LowTriMatrix_<T>
    {
        using alloc_ptr = CPUAllocator<T>;
        using gpualloc_ptr = GPUAllocator_<T>;   
    protected:
        // shape in each dimension, i.e. data_ has length n_rows_^2
        size_t n_;

        inline size_t data_id_(size_t i, size_t j) const
        {
            // deactivated if NDEBUG is defined
            assert(i <= n_ && j <= n_);
            // row index is always greater than column index in lower triangle
            assert(i >= j);

            // range checks are perfomred in constructor and above (if in debug mode)
            return (n_ * (j) - (j - 1) * (j) / 2) + (i - j);
        }

        // length of the array data_
        static size_t data_size_(size_t n)
        {
            return (n * n + n) / 2;
        }

        // raises an error, if the shape is not valid
        // Vector uses ulong as shape, so the check shape function has to be able
        // to deal with that
        static void check_size_(long unsigned int shape);

    public:
        //! construct an n x n matrix
        //! It is not guaranteed that the values will be initialized
        LowTriMatrix(size_t n,const alloc_ptr &cpualloc = Allocator(), const gpualloc_ptr &alloc = GPUAllocator());
        //! construct an n x n matrix initialized with value val
        LowTriMatrix(size_t n, T val,const alloc_ptr &cpualloc = Allocator(), const gpualloc_ptr &alloc = GPUAllocator());
        //! give dimension n and data as a Vector conatining the values of the lower traingular Matrix
        LowTriMatrix(size_t n, const Vector<T, Allocator, GPUAllocator> &data, const alloc_ptr &cpualloc = Allocator(), const gpualloc_ptr &alloc = GPUAllocator());
        //! create a matrix with shape (shape must be square)
        //! It is not guaranteed that the values will be initialized
        LowTriMatrix(const Shape &shape, const alloc_ptr &cpualloc = Allocator(), const gpualloc_ptr &alloc = GPUAllocator());
        //! create a non-initialized matrix with shape (shape must be square) and
        //! initialize all values to val
        LowTriMatrix(const Shape &shape, T val, const alloc_ptr &cpualloc = Allocator(), const gpualloc_ptr &alloc = GPUAllocator());
        //! construct a matrix by giving ownership of the raw data
        //! note: the data needs to be in the format defined by data_id_()
        //! @param n number of columns/rows of the matrix
        LowTriMatrix(size_t n, T *data, bool take_ownership = true,const alloc_ptr &cpualloc = Allocator(), const gpualloc_ptr &alloc = GPUAllocator());
        LowTriMatrix(const LowTriMatrix &);
        LowTriMatrix(LowTriMatrix &&);
        LowTriMatrix &operator=(const LowTriMatrix &);
        LowTriMatrix &operator=(LowTriMatrix &&);
        virtual ~LowTriMatrix();

        //! Provides element acces.
        //! If NDEBUG is **not** defined, range checks are performed and it will be
        //! checked that the first index is greater than or equal to the second.
        //!
        //! @param[in] i row index
        //! @param[in] j column index
        //! @return reference to matrix element i,j
        T &operator()(size_t i, size_t j);
        //! Provides element acces.
        //! If NDEBUG is **not** defined, range checks are performed and it will be
        //! checked that the first index is greater than or equal to the second.
        //! @param[in] i row index
        //! @param[in] j column index
        //! @return reference to matrix element i,j
        const T &operator()(size_t i, size_t j) const;

        //! @return number of rows/columns of the matrix
        Shape shape() const { return Shape(n_, n_); }

        void print() const;

        Vector<T, Allocator, GPUAllocator> get_diagonal() const;

        void set_diagonal(Vector<T, Allocator, GPUAllocator> &diag);
    };

    template <typename T, class Allocator, class GPUAllocator>
    void LowTriMatrix<T, Allocator, GPUAllocator>::check_size_(long unsigned int n)
    {
        if (n > SIZE_MAX || n > std::sqrt(SIZE_MAX))
        {
            throw std::out_of_range("Vector exceeds maximum LowTriMatrix size.");
        }
    }

    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator>::LowTriMatrix(size_t size, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) : 
    GPUTensor<T, Allocator, GPUAllocator>{data_size_(size), alloc, gpualloc}, n_{size}                                    
    {
        check_size_(size);
    }

    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator>::LowTriMatrix(size_t size, T val, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) : 
    LowTriMatrix(size, alloc, gpualloc)
    {
        check_size_(size);

        std::fill(this->data_, this->data_ + data_size_(size), val);
    }

    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator>::LowTriMatrix(const Shape &shape, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) : 
    LowTriMatrix(shape.first, alloc, gpualloc)
    {
        if (shape.first != shape.second)
        {
            throw std::logic_error("shape.first must be equal to shape.second.");
        }
    }

    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator>::LowTriMatrix(const Shape &shape, T val, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) : 
    LowTriMatrix{shape.first, val, alloc, gpualloc}
    {
        if (shape.first != shape.second)
        {
            throw std::logic_error("shape.first must be equal to shape.second.");
        }
    }


    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator>::LowTriMatrix(size_t n, const Vector<T, Allocator, GPUAllocator> &data, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) : 
    LowTriMatrix{n, alloc, gpualloc}
    {
        check_size_(data.size());
        #pragma omp for
        for (size_t i = 0; i < data_size_(n_); i++)
        {
            this->data_[i] = data[i];
        }
    }

    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator>::LowTriMatrix(size_t n, T *data, bool take_ownership, const alloc_ptr &cpualloc, const gpualloc_ptr &gpualloc)
    :  GPUTensor<T, Allocator, GPUAllocator>{cpualloc, gpualloc}, n_{n}
    {
        this->data_ = data;
        this->count_ = data_size_(n);
        this->is_owner_ = take_ownership;
    
    }


    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator>::~LowTriMatrix()
    {
    
    }

    // copy operations
    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator>::LowTriMatrix(const LowTriMatrix<T, Allocator, GPUAllocator> &other) : 
    n_{other.n_}, 
    GPUTensor<T, Allocator, GPUAllocator>(static_cast<const GPUTensor<T, Allocator, GPUAllocator>&>(other))
    {
        
    }

    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator> &LowTriMatrix<T, Allocator, GPUAllocator>::operator=(const LowTriMatrix<T, Allocator, GPUAllocator> &other)
    {
        if (this != &other)
        {
            GPUTensor<T, Allocator, GPUAllocator>::operator=(static_cast<const GPUTensor<T, Allocator, GPUAllocator>&>(other));
            n_ = other.size();
        }
        return *this;
    }

    // move operations
    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator>::LowTriMatrix(LowTriMatrix<T, Allocator, GPUAllocator> &&other) : 
    n_{other.n_}, GPUTensor<T, Allocator, GPUAllocator>{static_cast<GPUTensor<T, Allocator, GPUAllocator>&&>(other)}
    {
  
    }

    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator> &LowTriMatrix<T, Allocator, GPUAllocator>::operator=(LowTriMatrix<T, Allocator, GPUAllocator> &&other)
    {
       if (this != &other)
        {
            GPUTensor<T, Allocator, GPUAllocator>::operator=(static_cast<GPUTensor<T, Allocator, GPUAllocator>&&>(other));
            n_ = other.size();
            other.n_ = 0;
        }
        return *this;
    }

    template <typename T, class Allocator, class GPUAllocator>
    T &LowTriMatrix<T, Allocator, GPUAllocator>::operator()(size_t i, size_t j)
    {
        if (i < j)
        {
            size_t tmp = i;
            i = j;
            j = tmp;
        }
        return this->data_[data_id_(i, j)];
    }

    template <typename T, class Allocator, class GPUAllocator>
    const T &LowTriMatrix<T, Allocator, GPUAllocator>::operator()(size_t i, size_t j) const
    {
        if (i < j)
        {
            size_t tmp = i;
            i = j;
            j = tmp;
        }
        return this->data_[data_id_(i, j)];
    }

    template <typename T, class Allocator, class GPUAllocator>
    void LowTriMatrix<T, Allocator, GPUAllocator>::print() const
    {
        for (size_t i = 0; i < n_; i++)
        {
            for (size_t j = 0; j <= i; j++)
            {
                std::cout << this->data_[data_id_(i, j)] << ", ";
            }
            std::cout << std::endl;
        }
    }

    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator> LowTriMatrix<T, Allocator, GPUAllocator>::get_diagonal() const
    {
        size_t min_dim = n_;
        Vector<T, Allocator, GPUAllocator> diag(min_dim);

        for (size_t i = 0; i < min_dim; i++)
        {
            diag[i] = (this->data_[data_id_(i, i)]);
        }

        return diag;
    }

    template <typename T, class Allocator, class GPUAllocator>
    void LowTriMatrix<T, Allocator, GPUAllocator>::set_diagonal(Vector<T, Allocator, GPUAllocator> &diag)
    {
        size_t min_dim = n_;
#pragma omp for
        for (size_t i = 0; i < min_dim; i++)
        {
            this->data_[data_id_(i, i)] = diag[i];
        }
    }

} // namespace gpu
} // namespace lahva