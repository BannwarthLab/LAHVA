#pragma once
#include "impl/tensor/allocators.hpp"
#include "impl/tensor/cpu/tensor.hpp"
#include <omp.h>

namespace lahva
{
    namespace cpu
    {
        

    template<typename T>
    class LowTriMatrix_ : virtual public Tensor<T>
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

    template <class T, class Allocator = StdAllocator<T>>
    class LowTriMatrix : virtual public CPUTensor<T, Allocator>, virtual public LowTriMatrix_<T>
    {
        using alloc_ptr = CPUAllocator<T>;   
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
        LowTriMatrix(size_t n,const alloc_ptr &cpualloc = Allocator());
        template <typename U>
        LowTriMatrix(size_t n, const std::shared_ptr<CPUAllocator<U>> &alloc)
        : LowTriMatrix<T,Allocator>(n, Allocator(*alloc))
        { };
        //! construct an n x n matrix initialized with value val
        LowTriMatrix(size_t n, T val,const alloc_ptr &cpualloc = Allocator());
        template <typename U>
        LowTriMatrix(size_t n, T val, const std::shared_ptr<CPUAllocator<U>> &alloc)
        : LowTriMatrix<T,Allocator>(n, val, Allocator(*alloc))
        { };
        //! give dimension n and data as a Vector conatining the values of the lower traingular Matrix
        LowTriMatrix(size_t n, const Vector_<T> &data, const alloc_ptr &cpualloc = Allocator());
        //! create a matrix with shape (shape must be square)
        //! It is not guaranteed that the values will be initialized
        LowTriMatrix(const Shape &shape, const alloc_ptr &cpualloc = Allocator());
        //! create a non-initialized matrix with shape (shape must be square) and
        //! initialize all values to val
        LowTriMatrix(const Shape &shape, T val, const alloc_ptr &cpualloc = Allocator());
        //! construct a matrix by giving ownership of the raw data
        //! note: the data needs to be in the format defined by data_id_()
        //! @param n number of columns/rows of the matrix
        LowTriMatrix(size_t n, T *data, bool take_ownership = true,const alloc_ptr &cpualloc = Allocator());
        LowTriMatrix(const LowTriMatrix &);
        LowTriMatrix(LowTriMatrix &&) noexcept;
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

        Vector<T, Allocator> get_diagonal() const;

        void set_diagonal(Vector<T, Allocator> &diag);
    };

    template <typename T, class Allocator>
    void LowTriMatrix<T, Allocator>::check_size_(long unsigned int n)
    {
        if (n > SIZE_MAX || n > std::sqrt(SIZE_MAX))
        {
            throw std::out_of_range("Vector exceeds maximum LowTriMatrix size.");
        }
    }

    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator>::LowTriMatrix(size_t size, const alloc_ptr& alloc ) : 
    CPUTensor<T, Allocator>{data_size_(size), alloc}, n_{size}                                     
    {
        check_size_(size);
    }

    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator>::LowTriMatrix(size_t size, T val, const alloc_ptr& alloc ) : 
    LowTriMatrix(size, alloc)
    {
        check_size_(size);

        std::fill(this->data_, this->data_ + data_size_(size), val);
    }

    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator>::LowTriMatrix(const Shape &shape, const alloc_ptr& alloc ) : 
    LowTriMatrix(shape.first, alloc)
    {
        if (shape.first != shape.second)
        {
            throw std::logic_error("shape.first must be equal to shape.second.");
        }
    }

    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator>::LowTriMatrix(const Shape &shape, T val, const alloc_ptr& alloc ) : 
    LowTriMatrix{shape.first, val, alloc}
    {
        if (shape.first != shape.second)
        {
            throw std::logic_error("shape.first must be equal to shape.second.");
        }
    }


    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator>::LowTriMatrix(size_t n, const Vector_<T> &data, const alloc_ptr& alloc ) : 
    LowTriMatrix{n, alloc}
    {
        check_size_(data.size());
        #pragma omp for
        for (size_t i = 0; i < data_size_(n_); i++)
        {
            this->data_[i] = data[i];
        }
    }
    
    
    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator>::LowTriMatrix(size_t n, T *data, bool take_ownership,const alloc_ptr &cpualloc)
    :  CPUTensor<T, Allocator>{cpualloc}, n_{n}
    {
        this->data_ = data;
        this->count_ = data_size_(n);
        this->is_owner_ = take_ownership;
    
    }

    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator>::~LowTriMatrix()
    {
    
    }

    // copy operations
    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator>::LowTriMatrix(const LowTriMatrix<T, Allocator> &other) : 
    n_{other.n_}, 
    CPUTensor<T, Allocator>(other)
    {
        
    }

    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator> &LowTriMatrix<T, Allocator>::operator=(const LowTriMatrix<T, Allocator> &other)
    {
        if (this != &other)
        {
            CPUTensor<T, Allocator>::operator=(other);
            n_ = other.size();
        }
        return *this;
    }

    // move operations
    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator>::LowTriMatrix(LowTriMatrix<T, Allocator> &&other) noexcept :
    CPUTensor<T, Allocator>{std::move(other)}, n_{other.n_}
    {

    }

    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator> &LowTriMatrix<T, Allocator>::operator=(LowTriMatrix<T, Allocator> &&other)
    {
       if (this != &other)
        {
            CPUTensor<T, Allocator>::operator=(std::move(other));
            n_ = other.size();
            other.n_ = 0;
        }
        return *this;
    }

    template <typename T, class Allocator>
    T &LowTriMatrix<T, Allocator>::operator()(size_t i, size_t j)
    {
        if (i < j)
        {
            size_t tmp = i;
            i = j;
            j = tmp;
        }
        return this->data_[data_id_(i, j)];
    }

    template <typename T, class Allocator>
    const T &LowTriMatrix<T, Allocator>::operator()(size_t i, size_t j) const
    {
        if (i < j)
        {
            size_t tmp = i;
            i = j;
            j = tmp;
        }
        return this->data_[data_id_(i, j)];
    }

    template <typename T, class Allocator>
    void LowTriMatrix<T, Allocator>::print() const
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

    template <typename T, class Allocator>
    Vector<T, Allocator> LowTriMatrix<T, Allocator>::get_diagonal() const
    {
        size_t min_dim = n_;
        Vector<T, Allocator> diag(min_dim);

        for (size_t i = 0; i < min_dim; i++)
        {
            diag[i] = (this->data_[data_id_(i, i)]);
        }

        return diag;
    }

    template <typename T, class Allocator>
    void LowTriMatrix<T, Allocator>::set_diagonal(Vector<T, Allocator> &diag)
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