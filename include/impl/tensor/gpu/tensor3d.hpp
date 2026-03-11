#pragma once

#include "impl/tensor/allocators.hpp"
#include "impl/tensor/gpu/gputensor.hpp"
#include "impl/tensor/cpu/tensor.hpp"
#include "impl/tensor/cpu/matrix.hpp"
#include "impl/tensor/cpu/tensor3d.hpp"
#include "impl/tensor/gpu/vector.hpp"
#include "impl/blas/cpu/level1.h"
#include <initializer_list>
#include <omp.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cassert>
namespace lahva
{
    namespace gpu
    {
        
    
    
    template<typename T>
    class Tensor3D_ : public virtual GPUTensor_<T>, public virtual cpu::Tensor3D_<T>
    {
        public:
            virtual Shape shape() const  = 0;

            //! @param[in] i index in dimension 1
            //! @param[in] j index in dimension 2
            //! @param[in] k index in dimension 3
            //! @return reference to Matrix element i,j,k
            virtual T &operator()(size_t i, size_t j, size_t k) = 0;
            //! @param[in] i index in dimension 1
            //! @param[in] j index in dimension 2
            //! @param[in] k index in dimension 3
            //! @return reference to Matrix element i,j,k
            virtual const T &operator()(size_t i, size_t j, size_t k) const = 0;

    };

    // Forward declarations for GPU kernel wrapper functions
    template <typename T>
    void GetDiagonal(const CudaRuntime& cudart, const Tensor3D_<T>& m, GPUTensor_<T>& diag);

    template <typename T>
    void SetDiagonal(const CudaRuntime& cudart, const GPUTensor_<T>& diag, Tensor3D_<T>& m);


    //! @brief slim wrapper around a float or double array to allow easy acces with
    //!        three indices using the () operator.
    //!
    //! If NDEBUG is **not** defined, range checks are performed.
    //! The data is stored in column-major order in a 1D array.
    //!
    template <class T, class Allocator = CudaHostAllocator<T>, class GPUAllocator = CudaDeviceAllocator<T>>
    class Tensor3D : public GPUTensor<T, Allocator, GPUAllocator>, public virtual Tensor3D_<T>
    {
        using alloc_ptr = Allocator;
        using gpualloc_ptr = GPUAllocator;
    protected:
        // shape in each dimension, i.e. data_ has length n_1_*n_2_*n_3_
        size_t n_1_ = 0;
        size_t n_2_ = 0;
        size_t n_3_ = 0;

        // indicates whether the Matrix object owns the data and consequently is
        // responsible for freeing it

        inline size_t data_id_(size_t i, size_t j, size_t k) const
        {
            // deactivated if NDEBUG is defined
            assert(i < n_1_ && j < n_2_ && k < n_3_);

            // range checks are perfomred in constructor and above (if in debug mode)
            return i + j * n_1_ + k * n_1_ * n_2_;
        }

        // length of the array data_
        inline size_t data_size_(size_t n_1, size_t n_2, size_t n_3) 
        {
            return n_1 * n_2 * n_3;
        }

        // raises an error, if the shape is not valid
        // Vector uses size_t as shape, so the check shape function has to be able
        // to deal with that
        inline void check_size_(size_t, size_t, size_t);

    public:
        Tensor3D() : GPUTensor<T, Allocator, GPUAllocator>{} {};
        //! construct a square Tensor3D with dimensions n x n x n
        //! It is not guaranteed that the values will be initialized
        Tensor3D(size_t n, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());
        //! construct a square n x n x n Tensor3D initialized with value val
        Tensor3D(size_t n, T val, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());

        //! construct a Tensor3D with dimensions shape.first x shape.second x shape.third
        //! It is not guaranteed that the values will be initialized!
        Tensor3D(const Shape &shape, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());

        Tensor3D(const Shape &shape, std::initializer_list<T> init, bool row_major = false, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());

        template <typename U>
        Tensor3D(const Shape &shape, const CudaRuntime &cudart, const U &alloc)
            : Tensor3D<T,Allocator,GPUAllocator>(shape, cudart, static_cast<GPUAllocator>(alloc))
           {
            };

        Tensor3D(const Shape &shape, const CudaRuntime &cudart, const gpualloc_ptr &gpualloc = GPUAllocator());
        //! construct a Tensor3D of shape with initial values val
        Tensor3D(const Shape &shape, T val, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());
        //! construct a Tensor3D by giving ownership of the raw data
        //! note: the data needs to be in the format defined by data_id_()
        //! @param take_ownership if false, the data will not be freed, when the
        //!                       Tensor3D object is destructed. Use with care!
        Tensor3D(const Shape &shape, T *data, bool take_ownership = true, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());
        Tensor3D(const Shape &shape, const T *data, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());

        Tensor3D(const Tensor3D &);
        Tensor3D(Tensor3D &&);
        Tensor3D &operator=(const Tensor3D &);
        Tensor3D &operator=(Tensor3D &&);
        virtual ~Tensor3D();

        //! @param[in] i index in dimension 1
        //! @param[in] j index in dimension 2
        //! @param[in] k index in dimension 3
        //! @return reference to Tensor3D element i,j,k
        T &operator()(size_t i, size_t j, size_t k);
        //! @param[in] i index in dimension 1
        //! @param[in] j index in dimension 2
        //! @param[in] k index in dimension 3
        //! @return reference to Tensor3D element i,j,k
        const T &operator()(size_t i, size_t j, size_t k) const;

        //! @brief in-place, scalar addition
        Tensor3D &operator+=(T val);

        //! @return number of rows/columns of the Tensor3D
        Shape shape() const { return Shape{n_1_, n_2_, n_3_}; }

        //! prints the Tensor3D as string
        void print() const;

        void print(const char* file) const
        {
            std::ofstream os(file);
            os.precision(10);
            if (os.is_open())
            {
                // Print each slice through the first index
                for (size_t i = 0; i < n_1_; i++)
                {
                    os << "Slice [" << i << "]:\n";
                    Matrix<T> slice(Shape(n_2_, n_3_));
                    
                    // Extract slice i into a 2D matrix
                    for (size_t j = 0; j < n_2_; j++)
                    {
                        for (size_t k = 0; k < n_3_; k++)
                        {
                            slice(j, k) = this->data_[this->data_id_(i, j, k)];
                        }
                    }
                    slice.print(os);
                    os << "\n";
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

        Vector<T, Allocator, GPUAllocator> get_diagonal(const CudaRuntime& cudart) const;

        void set_diagonal(const CudaRuntime& cudart, const Vector<T, Allocator, GPUAllocator> &diag);

        bool ownsData() { return this->is_owner_; };
    };

    template <typename T, class Allocator, class GPUAllocator>
    void Tensor3D<T, Allocator, GPUAllocator>::check_size_(size_t n_1,
                                size_t n_2, size_t n_3)
{
        if (n_1 > SIZE_MAX)
        {
            throw std::out_of_range("Dimension 1 exceeds maximum Tensor size.");
        }
        else if (n_2 > SIZE_MAX)
        {
            throw std::out_of_range("Dimension 2 exceeds maximum Tensor size.");
        }
        else if (n_3 > SIZE_MAX)
        {
            throw std::out_of_range("Dimension 3 exceeds maximum Tensor size.");
        }
        else if (data_size_(n_1, n_2, n_3) > SIZE_MAX)
        {
            throw std::out_of_range("Exceeds maximum Tensor size.");
        }
    }

    template <typename T, class Allocator, class GPUAllocator>
    Tensor3D<T, Allocator, GPUAllocator>::Tensor3D(const Shape &shape, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) : 
    GPUTensor<T, Allocator, GPUAllocator>{shape.first*shape.second*shape.third, alloc, gpualloc}, 
    n_1_{shape.first}, n_2_{shape.second}, n_3_{shape.third}
    {
        this->check_size_(shape.first, shape.second, shape.third);
    }

    template <typename T, class Allocator, class GPUAllocator>
    Tensor3D<T, Allocator, GPUAllocator>::Tensor3D(const Shape &shape, std::initializer_list<T> init, bool row_major, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) : 
    Tensor3D(shape, alloc, gpualloc)
    {
        assert(init.size() == this->count_);
        if (row_major)
        {
            #pragma omp parallel for ordered schedule(static)
            for (size_t mu = 0; mu < this->n_1_; mu++)
            {
                for (size_t nu = 0; nu < this->n_2_; nu++)
                {
                    for (size_t kappa = 0; kappa < this->n_3_; kappa++)
                    {
                        #pragma omp ordered
                        this->data_[this->data_id_(mu, nu, kappa)] = *(init.begin() + (mu * this->n_2_ * this->n_3_+ nu * this->n_3_));
                    }
                }
            }
        }
        else
        {
            std::copy(init.begin(), init.end(), this->data_);
        }
    }

    template <typename T, class Allocator, class GPUAllocator>
    Tensor3D<T, Allocator, GPUAllocator>::Tensor3D(const Shape &shape, const CudaRuntime &cudart, const gpualloc_ptr &gpualloc) : 
    GPUTensor<T, Allocator, GPUAllocator>{gpualloc},
    n_1_{shape.first}, n_2_{shape.second}, n_3_{shape.third}
    {
        check_size_(shape.first, shape.second, shape.third);
        this->count_ = n_1_*n_2_*n_3_;
        this->is_owner_ = false;
        this->gpu_buffer = true;
    }

    template <typename T, class Allocator, class GPUAllocator>
    Tensor3D<T, Allocator, GPUAllocator>::Tensor3D(const Shape &shape, T *data, bool take_ownership, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) : 
    GPUTensor<T, Allocator, GPUAllocator>{alloc, gpualloc}, n_1_{shape.first}, n_2_{shape.second}, n_3_{shape.third}
    {
        this->data_ = data;
        this->count_ = n_1_*n_2_*n_3_;
        this->is_owner_ = take_ownership;
    }


    template <typename T, class Allocator, class GPUAllocator>
    Tensor3D<T, Allocator, GPUAllocator>::Tensor3D(const Shape &shape, T val, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) : 
    Tensor3D(shape, alloc, gpualloc)
    {
        std::fill(this->data_, this->data_ + data_size_(n_1_, n_2_, n_3_), val);
    }

    template <typename T, class Allocator, class GPUAllocator>
    Tensor3D<T, Allocator, GPUAllocator>::Tensor3D(size_t n, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) : 
    Tensor3D<T, Allocator, GPUAllocator>::Tensor3D(Shape(n, n, n, n), alloc, gpualloc) 
    {}

    template <typename T, class Allocator, class GPUAllocator>
    Tensor3D<T, Allocator, GPUAllocator>::Tensor3D(size_t n, T val, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) : 
    Tensor3D<T, Allocator, GPUAllocator>::Tensor3D(Shape(n, n, n, n), val, alloc, gpualloc) {}

    template <typename T, class Allocator, class GPUAllocator>
    Tensor3D<T, Allocator, GPUAllocator>::Tensor3D(const Shape &shape, const T *data, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) : 
    Tensor3D<T, Allocator, GPUAllocator>::Tensor3D(shape, alloc, gpualloc)
    {
        std::copy(data, data + this->data_size_(n_1_, n_2_, n_3_), this->data_);
    };

    template <typename T, class Allocator, class GPUAllocator>
    Tensor3D<T, Allocator, GPUAllocator>::~Tensor3D()
    {
        
    }

    // copy operations
    template <typename T, class Allocator, class GPUAllocator>
    Tensor3D<T, Allocator, GPUAllocator>::Tensor3D(const Tensor3D<T, Allocator, GPUAllocator> &other) : 
    GPUTensor<T, Allocator, GPUAllocator>{other}, 
    n_1_{other.n_1_}, n_2_{other.n_2_}, n_3_{other.n_3_}
    {
    }

    template <typename T, class Allocator, class GPUAllocator>
    Tensor3D<T, Allocator, GPUAllocator> &Tensor3D<T, Allocator, GPUAllocator>::operator=(const Tensor3D<T, Allocator, GPUAllocator> &other)
    {
        if (this != &other)
        {
            GPUTensor<T, Allocator, GPUAllocator>::operator=(other);

            n_1_ = other.n_1_;
            n_2_ = other.n_2_;
            n_3_ = other.n_3_;
        }

        return *this;
    }

    // move operations
    template <typename T, class Allocator, class GPUAllocator>
    Tensor3D<T, Allocator, GPUAllocator>::Tensor3D(Tensor3D<T, Allocator, GPUAllocator> &&other) : 
    GPUTensor<T, Allocator, GPUAllocator>{other},
    n_1_{other.n_1_}, n_2_{other.n_2_}, n_3_{other.n_3_}
    {
   

        other.n_1_ = 0;
        other.n_2_ = 0;
        other.n_3_ = 0;

    }

    template <typename T, class Allocator, class GPUAllocator>
    Tensor3D<T, Allocator, GPUAllocator> &Tensor3D<T, Allocator, GPUAllocator>::operator=(Tensor3D<T, Allocator, GPUAllocator> &&other)
    {
        if (this != &other)
        {
            GPUTensor<T, Allocator, GPUAllocator>::operator=(std::move(other));
            
            n_1_ = other.n_1_;
            n_2_ = other.n_2_;
            n_3_ = other.n_3_;

            other.n_1_ = 0;
            other.n_2_ = 0;
            other.n_3_ = 0;
        }
        return *this;
    }

    template <typename T, class Allocator, class GPUAllocator>
    T &Tensor3D<T, Allocator, GPUAllocator>::operator()(size_t i, size_t j, size_t k)
    {
        return this->data_[this->data_id_(i, j, k)];
    }

    template <typename T, class Allocator, class GPUAllocator>
    const T &Tensor3D<T, Allocator, GPUAllocator>::operator()(size_t i, size_t j, size_t k) const
    {
        return this->data_[this->data_id_(i, j, k)];
    }

    template <typename T, class Allocator, class GPUAllocator>
    Tensor3D<T, Allocator, GPUAllocator> &Tensor3D<T, Allocator, GPUAllocator>::operator+=(T val)
    {
    #pragma omp for
        for (size_t i = 0; i < data_size_(n_1_, n_2_, n_3_); i++)
        {
            this->data_[i] += val;
        }

        return *this;
    }

    template <typename T, class Allocator, class GPUAllocator>
    void Tensor3D<T, Allocator, GPUAllocator>::print() const
    {
        // Print each slice through the first index
        for (size_t i = 0; i < n_1_; i++)
        {
            std::cout << "Slice [" << i << "]:\n";
            Matrix<T> slice(Shape(n_2_, n_3_));
            
            // Extract slice i into a 2D matrix
            for (size_t j = 0; j < n_2_; j++)
            {
                for (size_t k = 0; k < n_3_; k++)
                {
                    slice(j, k) = this->data_[this->data_id_(i, j, k)];
                }
            }
            slice.print();
            std::cout << "\n";
        }
    }

    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator> Tensor3D<T, Allocator, GPUAllocator>::get_diagonal() const
    {
        size_t min_dim = std::min(std::min(n_1_, n_2_), n_3_);
        Vector<T, Allocator, GPUAllocator> diag(min_dim);
        size_t stride = n_1_ + n_1_ * n_2_;
        cpu::CopyVectors(diag.size(), this->data(), stride+1 ,diag.data(), 1);

        return diag;
    }

    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator> Tensor3D<T, Allocator, GPUAllocator>::get_diagonal(const CudaRuntime& cudart) const
    {
        size_t min_dim = std::min(std::min(n_1_, n_2_), n_3_);
        Vector<T, Allocator, GPUAllocator> diag(min_dim);
        gpu::GetDiagonal<T>(cudart, *this, diag);
        return diag;
    }

    template <typename T, class Allocator, class GPUAllocator>
    void Tensor3D<T, Allocator, GPUAllocator>::set_diagonal(const Vector<T, Allocator, GPUAllocator> &diag)
    {
        size_t min_dim = std::min(std::min(n_1_, n_2_), n_3_);
        if (diag.size() != min_dim)
        {
            throw std::runtime_error("The vector given to set diagonal doesn't correspond to the minimal dimension.");
        }
        size_t stride = n_1_ + n_1_ * n_2_;
        cpu::CopyVectors(diag.size(), diag.data(), 1, this->data(), stride+1);
    }

    template <typename T, class Allocator, class GPUAllocator>
    void Tensor3D<T, Allocator, GPUAllocator>::set_diagonal(const CudaRuntime& cudart, const Vector<T, Allocator, GPUAllocator> &diag)
    {
        size_t min_dim = std::min(std::min(n_1_, n_2_), n_3_);
        if (diag.size() != min_dim)
        {
            throw std::runtime_error("The vector given to set diagonal doesn't correspond to the minimal dimension.");
        }
        gpu::SetDiagonal<T>(cudart, diag, *this);
    }

    
    } // namespace gpu
} // namespace lahva
