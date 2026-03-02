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
        // Forward declaration
        template <class T, class Allocator>
        class Matrix;
    
    
    template<typename T>
    class Tensor4D_ : public virtual Tensor<T>
    {
        public:
            virtual Shape shape() const  = 0;

            //! @param[in] i index in dimension 1
            //! @param[in] j index in dimension 2
            //! @param[in] k index in dimension 3
            //! @param[in] l index in dimension 4
            //! @return reference to Tensor4D element i,j,k,l
            virtual T &operator()(size_t i, size_t j, size_t k, size_t l) = 0;
            //! @param[in] i index in dimension 1
            //! @param[in] j index in dimension 2
            //! @param[in] k index in dimension 3
            //! @param[in] l index in dimension 4
            //! @return reference to Tensor4D element i,j,k,l
            virtual const T &operator()(size_t i, size_t j, size_t k, size_t l) const = 0;

    };


    //! @brief slim wrapper around a float or double array to allow easy acces with
    //!        four indices using the () operator.
    //!
    //! If NDEBUG is **not** defined, range checks are performed.  d
    //! The data is stored in column-major order in a 1D array.
    //!
    template <class T, class Allocator = StdAllocator<T>>
    class Tensor4D : virtual public CPUTensor<T, Allocator>, virtual public Tensor4D_<T>
    {
        using alloc_ptr = CPUAllocator<T>;
    protected:
        // shape in each dimension, i.e. data_ has length n_1_*n_2_*n_3_*n_4_
        size_t n_1_;
        size_t n_2_;
        size_t n_3_;
        size_t n_4_;
    
        // indicates whether the Tensor4D object owns the data and consequently is
        // responsible for freeing it

        inline size_t data_id_(size_t i, size_t j, size_t k, size_t l) const
        {
            // deactivated if NDEBUG is defined
            assert(i < this->n_1_ && j < this->n_2_ && k < this->n_3_ && l < this->n_4_);

            // range checks are perfomred in constructor and above (if in debug mode)
            return i + j * n_1_ + k * n_1_ * n_2_ + l * n_1_ * n_2_ * n_3_;
        }

        // length of the array data_
        inline size_t data_size_(size_t n_1, size_t n_2, size_t n_3, size_t n_4)
        {
            return n_1 * n_2 * n_3 * n_4;
        }

        // raises an error, if the shape is not valid
        // Vector uses size_t as shape, so the check shape function has to be able
        // to deal with that
        inline void check_size_(size_t, size_t, size_t, size_t);

    public:
        Tensor4D() {};
        //! construct a Tensor4D with dimensions n x n x n x n
        //! It is not guaranteed that the values will be initialized
        Tensor4D(size_t n, const alloc_ptr &alloc = Allocator());
        template<typename U>
        Tensor4D(size_t n, const std::shared_ptr<CPUAllocator<U>> &alloc)
        : Tensor4D<T, Allocator>{n, Allocator(*alloc)} {};
        //! construct a n x n x n x n Tensor4D initialized with value val
        Tensor4D(size_t n, T val, const alloc_ptr &alloc = Allocator());
        template<typename U>
        Tensor4D(size_t n, T val, const std::shared_ptr<CPUAllocator<U>> &alloc)
        : Tensor4D<T, Allocator>{n, val, Allocator(*alloc)} {};
        //! construct a Tensor4D with dimensions shape.first x shape.second x shape.third x shape.fourth
        //! It is not guaranteed that the values will be initialized!
        Tensor4D(const Shape &shape, const alloc_ptr &alloc = Allocator());
        Tensor4D(const Shape &shape, std::initializer_list<T> init, bool row_major = false, const alloc_ptr &alloc = Allocator());
        template<typename U>
        Tensor4D(const Shape& shape, const std::shared_ptr<CPUAllocator<U>> &alloc)
        : Tensor4D<T, Allocator>{shape, Allocator(*alloc)} {};
        //! construct a Tensor4D of shape with initial values val
        Tensor4D(const Shape &shape, T val, const alloc_ptr &alloc = Allocator());
        template<typename U>
        Tensor4D(const Shape &shape, T val, const std::shared_ptr<CPUAllocator<U>> &alloc)
        : Tensor4D<T, Allocator>{shape, val, Allocator(*alloc)} {};
        //! construct a Tensor4D by giving ownership of the raw data
        //! note: the data needs to be in the format defined by data_id_()
        //! @param take_ownership if false, the data will not be freed, when the
        //!                       Matrix object is destructed. Use with care!
        Tensor4D(const Shape &shape, T *data, bool take_ownership = true, const alloc_ptr &alloc = Allocator());
        template<typename U>
        Tensor4D(const Shape &shape, T* data, bool take_ownership = true, const std::shared_ptr<CPUAllocator<U>> &alloc = Allocator())
        : Tensor4D<T, Allocator>{shape, data, take_ownership, Allocator(*alloc)} {};
        Tensor4D(const Shape &shape, const T *data, const alloc_ptr &alloc = Allocator());
        template<typename U>
        Tensor4D(const Shape &shape, T* data, const std::shared_ptr<CPUAllocator<U>> &alloc)
        : Tensor4D<T, Allocator>{shape, data, Allocator(*alloc)} {};

        Tensor4D(const Tensor4D &);
        Tensor4D(Tensor4D &&);
        Tensor4D &operator=(const Tensor4D &);
        Tensor4D &operator=(Tensor4D &&);
        virtual ~Tensor4D();

        //! @param[in] i index in dimension 1
        //! @param[in] j index in dimension 2
        //! @param[in] k index in dimension 3
        //! @param[in] l index in dimension 4
        //! @return reference to Tensor4D element i,j,k,l
        T &operator()(size_t i, size_t j, size_t k, size_t l);
        //! @param[in] i index in dimension 1
        //! @param[in] j index in dimension 2
        //! @param[in] k index in dimension 3
        //! @param[in] l index in dimension 4
        //! @return reference to Tensor4D element i,j,k,l
        const T &operator()(size_t i, size_t j, size_t k, size_t l) const;

        //! @brief in-place, scalar addition
        Tensor4D &operator+=(T val);

        //! @return number of rows/columns of the Tensor4D
        Shape shape() const { return Shape{n_1_, n_2_, n_3_, n_4_}; }

        //! prints the Tensor4D as string
        void print() const;

        void print(const char* file) const
        {
            std::ofstream os(file);
            os.precision(10);
            if (os.is_open())
            {
                Matrix<T> temp((n_1_*n_2_), (n_3_*n_4_)); 
                // Transform 4D tensor to 2D matrix with contracted indices for printing
                for (size_t mu = 0; mu < n_1_; mu++)
                {
                    for (size_t nu = 0; nu < n_2_; nu++)
                    {
                        size_t i = mu * n_2_ + nu;
                        for (size_t kappa = 0; kappa < n_3_; kappa++)
                        {
                            for (size_t lambda = 0; lambda < n_4_; lambda++)
                            {
                                size_t j = kappa * n_4_ + lambda;
                                temp(i,j) = this->data_[this->data_id_(mu, nu, kappa, lambda)];
                            }
                        }
                    }
                }
                temp.print(os);
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

        Matrix<T, Allocator> get_coulomb_eri() const;
        Matrix<T, Allocator> get_exchange_eri() const;

        bool ownsData() { return this->is_owner_; };
    };

     template <typename T, class Allocator>
    void Tensor4D<T, Allocator>::check_size_(size_t n_1,
                                size_t n_2, size_t n_3, size_t n_4)
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
        else if (n_4 > SIZE_MAX)
        {
            throw std::out_of_range("Dimension 4 exceeds maximum Tensor size.");
        }
        else if (data_size_(n_1, n_2, n_3, n_4) > SIZE_MAX)
        {
            throw std::out_of_range("Exceeds maximum Tensor size.");
        }
    }

    template <typename T, class Allocator>
    Tensor4D<T, Allocator>::Tensor4D(const Shape &shape, const alloc_ptr &alloc) : 
    CPUTensor<T, Allocator>{data_size_(shape.first, shape.second, shape.third, shape.fourth), alloc}, n_1_{shape.first}, n_2_{shape.second}, n_3_{shape.third}, n_4_{shape.fourth}
    {
        check_size_(shape.first, shape.second, shape.third, shape.fourth);
    }

    template <typename T, class Allocator>
    Tensor4D<T, Allocator>::Tensor4D(const Shape &shape, T *data, bool take_ownership, const alloc_ptr &alloc) : 
    CPUTensor<T, Allocator>{alloc}, n_1_{shape.first}, n_2_{shape.second}, n_3_{shape.third}, n_4_{shape.fourth}
    {
        this->data_ = data;
        this->count_ = n_1_ * n_2_ * n_3_ * n_4_;
        this->is_owner_ = take_ownership;
    }


    template <typename T, class Allocator>
    Tensor4D<T, Allocator>::Tensor4D(const Shape &shape, T val, const alloc_ptr &alloc) : 
    Tensor4D(shape, alloc)
    {
        std::fill(this->data_, this->data_ + data_size_(n_1_, n_2_, n_3_, n_4_), val);
    }

    template <typename T, class Allocator>
    Tensor4D<T, Allocator>::Tensor4D(const Shape &shape, std::initializer_list<T> init, bool row_major, const alloc_ptr &alloc) :
    Tensor4D(shape, alloc)
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
                        for (size_t lambda = 0; lambda < this->n_4_; lambda++)
                        {
                            #pragma omp ordered
                            this->data_[this->data_id_(mu, nu, kappa, lambda)] = *(init.begin() + (mu * this->n_2_ * this->n_3_ * this->n_4_ + nu * this->n_3_ * this->n_4_ + kappa * this->n_4_ + lambda));
                        }
                    }
                }
            }
        }
        else
        {
            std::copy(init.begin(), init.end(), this->data_);
        }
        
        
    };
    

    template <typename T, class Allocator>
    Tensor4D<T, Allocator>::Tensor4D(size_t n, const alloc_ptr &alloc) : 
    Tensor4D<T, Allocator>::Tensor4D(Shape(n, n, n, n), alloc) 
    {}

    template <typename T, class Allocator>
    Tensor4D<T, Allocator>::Tensor4D(size_t n, T val, const alloc_ptr &alloc) : 
    Tensor4D<T, Allocator>::Tensor4D(Shape(n, n, n, n), val, alloc) {}

    template <typename T, class Allocator>
    Tensor4D<T, Allocator>::Tensor4D(const Shape &shape, const T *data, const alloc_ptr &alloc) : 
    Tensor4D<T, Allocator>::Tensor4D(shape, alloc)
    {
        std::copy(data, data + this->data_size_(n_1_, n_2_, n_3_, n_4_), this->data_);
    };

    template <typename T, class Allocator>
    Tensor4D<T, Allocator>::~Tensor4D()
    {
        
    }

    // copy operations
    template <typename T, class Allocator>
    Tensor4D<T, Allocator>::Tensor4D(const Tensor4D<T, Allocator> &other) : 
    CPUTensor<T, Allocator>{other}, 
    n_1_{other.n_1_}, n_2_{other.n_2_}, n_3_{other.n_3_}, n_4_{other.n_4_}
    {
        
    }

    template <typename T, class Allocator>
    Tensor4D<T, Allocator> &Tensor4D<T, Allocator>::operator=(const Tensor4D<T, Allocator> &other)
    {
        if (this != &other)
        {
            CPUTensor<T, Allocator>::operator=(other);

            n_1_ = other.n_1_;
            n_2_ = other.n_2_;
            n_3_ = other.n_3_;
            n_4_ = other.n_4_;
        }

        return *this;
    }

    // move operations
    template <typename T, class Allocator>
    Tensor4D<T, Allocator>::Tensor4D(Tensor4D<T, Allocator> &&other) :
    CPUTensor<T, Allocator>{other},
    n_1_{other.n_1_}, n_2_{other.n_2_}, n_3_{other.n_3_}, n_4_{other.n_4_}
    {

        other.n_1_ = 0;
        other.n_2_ = 0;
        other.n_3_ = 0;
        other.n_4_ = 0;

    }

    template <typename T, class Allocator>
    Tensor4D<T, Allocator> &Tensor4D<T, Allocator>::operator=(Tensor4D<T, Allocator> &&other)
    {
        if (this != &other)
        {
            CPUTensor<T, Allocator>::operator=(std::move(other));
            
            n_1_ = other.n_1_;
            n_2_ = other.n_2_;
            n_3_ = other.n_3_;
            n_4_ = other.n_4_;

            other.n_1_ = 0;
            other.n_2_ = 0;
            other.n_3_ = 0;
            other.n_4_ = 0;
        }
        return *this;
    }

    template <typename T, class Allocator>
    T &Tensor4D<T, Allocator>::operator()(size_t i, size_t j, size_t k, size_t l)
    {
        return this->data_[this->data_id_(i, j, k, l)];
    }

    template <typename T, class Allocator>
    const T &Tensor4D<T, Allocator>::operator()(size_t i, size_t j, size_t k, size_t l) const
    {
        return this->data_[this->data_id_(i, j, k, l)];
    }

    template <typename T, class Allocator>
    Tensor4D<T, Allocator> &Tensor4D<T, Allocator>::operator+=(T val)
    {
    #pragma omp for
        for (size_t i = 0; i < data_size_(n_1_, n_2_, n_3_, n_4_); i++)
        {
            this->data_[i] += val;
        }

        return *this;
    }

    template <typename T, class Allocator>
    void Tensor4D<T, Allocator>::print() const
    {
        Matrix<T> temp((n_1_*n_2_), (n_3_*n_4_)); 
        // Transform 4D tensor to 2D matrix with contracted indices for printing
        for (size_t mu = 0; mu < n_1_; mu++)
        {
            for (size_t nu = 0; nu < n_2_; nu++)
            {
                size_t i = mu * n_2_ + nu;
                for (size_t kappa = 0; kappa < n_3_; kappa++)
                {
                    for (size_t lambda = 0; lambda < n_4_; lambda++)
                    {
                        size_t j = kappa * n_4_ + lambda;
                        temp(i,j) = this->data_[this->data_id_(mu, nu, kappa, lambda)];
                    }
                }
            }
        }

        temp.print();
    }

    template <typename T, class Allocator>
    Vector<T, Allocator> Tensor4D<T, Allocator>::get_diagonal() const
    {
        size_t min_dim = std::min(std::min(n_1_, n_2_), std::min(n_3_, n_4_));
        Vector<T, Allocator> diag(min_dim);
        size_t stride = n_1_ + n_1_ * n_2_ + n_1_ * n_2_ * n_3_;
        cpu::CopyVectors(diag.size(), this->data(), stride+1, diag.data(), 1);

        return diag;
    }

     template <typename T, class Allocator>
    void Tensor4D<T, Allocator>::set_diagonal(const Vector<T, Allocator> &diag)
    {
      size_t min_dim = std::min(std::min(n_1_, n_2_), std::min(n_3_, n_4_));
      if (diag.size() != min_dim)
      {
         throw std::runtime_error("The vector given to set the diagonal doesn't correspond to the minimal dimension.");
      }
      size_t stride = n_1_ + n_1_ * n_2_ + n_1_ * n_2_ * n_3_;
      cpu::CopyVectors(diag.size(), diag.data(), 1, this->data(), stride+1);
    }

    // QC functions
    template<typename T, class Allocator>
    Matrix<T, Allocator> Tensor4D<T, Allocator>::get_coulomb_eri() const
    {
        return Matrix<T, Allocator>(*this);
    }

    template<typename T, class Allocator>
    Matrix<T, Allocator> Tensor4D<T, Allocator>::get_exchange_eri() const
    {
        Matrix<T, Allocator> eri_exchange_full(Shape(n_1_*n_2_, n_3_*n_4_), 0.0);
        T* exchange_ptr = eri_exchange_full.data();
    
        for (size_t col = 0; col < n_3_*n_4_; ++col) {
            size_t k = col % n_4_;
            size_t l = col / n_4_;
            size_t base_eri_idx = l*n_1_ + k*n_1_*n_2_;
            
            cpu::CopyVectors(n_1_, this->data() + base_eri_idx, 1, exchange_ptr + col*n_1_*n_2_, 1);
            cpu::CopyVectors(n_1_, this->data() + base_eri_idx + n_1_*n_2_*n_3_, 1, exchange_ptr + n_1_ + col*n_1_*n_2_, 1);
        }
        return eri_exchange_full;
    }


    } // namespace cpu
} // namespace lahva
