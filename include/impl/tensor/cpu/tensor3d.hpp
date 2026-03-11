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
    class Tensor3D_ : public virtual Tensor<T>
    {
        public:
            virtual Shape shape() const  = 0;

            //! @param[in] i index in dimension 1
            //! @param[in] j index in dimension 2
            //! @param[in] k index in dimension 3
            //! @return reference to Tensor3D element i,j,k
            virtual T &operator()(size_t i, size_t j, size_t k) = 0;
            //! @param[in] i index in dimension 1
            //! @param[in] j index in dimension 2
            //! @param[in] k index in dimension 3
            //! @return reference to Tensor3D element i,j,k
            virtual const T &operator()(size_t i, size_t j, size_t k) const = 0;

    };


    //! @brief slim wrapper around a float or double array to allow easy acces with
    //!        four indices using the () operator.
    //!
    //! If NDEBUG is **not** defined, range checks are performed.
    //! The data is stored in column-major order in a 1D array.
    //!
    template <class T, class Allocator = StdAllocator<T>>
    class Tensor3D : virtual public CPUTensor<T, Allocator>, virtual public Tensor3D_<T>
    {
        using alloc_ptr = CPUAllocator<T>;
    protected:
        // shape in each dimension, i.e. data_ has length n_1_*n_2_*n_3_
        size_t n_1_;
        size_t n_2_;
        size_t n_3_;
        
        // cached non-owning vector view of the flattened tensor data
        Vector<T, Allocator> vec_view_;
        
        // initializes the cached vector view to point to the tensor data
        void init_vec_view_() {
            vec_view_ = Vector<T, Allocator>(this->count_, this->data_, false);
        }
    
        // indicates whether the Tensor3D object owns the data and consequently is
        // responsible for freeing it

        inline size_t data_id_(size_t i, size_t j, size_t k) const
        {
            // deactivated if NDEBUG is defined
            assert(i < this->n_1_ && j < this->n_2_ && k < this->n_3_);

            // range checks are performed in constructor and above (if in debug mode)
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
        Tensor3D() {};
        //! construct a Tensor3D with dimensions n x n x n
        //! It is not guaranteed that the values will be initialized
        Tensor3D(size_t n, const alloc_ptr &alloc = Allocator());
        template<typename U>
        Tensor3D(size_t n, const std::shared_ptr<CPUAllocator<U>> &alloc)
        : Tensor3D<T, Allocator>{n, Allocator(*alloc)} {};
        //! construct a n x n x n Tensor3D initialized with value val
        Tensor3D(size_t n, T val, const alloc_ptr &alloc = Allocator());
        template<typename U>
        Tensor3D(size_t n, T val, const std::shared_ptr<CPUAllocator<U>> &alloc)
        : Tensor3D<T, Allocator>{n, val, Allocator(*alloc)} {};
        //! construct a Tensor3D with dimensions shape.first x shape.second x shape.third
        //! It is not guaranteed that the values will be initialized!
        Tensor3D(const Shape &shape, const alloc_ptr &alloc = Allocator());
        Tensor3D(const Shape &shape, std::initializer_list<T> init, bool row_major = false, const alloc_ptr &alloc = Allocator());
        template<typename U>
        Tensor3D(const Shape& shape, const std::shared_ptr<CPUAllocator<U>> &alloc)
        : Tensor3D<T, Allocator>{shape, Allocator(*alloc)} {};
        //! construct a Tensor3D of shape with initial values val
        Tensor3D(const Shape &shape, T val, const alloc_ptr &alloc = Allocator());
        template<typename U>
        Tensor3D(const Shape &shape, T val, const std::shared_ptr<CPUAllocator<U>> &alloc)
        : Tensor3D<T, Allocator>{shape, val, Allocator(*alloc)} {};
        //! construct a Tensor3D by giving ownership of the raw data
        //! note: the data needs to be in the format defined by data_id_()
        //! @param take_ownership if false, the data will not be freed, when the
        //!                       Matrix object is destructed. Use with care!
        Tensor3D(const Shape &shape, T *data, bool take_ownership = true, const alloc_ptr &alloc = Allocator());
        template<typename U>
        Tensor3D(const Shape &shape, T* data, bool take_ownership = true, const std::shared_ptr<CPUAllocator<U>> &alloc = Allocator())
        : Tensor3D<T, Allocator>{shape, data, take_ownership, Allocator(*alloc)} {};
        Tensor3D(const Shape &shape, const T *data, const alloc_ptr &alloc = Allocator());
        template<typename U>
        Tensor3D(const Shape &shape, T* data, const std::shared_ptr<CPUAllocator<U>> &alloc)
        : Tensor3D<T, Allocator>{shape, data, Allocator(*alloc)} {};

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

        bool ownsData() { return this->is_owner_; };

        //! @brief Returns a const reference to the cached non-owning vector view
        const Vector<T, Allocator>& as_vec() const;
        Vector<T, Allocator>& as_vec();
    };

     template <typename T, class Allocator>
    void Tensor3D<T, Allocator>::check_size_(size_t n_1,
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

    template <typename T, class Allocator>
    Tensor3D<T, Allocator>::Tensor3D(const Shape &shape, const alloc_ptr &alloc) : 
    CPUTensor<T, Allocator>{data_size_(shape.first, shape.second, shape.third), alloc}, n_1_{shape.first}, n_2_{shape.second}, n_3_{shape.third}
    {
        check_size_(shape.first, shape.second, shape.third);
        init_vec_view_();
    }

    template <typename T, class Allocator>
    Tensor3D<T, Allocator>::Tensor3D(const Shape &shape, T *data, bool take_ownership, const alloc_ptr &alloc) : 
    CPUTensor<T, Allocator>{alloc}, n_1_{shape.first}, n_2_{shape.second}, n_3_{shape.third}
    {
        this->data_ = data;
        this->count_ = n_1_ * n_2_ * n_3_;
        this->is_owner_ = take_ownership;
        init_vec_view_();
    }


    template <typename T, class Allocator>
    Tensor3D<T, Allocator>::Tensor3D(const Shape &shape, T val, const alloc_ptr &alloc) : 
    Tensor3D(shape, alloc)
    {
        std::fill(this->data_, this->data_ + data_size_(n_1_, n_2_, n_3_), val);
        init_vec_view_();
    }

    template <typename T, class Allocator>
    Tensor3D<T, Allocator>::Tensor3D(const Shape &shape, std::initializer_list<T> init, bool row_major, const alloc_ptr &alloc) :
    Tensor3D(shape, alloc)
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
                        this->data_[this->data_id_(mu, nu, kappa)] = *(init.begin() + (mu * this->n_2_ * this->n_3_ + nu * this->n_3_ + kappa));
                    }
                }
            }
        }
        else
        {
            std::copy(init.begin(), init.end(), this->data_);
        }
        
        init_vec_view_();
    };
    

    template <typename T, class Allocator>
    Tensor3D<T, Allocator>::Tensor3D(size_t n, const alloc_ptr &alloc) : 
    Tensor3D<T, Allocator>::Tensor3D(Shape(n, n, n), alloc) 
    {}

    template <typename T, class Allocator>
    Tensor3D<T, Allocator>::Tensor3D(size_t n, T val, const alloc_ptr &alloc) : 
    Tensor3D<T, Allocator>::Tensor3D(Shape(n, n, n), val, alloc) {}

    template <typename T, class Allocator>
    Tensor3D<T, Allocator>::Tensor3D(const Shape &shape, const T *data, const alloc_ptr &alloc) : 
    Tensor3D<T, Allocator>::Tensor3D(shape, alloc)
    {
        std::copy(data, data + this->data_size_(n_1_, n_2_, n_3_), this->data_);
        init_vec_view_();
    };

    template <typename T, class Allocator>
    Tensor3D<T, Allocator>::~Tensor3D()
    {
        
    }

    // copy operations
    template <typename T, class Allocator>
    Tensor3D<T, Allocator>::Tensor3D(const Tensor3D<T, Allocator> &other) : 
    CPUTensor<T, Allocator>{other}, 
    n_1_{other.n_1_}, n_2_{other.n_2_}, n_3_{other.n_3_}
    {
        init_vec_view_();
    }

    template <typename T, class Allocator>
    Tensor3D<T, Allocator> &Tensor3D<T, Allocator>::operator=(const Tensor3D<T, Allocator> &other)
    {
        if (this != &other)
        {
            CPUTensor<T, Allocator>::operator=(other);

            n_1_ = other.n_1_;
            n_2_ = other.n_2_;
            n_3_ = other.n_3_;
            init_vec_view_();
        }

        return *this;
    }

    // move operations
    template <typename T, class Allocator>
    Tensor3D<T, Allocator>::Tensor3D(Tensor3D<T, Allocator> &&other) :
    CPUTensor<T, Allocator>{other},
    n_1_{other.n_1_}, n_2_{other.n_2_}, n_3_{other.n_3_}
    {

        other.n_1_ = 0;
        other.n_2_ = 0;
        other.n_3_ = 0;
        
        init_vec_view_();

    }

    template <typename T, class Allocator>
    Tensor3D<T, Allocator> &Tensor3D<T, Allocator>::operator=(Tensor3D<T, Allocator> &&other)
    {
        if (this != &other)
        {
            CPUTensor<T, Allocator>::operator=(std::move(other));
            
            n_1_ = other.n_1_;
            n_2_ = other.n_2_;
            n_3_ = other.n_3_;
            
            init_vec_view_();

            other.n_1_ = 0;
            other.n_2_ = 0;
            other.n_3_ = 0;
        }
        return *this;
    }

    template <typename T, class Allocator>
    T &Tensor3D<T, Allocator>::operator()(size_t i, size_t j, size_t k)
    {
        return this->data_[this->data_id_(i, j, k)];
    }

    template <typename T, class Allocator>
    const T &Tensor3D<T, Allocator>::operator()(size_t i, size_t j, size_t k) const
    {
        return this->data_[this->data_id_(i, j, k)];
    }

    template <typename T, class Allocator>
    Tensor3D<T, Allocator> &Tensor3D<T, Allocator>::operator+=(T val)
    {
    #pragma omp for
        for (size_t i = 0; i < data_size_(n_1_, n_2_, n_3_); i++)
        {
            this->data_[i] += val;
        }

        return *this;
    }

    template <typename T, class Allocator>
    void Tensor3D<T, Allocator>::print() const
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

    template <typename T, class Allocator>
    Vector<T, Allocator> Tensor3D<T, Allocator>::get_diagonal() const
    {
        size_t min_dim = std::min(std::min(n_1_, n_2_), n_3_);
        Vector<T, Allocator> diag(min_dim);
        size_t stride = n_1_ + n_1_ * n_2_;
        cpu::CopyVectors(diag.size(), this->data(), stride+1, diag.data(), 1);

        return diag;
    }

     template <typename T, class Allocator>
    void Tensor3D<T, Allocator>::set_diagonal(const Vector<T, Allocator> &diag)
    {
      size_t min_dim = std::min(std::min(n_1_, n_2_), n_3_);
      if (diag.size() != min_dim)
      {
         throw std::runtime_error("The vector given to set the diagonal doesn't correspond to the minimal dimension.");
      }
      size_t stride = n_1_ + n_1_ * n_2_;
      cpu::CopyVectors(diag.size(), diag.data(), 1, this->data(), stride+1);
    }

    template <typename T, class Allocator>
    const Vector<T, Allocator>& Tensor3D<T, Allocator>::as_vec() const {
        return vec_view_;
    }

    template <typename T, class Allocator>
    Vector<T, Allocator>& Tensor3D<T, Allocator>::as_vec() {
        return vec_view_;
    }

    } // namespace cpu
} // namespace lahva
