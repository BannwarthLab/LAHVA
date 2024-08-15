#include "linalg.hpp"
#include <climits>
#include <cmath>
#include <assert.h>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include "runtime.hpp"
#ifdef _CUDA
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "utils/utils.hpp"
#endif

namespace tcgmtensor
{

    ///////////////////////////////////////////////////////////////////////////
    // Vector class
    ///////////////////////////////////////////////////////////////////////////
    template<typename T>
    Vector<T>::Vector( size_type count ) : n_entries_{count}, data_{new T[count]}
    {   
    };

    template<typename T>
    Vector<T>::Vector( size_type count, const T& value) : Vector{count}
    {
        std::fill(data_, data_ + count, value);
    };

    template<typename T>
    Vector<T>::Vector(size_type count, T* ptr, bool take_ownership) : 
    n_entries_{count}, data_{ptr}, is_owner_{take_ownership}
    {

    };

    template <typename T>
    Vector<T>::Vector(size_type count, const T *ptr) : Vector{count}
    {   
        std::copy(ptr, ptr + count, data_);
    };

    
    template <typename T>
    Vector<T>::Vector(const Vector& x) : Vector{x.size()}
    {
        std::copy(x.data(), x.data() + x.size(), data_);
    };
    
    
    template <typename T>
    Vector<T> &Vector<T>::operator=(const Vector& other) {
    if (this != &other)
    {
        if (this->is_owner_) delete[] data_;
        data_ = new T[other.size()];
        is_owner_ = true;
        n_entries_ = other.size();
        std::copy(other.data(), other.data() + other.size(), data_);
        
        if (this->device_ptr_) this->device_ptr_.reset(allocate<T>(this->size()));
      }
      return *this;
      };
    
    template <typename T>
    Vector<T> &Vector<T>::operator=(Vector&& other) 
    {
        if (this != &other)
        {
        this->data_ = other.data_;
        
        if (n_entries_ == other.size())
        {
            this->device_ptr_ = std::move(other.device_ptr_);
            this->is_on_device_ = other.is_on_device_;
        }
        else
        {
            this->device_ptr_.reset();
            this->is_on_device_ = false;
        }

        this->n_entries_ = other.n_entries_;
        other.data_ = nullptr;
        other.n_entries_ = 0;
        this->is_owner_ = other.is_owner_;

        other.is_owner_ = false;
        other.is_on_device_ = false;
        }
        return *this;
    };
    

    template <typename T>
    Vector<T>::~Vector(){
        if (is_owner_)
        {
            delete[] data_;
        }
    };

    template <typename T>
    void Vector<T>::print() const
    {
        for (uint i = 0; i < this->size(); i++)
        {
            printf("%f, ", this->data()[i]);
            printf("\n");
        }
    }

    

    

    ///////////////////////////////////////////////////////////////////////////
    // Matrix class
    ///////////////////////////////////////////////////////////////////////////

    template <typename T>
    void Matrix<T>::check_size_(size_t n_rows,
                                size_t n_cols)
    {
        if (n_cols > UINT_MAX)
        {
            throw std::out_of_range("Number of columns exceeds maximum matrix size.");
        }
        else if (n_rows > UINT_MAX)
        {
            throw std::out_of_range("Number of rows exceeds maximum matrix size.");
        }
        else if (data_size_(n_rows, n_cols) > SIZE_MAX)
        {
            throw std::out_of_range("Vector exceeds maximum matrix size.");
        }
    }

    template <typename T>
    Matrix<T>::Matrix(const Shape &shape) : n_rows_{shape.first}, n_cols_{shape.second},
                                            data_{new T[data_size_(n_rows_, n_cols_)]}
    {
        check_size_(shape.first, shape.second);
    }

    template <typename T>
    Matrix<T>::Matrix(const Shape &shape, T val) : Matrix(shape)
    {
        std::fill(data_, data_ + data_size_(n_rows_, n_cols_), val);
    }

    template <typename T>
    Matrix<T>::Matrix(uint n) : Matrix<T>::Matrix(Shape(n, n)) {}

    template <typename T>
    Matrix<T>::Matrix(uint n, T val) : Matrix<T>::Matrix(Shape(n, n), val) {}

    template <typename T>
    Matrix<T>::Matrix(const Shape &shape, const T *data) : Matrix<T>::Matrix(shape)
    {
        std::copy(data, data + data_size_(n_rows_, n_cols_), data_);
    };

    template <typename T>
    Matrix<T>::Matrix(const tcgmtensor::Vector<tcgmtensor::Vector<T>> &data) : n_rows_{(uint)(data.size())},
                                                                               n_cols_{(uint)(data[0].size())},
                                                                               data_{new T[data_size_(n_rows_, n_cols_)]}
    {
        check_size_(data.size(), data[0].size());

        for (uint i = 0; i < n_rows_; i++)
        {
            if (data[i].size() != n_cols_)
            {
                throw std::logic_error("All rows must have the same length.");
            }

            for (uint j = 0; j < n_cols_; j++)
            {
                data_[data_id_(i, j)] = data[i][j];
            }
        }
    }

    template <typename T>
    Matrix<T>::~Matrix()
    {
        if (this->is_owner_ && data_ != nullptr)
        {
            delete[] data_;
        }
    }

    // copy operations
    template <typename T>
    Matrix<T>::Matrix(const Matrix<T> &other) : n_rows_{other.n_rows_},
                                                n_cols_{other.n_cols_},
                                                data_{new T[data_size_(n_rows_, n_cols_)]}
    {
        std::copy(other.data_, other.data_ + data_size_(n_rows_, n_cols_), data_);
        this->copyGPUTensor(other);
    }

    template <typename T>
    Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other)
    {
        if (this != &other)
        {   
            if (is_owner_) delete[] data_;
            data_ = new T[data_size_(other.n_rows_, other.n_cols_)];
            is_owner_ = true;
            n_rows_ = other.n_rows_;
            n_cols_ = other.n_cols_;
            std::copy(other.data_, other.data_ + data_size_(n_rows_, n_cols_), data_);
            this->copyGPUTensor(other);
        }

        return *this;
    }

    // move operations
    template <typename T>
    Matrix<T>::Matrix(Matrix<T> &&other) : n_rows_{other.n_rows_},
                                           n_cols_{other.n_cols_}
    {
        this->data_ = other.data_;

        if (n_cols_*n_rows_ == other.n_cols_*other.n_rows_)
        {
            this->device_ptr_ = std::move(other.device_ptr_);
            this->is_on_device_ = other.is_on_device_;
        }
        else
        {
            this->device_ptr_.reset();
            this->is_on_device_ = false;
        }
    
        other.data_ = nullptr;
        other.n_rows_ = 0;
        other.n_cols_ = 0;
        this->is_owner_ = other.is_owner_;

        other.is_owner_ = false;
        other.is_on_device_ = false;
    }

    template <typename T>
    Matrix<T> &Matrix<T>::operator=(Matrix<T> &&other)
    {
        if (this != &other)
        {
            
            if (is_owner_ and data_ != nullptr)
            {
                delete[] data_;
            }
            data_ = other.data_;
            n_rows_ = other.n_rows_;
            n_cols_ = other.n_cols_;
            is_owner_ = other.is_owner_;
            
            if (n_cols_*n_rows_ == other.n_cols_*other.n_rows_)
            {
                this->device_ptr_ = std::move(other.device_ptr_);
                this->is_on_device_ = other.is_on_device_;
            }
            else
            {
                this->device_ptr_.reset();
                this->is_on_device_ = false;
            }
            other.is_on_device_ = false;

            other.data_ = nullptr;
            other.n_rows_ = 0;
            other.n_cols_ = 0;
            other.is_owner_ = false;            
        }
        return *this;
    }

    template <typename T>
    T &Matrix<T>::operator()(uint i, uint j)
    {
        return data_[data_id_(i, j)];
    }

    template <typename T>
    const T &Matrix<T>::operator()(uint i, uint j) const
    {
        return data_[data_id_(i, j)];
    }

    template <typename T>
    Matrix<T> &Matrix<T>::operator+=(T val)
    {
        #pragma omp for 
        for (size_t i = 0; i < data_size_(n_rows_, n_cols_); i++)
        {
            data_[i] += val;
        }

        return *this;
    }

    template <typename T>
    void Matrix<T>::print() const
    {
        for (uint i = 0; i < n_rows_; i++)
        {
            for (uint j = 0; j < n_cols_; j++)
            {
                printf("%f, ", data_[data_id_(i, j)]);
            }
            printf("\n");
        }
    }

    //! @param A matrix to symmetrize
    //! @return (A+A^T)/2
    template <typename T>
    void Matrix<T>::symmetrize()
    {
        assert(n_cols_ == n_rows_);

        Matrix<T> copy = *this;
        #pragma omp parallel for shared(data_, copy)
        for (uint i = 0; i < n_cols_; i++)
        {
            for (uint j = 0; j < n_cols_; j++)
            {
                data_[data_id_(i, j)] = 0.5 * (copy(i, j) + copy(j, i));
            }
        }
    }

    template <typename T>
    Vector<T> Matrix<T>::get_diagonal() const
    {
        size_t min_dim = std::min(n_cols_, n_rows_);
        Vector<T> diag(min_dim);
        #pragma omp parallel for shared(diag,data_) 
        for (size_t i = 0; i < min_dim; i++)
        {
            diag[i] = (data_[data_id_(i, i)]);
        }

        return diag;
    }

    template <typename T>
    void Matrix<T>::set_diagonal(const Vector<T> &diag)
    {
        size_t min_dim = std::min(n_cols_, n_rows_);
        #pragma omp parallel for shared(diag,data_) 
        for (size_t i = 0; i < min_dim; i++)
        {
            data_[data_id_(i, i)] = diag[i];
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Lower Triangular Matrix class
    ///////////////////////////////////////////////////////////////////////////

    template <typename T>
    void LowTriMatrix<T>::check_size_(long unsigned int n)
    {
        if (n > UINT_MAX || n > std::sqrt(SIZE_MAX))
        {
            throw std::out_of_range("Vector exceeds maximum LowTriMatrix size.");
        }
    }

    template <typename T>
    LowTriMatrix<T>::LowTriMatrix(uint size) : n_{size},
                                               data_{new T[data_size_(size)]}
    {
        check_size_(size);
    }

    template <typename T>
    LowTriMatrix<T>::LowTriMatrix(uint size, T val) : LowTriMatrix(size)
    {
        check_size_(size);

        std::fill(data_, data_ + data_size_(size), val);
    }

    template <typename T>
    LowTriMatrix<T>::LowTriMatrix(const Shape &shape) : LowTriMatrix<T>::LowTriMatrix(shape.first)
    {
        if (shape.first != shape.second)
        {
            throw std::logic_error("shape.first must be equal to shape.second.");
        }
    }

    template <typename T>
    LowTriMatrix<T>::LowTriMatrix(const Shape &shape, T val) : LowTriMatrix<T>::LowTriMatrix(shape.first, val)
    {
        if (shape.first != shape.second)
        {
            throw std::logic_error("shape.first must be equal to shape.second.");
        }
    }

    template <typename T>
    LowTriMatrix<T>::LowTriMatrix(const tcgmtensor::Vector<tcgmtensor::Vector<T>> &data) : n_{static_cast<uint>(data.size())},
                                                                                           data_{new T[data_size_(data.size())]}
    {
        check_size_(data.size());

        for (uint i = 0; i < n_; i++)
        {
            if (data[i].size() != i + 1)
            {
                throw std::logic_error("Row " + std::to_string(i) + " has wrong length.");
            }
            #pragma omp for
            for (uint j = 0; j <= i; j++)
            {
                data_[data_id_(i, j)] = data[i][j];
            }
        }
    }

    template <typename T>
    LowTriMatrix<T>::LowTriMatrix(uint n, const tcgmtensor::Vector<T> &data) : n_{n},
                                                                               data_{new T[data.size()]}
    {
        check_size_(data.size());
        #pragma omp for
        for (uint i = 0; i < data_size_(n_); i++)
        {
            data_[i] = data[i];
        }
    }

    template <typename T>
    LowTriMatrix<T>::~LowTriMatrix()
    {
        delete[] data_;
    }

    // copy operations
    template <typename T>
    LowTriMatrix<T>::LowTriMatrix(const LowTriMatrix<T> &other) : n_{other.n_},
                                                                  data_{new T[data_size_(other.n_)]}
    {
        std::copy(other.data_, other.data_ + data_size_(n_), data_);
    }

    template <typename T>
    LowTriMatrix<T> &LowTriMatrix<T>::operator=(const LowTriMatrix<T> &other)
    {
        if (this != &other)
        {
            delete[] data_;
            data_ = new T[data_size_(other.n_)];

            n_ = other.n_;
            std::copy(other.data_, other.data_ + data_size_(n_), data_);
        }

        return *this;
    }

    // move operations
    template <typename T>
    LowTriMatrix<T>::LowTriMatrix(LowTriMatrix<T> &&other) : n_{other.n_}
    {
        this->data_ = other.data_;

        other.data_ = nullptr;
        other.n_ = 0;
    }

    template <typename T>
    LowTriMatrix<T> &LowTriMatrix<T>::operator=(LowTriMatrix<T> &&other)
    {
        if (this != &other)
        {
            delete[] data_;

            data_ = other.data_;
            n_ = other.n_;

            other.data_ = nullptr;
            other.n_ = 0;
        }

        return *this;
    }

    template <typename T>
    T &LowTriMatrix<T>::operator()(uint i, uint j)
    {
        if (i < j)
        {
            uint tmp = i;
            i = j;
            j = tmp;
        }
        return data_[data_id_(i, j)];
    }

    template <typename T>
    const T &LowTriMatrix<T>::operator()(uint i, uint j) const
    {
        if (i < j)
        {
            uint tmp = i;
            i = j;
            j = tmp;
        }
        return data_[data_id_(i, j)];
    }

    template <typename T>
    void LowTriMatrix<T>::print() const
    {
        for (uint i = 0; i < n_; i++)
        {
            for (uint j = 0; j <= i; j++)
            {
                std::cout << data_[data_id_(i, j)];
            }
            printf("\n");
        }
    }

    template <typename T>
    Vector<T> LowTriMatrix<T>::get_diagonal() const
    {
        size_t min_dim = n_;
        Vector<T> diag(min_dim);

        for (size_t i = 0; i < min_dim; i++)
        {
            diag[i] = (data_[data_id_(i, i)]);
        }

        return diag;
    }

    template <typename T>
    void LowTriMatrix<T>::set_diagonal(Vector<T> &diag)
    {
        size_t min_dim = n_;
        #pragma omp for
        for (size_t i = 0; i < min_dim; i++)
        {
            data_[data_id_(i, i)] = diag[i];
        }
    }

    template class Matrix<double>;
    template class Matrix<float>;
    template class Matrix<int>;
    template class LowTriMatrix<double>;
    template class LowTriMatrix<float>;
    template class LowTriMatrix<int>;
    template class Vector<float>;
    template class Vector<double>;
    template class Vector<int>;

} // namespace sqmbox
