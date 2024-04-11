#include "linalg.hpp"
#include <climits>
#include <cmath>
#include <assert.h>
#include <stdexcept>
#include <iostream>
#include "runtime.hpp"
#ifdef _CUDA
  #include "cublas_v2.h"
  #include "cuda_runtime.h"
  #include "utils/utils.hpp"
#endif

namespace tcgmtensor {

///////////////////////////////////////////////////////////////////////////
// Vector class
///////////////////////////////////////////////////////////////////////////
  
  template<typename T>
  Vector<T>::~Vector()
  {

  };

template<typename T>
void Vector<T>::print() const {
  for (uint i = 0; i < this->size(); i++) {
      printf("%f, ", this->data()[i]);
    printf("\n");
  }
}  


///////////////////////////////////////////////////////////////////////////
// Matrix class
///////////////////////////////////////////////////////////////////////////

template<typename T>
void Matrix<T>::check_size_(size_t n_rows, 
      size_t n_cols) {
  if (n_cols > UINT_MAX) {
    throw std::out_of_range("Number of columns exceeds maximum matrix size.");
  } else if (n_rows > UINT_MAX) {
    throw std::out_of_range("Number of rows exceeds maximum matrix size.");
  } else if (data_size_(n_rows, n_cols) > SIZE_MAX) {
    throw std::out_of_range("Vector exceeds maximum matrix size.");
  }
}

template<typename T>
Matrix<T>::Matrix(const Shape& shape) : n_rows_{shape.first}, n_cols_{shape.second},
    data_{new T[data_size_(n_rows_, n_cols_)]}{
  check_size_(shape.first, shape.second);
  
}

template<typename T>
Matrix<T>::Matrix(const Shape& shape, T val) : Matrix(shape) {
  for (size_t i = 0; i < data_size_(n_rows_, n_cols_); i++) {
    data_[i] = val;
  }
}

template<typename T>
Matrix<T>::Matrix(uint n) : Matrix<T>::Matrix(Shape(n,n)) {}

template<typename T>
Matrix<T>::Matrix(uint n, T val) : Matrix<T>::Matrix(Shape(n,n), val) {}

template<typename T>
Matrix<T>::Matrix(const tcgmtensor::Vector<tcgmtensor::Vector<T>>& data) : 
    n_rows_{(uint) (data.size())},
    n_cols_{(uint) (data.at(0).size())},
    data_{new T[data_size_(n_rows_, n_cols_)]} {
  check_size_(data.size(), data.at(0).size());

  for (uint i = 0; i < n_rows_; i++) {
    if (data[i].size() != n_cols_){
      throw std::logic_error("All rows must have the same length.");
    }

    for (uint j = 0; j < n_cols_; j++) {
      data_[data_id_(i,j)] = data[i][j];
    }
  }
}

template<typename T>
Matrix<T>::~Matrix() {
  if (this->is_owner_){
    delete[] data_;
  }
}

// copy operations
template<typename T>
Matrix<T>::Matrix(const Matrix<T>& other) : n_rows_{other.n_rows_}, 
      n_cols_{other.n_cols_},
      data_{new T[data_size_(n_rows_, n_cols_)]} {
  std::copy(other.data_, other.data_ + data_size_(n_rows_, n_cols_), data_);
  this->device_ptr_.reset(allocate<T>(n_cols_*n_rows_));
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other) {
  if (this != &other) {
    delete[] data_;
    data_ = new T[data_size_(other.n_rows_, other.n_cols_)];

    n_rows_ = other.n_rows_;
    n_cols_ = other.n_cols_;
    std::copy(other.data_, other.data_ + data_size_(n_rows_, n_cols_), data_);
    this->device_ptr_.reset(allocate<T>(n_cols_*n_rows_));
  }

  return *this;
}

// move operations
template<typename T>
Matrix<T>::Matrix(Matrix<T>&& other) : n_rows_{other.n_rows_}, 
      n_cols_{other.n_cols_}  {
  this->data_ = other.data_;

  other.data_ = nullptr;
  other.n_rows_ = 0;
  other.n_cols_ = 0;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& other) {
  if (this != &other) {
    delete[] data_;

    data_ = other.data_;
    n_rows_ = other.n_rows_;
    n_cols_ = other.n_cols_;

    other.data_ = nullptr;
    other.n_rows_ = 0;
    other.n_cols_ = 0;
    this->device_ptr_.reset(allocate<T>(n_cols_*n_rows_));
  }

  return *this;
}

template<typename T>
T& Matrix<T>::operator()(uint i, uint j) {
  return data_[data_id_(i,j)];
}

template<typename T>
const T& Matrix<T>::operator()(uint i, uint j) const {
  return data_[data_id_(i,j)];
}


template<typename T>
Matrix<T>& Matrix<T>::operator+=(T val) {
  for (size_t i = 0; i < data_size_(n_rows_, n_cols_); i++) {
    data_[i] += val;
  }

  return *this;
}

template<typename T>
void Matrix<T>::print() const {
  for (uint i = 0; i < n_rows_; i++) {
    for(uint j = 0; j < n_cols_; j++) {
      printf("%f, ", data_[data_id_(i,j)]);
    }
    printf("\n");
  }
}

template<typename T>
Vector<T> Matrix<T>::get_diagonal() const 
{
  size_t min_dim = min(n_cols_, n_rows_);
  Vector<T> diag(min_dim);

  for (size_t i =0 ; i < min_dim; i++)
  {
    diag.push_back(data_[data_id_(i,i)]);
  }

  return diag;
}

template<typename T>
void Matrix<T>::set_diagonal(Vector<T>& diag)
{
  size_t min_dim = min(n_cols_, n_rows_);

  for (size_t i = 0 ; i < min_dim; i++)
  {
    data_[data_id_(i,i)] = diag(i);
  }
}

  template<typename T>
  const void GPUTensor<T>::copy2device(const CudaRuntime& cudart) const
  {
#ifdef _CUDA
    cudaError_t stat_;
    stat_ = cudaSetDevice(cudart.device_id());
    get_cuda_error(stat_);

    this->device_ptr_.reset(allocate<T>(this->size()));
    cudaError_t stat = cudaMemcpy(this->device_ptr_.get(), this->data(), this->size()*sizeof(T), cudaMemcpyHostToDevice);
    get_cuda_error(stat);
    this->is_on_device_ = true;
#endif  
  };
  
  template<typename T>
  void GPUTensor<T>::copy2host(const CudaRuntime& cudart)
  {
#ifdef _CUDA 
    cudaError_t stat_;
    stat_ = cudaSetDevice(cudart.device_id());
    get_cuda_error(stat_);
    if (this->is_on_device_)
    {
      stat_ = cudaMemcpy(this->data(), this->device_ptr_.get(), this->size()*sizeof(T), cudaMemcpyDeviceToHost);
      get_cuda_error(stat_);
      this->is_on_device_ = false;
    }
#endif  
  };

///////////////////////////////////////////////////////////////////////////
// Lower Triangular Matrix class
///////////////////////////////////////////////////////////////////////////

template<typename T>
void LowTriMatrix<T>::check_size_(long unsigned int n) {
  if (n > UINT_MAX || n > std::sqrt(SIZE_MAX)) {
    throw std::out_of_range("Vector exceeds maximum LowTriMatrix size.");
  }
}

template<typename T>
LowTriMatrix<T>::LowTriMatrix(uint size) : n_{size}, 
    data_{new T[data_size_(size)]} {
  check_size_(size);
}

template<typename T>
LowTriMatrix<T>::LowTriMatrix(uint size, T val) : LowTriMatrix(size) {
  check_size_(size);

  for (size_t i = 0; i < data_size_(size); i++) {
    data_[i] = val;
  }
}

template<typename T>
LowTriMatrix<T>::LowTriMatrix(const Shape& shape) : 
    LowTriMatrix<T>::LowTriMatrix(shape.first) {
  if (shape.first != shape.second) {
    throw std::logic_error("shape.first must be equal to shape.second.");
  }
}

template<typename T>
LowTriMatrix<T>::LowTriMatrix(const Shape& shape, T val) : 
    LowTriMatrix<T>::LowTriMatrix(shape.first, val) {
  if (shape.first != shape.second) {
    throw std::logic_error("shape.first must be equal to shape.second.");
  }
}

template<typename T>
LowTriMatrix<T>::LowTriMatrix(const tcgmtensor::Vector<tcgmtensor::Vector<T>>& data) : 
    n_{static_cast<uint>(data.size())},
    data_{new T[data_size_(data.size())]} {
  check_size_(data.size());

  for (uint i = 0; i < n_; i++) {
    if (data[i].size() != i+1){
      throw std::logic_error("Row " + std::to_string(i) + " has wrong length.");
    }

    for (uint j = 0; j <= i; j++) {
      data_[data_id_(i,j)] = data[i][j];
    }
  }
}

template<typename T>
LowTriMatrix<T>::LowTriMatrix(uint n, const tcgmtensor::Vector<T>& data) : 
    n_{n},
    data_{new T[data.size()]} {
    check_size_(data.size());
  
    for (uint i = 0; i < data_size_(n_); i++) {
        data_[i] = data[i];
    }
}

template<typename T>
LowTriMatrix<T>::~LowTriMatrix() {
  delete[] data_;
}

// copy operations
template<typename T>
LowTriMatrix<T>::LowTriMatrix(const LowTriMatrix<T>& other) : 
    n_{other.n_}, 
    data_{new T[data_size_(other.n_)]} {
  std::copy(other.data_, other.data_ + data_size_(n_), data_);
}

template<typename T>
LowTriMatrix<T>& LowTriMatrix<T>::operator=(const LowTriMatrix<T>& other) {
  if (this != &other) {
    delete[] data_;
    data_ = new T[data_size_(other.n_)];

    n_ = other.n_;
    std::copy(other.data_, other.data_ + data_size_(n_), data_);
  }

  return *this;
}

// move operations
template<typename T>
LowTriMatrix<T>::LowTriMatrix(LowTriMatrix<T>&& other) : 
    n_{other.n_} {
  this->data_ = other.data_;

  other.data_ = nullptr;
  other.n_ = 0;
}

template<typename T>
LowTriMatrix<T>& LowTriMatrix<T>::operator=(LowTriMatrix<T>&& other) {
  if (this != &other) {
    delete[] data_;

    data_ = other.data_;
    n_ = other.n_;

    other.data_ = nullptr;
    other.n_ = 0;
  }

  return *this;
}

template<typename T>
T& LowTriMatrix<T>::operator()(uint i, uint j) {
  return data_[data_id_(i,j)];
}

template<typename T>
const T& LowTriMatrix<T>::operator()(uint i, uint j) const {
  return data_[data_id_(i,j)];
}


template<typename T>
void LowTriMatrix<T>::print() const {
  for (uint i = 0; i < n_; i++) {
    for(uint j = 0; j <= i; j++) {
      std::cout << data_[data_id_(i,j)] ;
    }
    printf("\n");
  }
}


template<typename T>
Vector<T> LowTriMatrix<T>::get_diagonal() const 
{
  size_t min_dim = min(n_cols_, n_rows_);
  Vector<T> diag(min_dim);

  for (size_t i =0 ; i < min_dim; i++)
  {
    diag.push_back(data_[data_id_(i,i)]);
  }

  return diag;
}

template<typename T>
void LowTriMatrix<T>::set_diagonal(Vector<T>& diag)
{
  size_t min_dim = min(n_cols_, n_rows_);

  for (size_t i = 0 ; i < min_dim; i++)
  {
    data_[data_id_(i,i)] = diag(i);
  }
}

template class Matrix<double>;
template class Matrix<float>;
template class LowTriMatrix<double>;
template class LowTriMatrix<float>;
template class Vector<float>;
template class Vector<double>;
template class GPUTensor<double>;
template class GPUTensor<float>;

} // namespace sqmbox

