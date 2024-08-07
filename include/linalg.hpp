#ifndef TCGMTENSOR_LINALG_H
#define TCGMTENSOR_LINALG_H

#include <memory>
#include <vector>
#include <iterator>
#include <assert.h>
#pragma warning(disable : 2282)
#pragma warning(disable : 815)
#pragma warning(disable : 858)
#include <iostream>
#include "impl/tensor.hpp"

namespace tcgmtensor
{

  typedef unsigned int uint;
  typedef unsigned short ushort;

  template <class T>
  class Vector : virtual public GPUTensor<T>
  {

  protected:
    size_t n_entries_ = 0;
    size_t inc_ = 1;
    bool is_owner_ = true;
    T *data_ = nullptr;

  public:
    using size_type = std::size_t;
    using Allocator = std::allocator<T>;

    Vector() {};
    Vector(size_type count);
    Vector(const Vector &x);
    Vector(size_type count,
           const T &value);
    Vector(size_type count, T *ptr, bool take_onwership = true);
    Vector(size_type count, const T *ptr);
    ~Vector();
    Vector &operator=(const Vector &other);
    Vector &operator=(Vector &&other);
    template<typename D>
    T &operator[](D index) const
    {
      return data_[static_cast<size_t>(index)];
    };
    template<typename D>
    T &operator[](D index)
    {
      return data_[static_cast<size_t>(index)];
    };
    

    T *data() override { return this->data_; }

    const T *data() const override { return this->data_; }

    size_t size() override { return n_entries_; }

    const size_t size() const override { return n_entries_;}

    T* begin() const {return &data_[0];};
    T* end() const {return &data_[n_entries_-1];};

    T sum() const
    {
      T res = (T)0;
    
      for (size_t i = 0; i < this->size(); i++)
      {
        res += this->data_[i];
      }
      return res;
    }
    void print() const;

    T sum()
    {
      T res = (T)0;
      for (size_t i = 0; i < this->size(); i++)
      {
        res += this->data_[i];
      }
      return res;
    }
  };

  //! Shape of a matrix
  using Shape = std::pair<uint, uint>;

  //! @brief slim wrapper around a float or double array to allow easy acces with
  //!        two indeces using the () operator.
  //!
  //! If NDEBUG is **not** defined, range checks are performed.  d
  //! The data is stored in column-major order in a 1D array.
  //!
  template <typename T>
  class Matrix : public GPUTensor<T>
  {

  protected:
    // shape in each dimension, i.e. data_ has length n_rows_*n_cols
    uint n_rows_ = 0;
    uint n_cols_ = 0;
    T *data_ = nullptr;

    // indicates whether the Matrix object owns the data and consequently is
    // responsible for freeing it
    bool is_owner_ = true;

    inline size_t data_id_(uint i, uint j) const
    {
      // deactivated if NDEBUG is defined
      assert(i < n_rows_ && j < n_cols_);

      // range checks are perfomred in constructor and above (if in debug mode)
      return i + j * n_rows_;
    }

    // length of the array data_
    static size_t data_size_(uint n_rows, uint n_cols)
    {
      return n_rows * n_cols;
    }

    // raises an error, if the shape is not valid
    // Vector uses size_t as shape, so the check shape function has to be able
    // to deal with that
    static void check_size_(size_t, size_t);

  public:
    Matrix() {};
    //! construct a square matrix with dimensions n x n
    //! It is not guaranteed that the values will be initialized
    Matrix(uint n);
    //! construct a square n x n matrix initialized with value val
    Matrix(uint n, T val);

    //! construct a matrix with dimensions shape.first x shape.second
    //! It is not guaranteed that the values will be initialized!
    Matrix(const Shape &shape);
    //! construct a matrix of shape with initial values val
    Matrix(const Shape &shape, T val);
    //! construct a matrix by giving ownership of the raw data
    //! note: the data needs to be in the format defined by data_id_()
    //! @param take_ownership if false, the data will not be freed, when the
    //!                       Matrix object is destructed. Use with care!
    Matrix(const Shape &shape, T *data, bool take_ownership = true) : n_rows_{shape.first}, n_cols_{shape.second}, data_{data},
                                                                      is_owner_{take_ownership} {}
    Matrix(const Shape &shape, const T *data);

    Matrix(const tcgmtensor::Vector<tcgmtensor::Vector<T>> &data);

    Matrix(const Matrix &);
    Matrix(Matrix &&);
    Matrix &operator=(const Matrix &);
    Matrix &operator=(Matrix &&);
    virtual ~Matrix();

    //! @param[in] i row index
    //! @param[in] j column index
    //! @return reference to matrix element i,j
    T &operator()(uint i, uint j);
    //! @param[in] i row index
    //! @param[in] j column index
    //! @return reference to matrix element i,j
    const T &operator()(uint i, uint j) const;

    //! @brief in-place, scalar addition
    Matrix &operator+=(T val);

    //! @return pointer to contiguous raw data
    T *data() { return data_; }
    //! @return pointer to contiguous raw data
    const T *data() const { return data_; }

    //! @return number of rows/columns of the matrix
    Shape shape() const { return Shape{n_rows_, n_cols_}; }

    //! prints the matrix as string
    void print() const;

    inline size_t size() override { return n_cols_ * n_rows_; };

    const inline size_t size() const { return n_cols_ * n_rows_; };

    Vector<T> get_diagonal() const;

    void set_diagonal(Vector<T> &diag);

    void symmetrize();

    bool ownsData() { return is_owner_; };
  };

  //! slim wrapper around a float or double array to represent square, lower
  //! triangular matrices that can also be used to store symmetric matrices
  //!
  //! The data is stored in column-major order.
  template <typename T>
  class LowTriMatrix : public GPUTensor<T>
  {
  protected:
    // shape in each dimension, i.e. data_ has length n_rows_^2
    uint n_;
    T *data_;
    bool is_on_device_ = false;
    inline size_t data_id_(uint i, uint j) const
    {
      // deactivated if NDEBUG is defined
      assert(i <= n_ && j <= n_);
      // row index is always greater than column index in lower triangle
      assert(i >= j);

      // range checks are perfomred in constructor and above (if in debug mode)
      return (n_ * (j) - (j - 1) * (j) / 2) + (i - j);
    }

    // length of the array data_
    static size_t data_size_(uint n)
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
    LowTriMatrix(uint n);
    //! construct an n x n matrix initialized with value val
    LowTriMatrix(uint n, T val);
    //! give dimension n and data as a Vector conatining the values of the lower traingular Matrix
    LowTriMatrix(uint n, const tcgmtensor::Vector<T> &data);
    //! create a matrix with shape (shape must be square)
    //! It is not guaranteed that the values will be initialized
    LowTriMatrix(const Shape &shape);
    //! create a non-initialized matrix with shape (shape must be square) and
    //! initialize all values to val
    LowTriMatrix(const Shape &shape, T val);
    //! construct a matrix by giving ownership of the raw data
    //! note: the data needs to be in the format defined by data_id_()
    //! @param n number of columns/rows of the matrix
    LowTriMatrix(uint n, T *data) : n_{n},
                                    data_{data} {}

    //! @brief construct a matrix from a 2D Vector
    //!
    //! The first element of data must be a Vector of length one, the second
    //! element must have length two and so on
    LowTriMatrix(const tcgmtensor::Vector<tcgmtensor::Vector<T>> &data);

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
    T &operator()(uint i, uint j);
    //! Provides element acces.
    //! If NDEBUG is **not** defined, range checks are performed and it will be
    //! checked that the first index is greater than or equal to the second.
    //! @param[in] i row index
    //! @param[in] j column index
    //! @return reference to matrix element i,j
    const T &operator()(uint i, uint j) const;
    //! @return pointer to contiguous raw data
    T *data() { return data_; }
    //! @return pointer to contiguous raw data
    const T *data() const { return data_; }

    //! @return number of rows/columns of the matrix
    Shape shape() const { return Shape(n_, n_); }

    void print() const;

    inline size_t size() override { return data_size_(n_); };

    const inline size_t size() const { return data_size_(n_); };

    Vector<T> get_diagonal() const;

    void set_diagonal(Vector<T> &diag);
  };

  //! Computes the dot product of row i of A (assuming it represents a symmetric
  //! matrix in packed form) and column i of B (assuming it to represent a
  //! symmetric matrix as well)
  template <typename T>
  T row_dot_col_symm(const LowTriMatrix<T> &A, const LowTriMatrix<T> &B, uint i)
  {
    assert(A.shape().first == B.shape().first);

    uint n = A.shape().first;
    T result = 0.0;

    for (uint j = 0; j < i; j++)
    {
      result += A(i, j) * B(i, j);
    }

    for (uint j = i; j < n; j++)
    {
      result += A(j, i) * B(j, i);
    }

    return result;
  }

  //! Computes the dot product of row i of A and column i of B
  template <typename T>
  T row_dot_col(const Matrix<T> &A, const Matrix<T> &B, uint i)
  {
    assert(A.shape().first == A.shape().second);
    assert(B.shape().first == B.shape().second);
    assert(A.shape().first == B.shape().first);

    T result = 0.0;

    for (uint j = 0; j < A.shape().first; j++)
    {
      result += A(i, j) * B(j, i);
    }

    return result;
  }

  //! @param A matrix to symmetrize
  //! @return (A+A^T)/2
  template <typename T>
  LowTriMatrix<T> symmetrizeLowTri(const Matrix<T> &A)
  {
    assert(A.shape().first == A.shape().second);

    LowTriMatrix<T> result(A.shape().first);

    for (uint i = 0; i < A.shape().first; i++)
    {
      for (uint j = 0; j <= i; j++)
      {
        result(i, j) = 0.5 * (A(i, j) + A(j, i));
      }
    }

    return result;
  }

  template <typename T>
  void move_Vector_into_Vector(tcgmtensor::Vector<T> &vec1, tcgmtensor::Vector<T> &vec2)
  {
    vec2.insert(vec2.end(), std::make_move_iterator(vec1.begin()), std::make_move_iterator(vec1.end()));
    vec1.erase(vec1.begin(), vec1.end());
  }

  template <typename T>
  void copy_Vector_into_Vector(tcgmtensor::Vector<T> &vec1, tcgmtensor::Vector<T> &vec2)
  {
    vec2.insert(vec2.end(), vec1.begin(), vec1.end());
    // vec1.erase(vec1.begin(),vec1.end());
  }

} // namespace sqmbox

#endif // SQMBOX_LINALG_H