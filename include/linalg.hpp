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
#include "impl/tensor/cpu/tensor.hpp"
#ifdef _CUDA
#include "impl/tensor/gpu/vector.hpp"
#include "impl/tensor/gpu/matrix.hpp"
#include "impl/tensor/gpu/lowtrimatrix.hpp"
#endif
#include "impl/tensor/cpu/vector.hpp"
#include "impl/tensor/cpu/matrix.hpp"
#include "impl/tensor/cpu/lowtrimatrix.hpp"
#include "impl/tensor/allocators.hpp"
namespace tcgmtensor
{

    typedef unsigned int uint;
    typedef unsigned short ushort;




    ////! Computes the dot product of row i of A (assuming it represents a symmetric
    ////! matrix in packed form) and column i of B (assuming it to represent a
    ////! symmetric matrix as well)
    //template <typename T>
    //T row_dot_col_symm(const LowTriMatrix<T> &A, const LowTriMatrix<T> &B, uint i)
    //{
        //assert(A.shape().first == B.shape().first);

        //uint n = A.shape().first;
        //T result = 0.0;

        //for (uint j = 0; j < i; j++)
        //{
            //result += A(i, j) * B(i, j);
        //}

        //for (uint j = i; j < n; j++)
        //{
            //result += A(j, i) * B(j, i);
        //}

        //return result;
    //}

    ////! Computes the dot product of row i of A and column i of B
    //template <typename T>
    //T row_dot_col(const Matrix<T> &A, const Matrix<T> &B, uint i)
    //{
        //assert(A.shape().first == A.shape().second);
        //assert(B.shape().first == B.shape().second);
        //assert(A.shape().first == B.shape().first);

        //T result = 0.0;

        //for (uint j = 0; j < A.shape().first; j++)
        //{
            //result += A(i, j) * B(j, i);
        //}

        //return result;
    //}

    ////! @param A matrix to symmetrize
    ////! @return (A+A^T)/2
    //template <typename T>
    //LowTriMatrix<T> symmetrizeLowTri(const Matrix<T> &A)
    //{
        //assert(A.shape().first == A.shape().second);

        //LowTriMatrix<T> result(A.shape().first);

        //for (uint i = 0; i < A.shape().first; i++)
        //{
            //for (uint j = 0; j <= i; j++)
            //{
                //result(i, j) = 0.5 * (A(i, j) + A(j, i));
            //}
        //}

        //return result;
    //}

    //template <typename T>
    //void move_Vector_into_Vector(tcgmtensor::Vector<T> &vec1, tcgmtensor::Vector<T> &vec2)
    //{
        //vec2.insert(vec2.end(), std::make_move_iterator(vec1.begin()), std::make_move_iterator(vec1.end()));
        //vec1.erase(vec1.begin(), vec1.end());
    //}

    //template <typename T>
    //void copy_Vector_into_Vector(tcgmtensor::Vector<T> &vec1, tcgmtensor::Vector<T> &vec2)
    //{
        //vec2.insert(vec2.end(), vec1.begin(), vec1.end());
        //// vec1.erase(vec1.begin(),vec1.end());
    //}

} // namespace sqmbox

#endif // SQMBOX_LINALG_H