#ifndef LAHVA_ADD_LEVEL1_CPU_HPP
#define LAHVA_ADD_LEVEL1_CPU_HPP
#include "linalg.hpp"
#include "const.h"
#include "level1.h"

namespace lahva{
    namespace cpu
    {
        
        template<typename T>
        double ComputeTrace(const CPURuntime& cudart, const Matrix_<T>&, bool use_diag = false);
        double ComputeTrace(const CPURuntime& cudart, const Vector_<double>& diag);

        template<typename T>
        T FrobeniusNorm(const Tensor<T>& mat);

        template<typename T>
        T FrobeniusNorm(const Tensor<T>& mat, const Tensor<T>& mat2);

        template<typename T>
        T FrobeniusNorm(const CPURuntime& cudart, const Tensor<T>& mat, const Tensor<T>& mat2)
        {
            return FrobeniusNorm(mat, mat2);
        };

        template<typename T>
        T FrobeniusNorm(const CPURuntime& cudart, const Tensor<T>& mat);

        template<typename T>
        T FrobeniusInnerProduct(const Matrix_<T>& mat1, const Matrix_<T>& mat2);
        template<typename T>
        T FrobeniusInnerProduct(const CPURuntime& rt, const Matrix_<T>& mat1, const Matrix_<T>& mat2);

        template<typename T, typename U>
        void GetDiagonal(const CPURuntime& cudart, const Matrix<T, U>& mat, Vector<T,U>& vec)
        {
            vec = mat.get_diagonal();
        };

        template<typename T, typename U>
        void SetDiagonal(const CPURuntime& cudart, const Vector<T, U>& vec, Matrix<T, U>& m)
        {
            m.set_diagonal(vec);
        }; 
#ifdef _CUDA
        template<typename T, typename U, typename V>
        void GetDiagonal(const CPURuntime& cudart, const gpu::Matrix<T, U, V>& mat, gpu::Vector<T, U, V>& vec)
        {
            size_t max_dim = std::max(mat.shape().first, mat.shape().second);
            cpu::CopyVectors(vec.size(), mat->data(), max_dim+1 ,vec.data(), 1);
        };
        template<typename T, typename U, typename V>
        void GetDiagonal(const gpu::Matrix<T, U, V>& mat, gpu::Vector<T, U, V>& vec)
        {
            size_t max_dim = std::max(mat.shape().first, mat.shape().second);
            cpu::CopyVectors(vec.size(), mat.data(), max_dim+1 ,vec.data(), 1);
        };


        template<typename T, typename U, typename V>
        void SetDiagonal(const CPURuntime& cudart, const gpu::Vector<T, U, V>& vec, gpu::Matrix<T, U, V>& m)
        {
            m.set_diagonal(vec);
        };  

#endif
        template<typename T>
        using func_t1D = T (*) (T);

        template<typename T>
        void ApplyKernel(func_t1D<T> funcPtr , Tensor<T>& mat)
        {
            #pragma omp parallel for collapse(1)
            for (size_t i = 0; i < mat.size(); i++)
            {
                mat.data()[i] = funcPtr(mat.data()[i]);
            }
        };

        template<typename T>
        using func_t0D = T (*) ();

        template<typename T>
        void ApplyKernel(func_t0D<T> funcPtr , Tensor<T>& mat)
        {
            #pragma omp parallel for collapse(1)
            for (size_t i = 0; i < mat.size(); i++)
            {
                mat.data()[i] = funcPtr();
            }
        };

        template<typename... Args>
        void ApplyKernel(const CPURuntime& rt_, Args&&... args) {
            (ApplyKernel(args...));                    
        }
        

       
    } // namespace gpu
    
}   
#endif