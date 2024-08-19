#include "impl/cpu/additional-level1.hpp"
#include <omp.h>
#include "../../utils/utils.hpp"
namespace tcgmtensor
{
    namespace cpu
    {
        /// @brief 
        /// @param cudart 
        /// @param matrix 
        /// @param use_diag 
        /// @return 
        template<>
        double ComputeTrace<double>(const CPURuntime& cudart, const Matrix<double>& matrix, bool use_diag)
        {
            double trace = 0;
            #pragma omp parallel for shared(matrix) reduction(+:trace)
            for(size_t i = 0 ; i < matrix.shape().first ; i++)
                trace+=(double)matrix(i,i);
            return trace;
        };

        template<>
        double ComputeTrace<float>(const CPURuntime& cudart, const Matrix<float>& matrix, bool use_diag)
        {
            double trace = 0;
            #pragma omp parallel for shared(matrix) reduction(+:trace)
            for(size_t i = 0 ; i < matrix.shape().first ; i++)
                trace+=(double)matrix(i,i);
            return trace;
        };

        double ComputeTrace(const CPURuntime& cudart, const Vector<double>& diag)
        {
            double trace = 0;
            #pragma omp parallel for shared(diag) reduction(+:trace)
            for(size_t i = 0 ; i < diag.size() ; i++)
                trace+=(double)diag[i];
            return trace;
        };

        template<typename T>
        T FrobeniusNorm(const Tensor<T>& mat)
        {
            T norm = 0;
            #pragma omp parallel for simd shared(mat) reduction(+:norm)  
            for (int i = 0 ; i < mat.size() ; i++)
            {
                norm += mat.data()[i]*mat.data()[i];
            };
            norm = std::sqrt(norm);
            return norm;
        }

        template<typename T>
        T FrobeniusNorm(const CPURuntime& rt, const Tensor<T>& mat)
        {
            return FrobeniusNorm(mat);
        };

        template<typename T>
        T FrobeniusInnerProduct(const Matrix<T>& mat1, const Matrix<T>& mat2)
        {
            T norm = 0;
            check_equal_size(mat1, mat2);
            #pragma omp parallel for simd shared(mat1,mat2) reduction(+:norm)  
            for (int i = 0 ; i < (mat1.size()) ; i++)
            {
                norm += mat1.data()[i]*mat2.data()[i];
            }
            return norm;
        }

        template<typename T>
        T FrobeniusInnerProduct(const CPURuntime& rt, const Matrix<T>& mat1, const Matrix<T>& mat2)
        {
            return FrobeniusInnerProduct(mat1, mat2);
        };

        template<typename T>
        void GetDiagonal(const CPURuntime& cudart, const Matrix<T>& mat, Vector<T>& vec)
        {
            vec = mat.get_diagonal();
        };

        template<typename T>
        void SetDiagonal(const CPURuntime& cudart, const Vector<T>& vec, Matrix<T>& m)
        {
            m.set_diagonal(vec);
        };  

        template float FrobeniusNorm<float>(const Tensor<float>& mat);
        template double FrobeniusNorm<double>(const Tensor<double>& mat);
        template double FrobeniusNorm<double>(const CPURuntime& rt, const Tensor<double>& mat);
        template float FrobeniusNorm<float>(const CPURuntime& rt, const Tensor<float>& mat);
        template float FrobeniusInnerProduct<float>(const Matrix<float>& mat1, const Matrix<float>& mat2);
        template double FrobeniusInnerProduct<double>(const Matrix<double>& mat1, const Matrix<double>& mat2 );
        template double FrobeniusInnerProduct<double>(const CPURuntime& rt, const Matrix<double>& mat1, const Matrix<double>& mat2);
        template float FrobeniusInnerProduct<float>(const CPURuntime& rt, const Matrix<float>& mat1, const Matrix<float>& mat2);
        template void GetDiagonal<double>(const CPURuntime& cudart, const Matrix<double>& mat, Vector<double>& vec);
        template void GetDiagonal<float>(const CPURuntime& cudart, const Matrix<float>& mat, Vector<float>& vec);
        template void SetDiagonal<double>(const CPURuntime& cudart, const Vector<double>& vec, Matrix<double>& m);
        template void SetDiagonal<float>(const CPURuntime& cudart, const Vector<float>& vec, Matrix<float>& m);


    } // namespace cpu
    
} // namespace tcgmtensor
