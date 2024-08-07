#include "impl/cpu/additional-level1.hpp"

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
            for(size_t i = 0 ; i < matrix.shape().first ; i++)
                trace+=(double)matrix(i,i);
            return trace;
        };

        template<>
        double ComputeTrace<float>(const CPURuntime& cudart, const Matrix<float>& matrix, bool use_diag)
        {
            double trace = 0;
            for(size_t i = 0 ; i < matrix.shape().first ; i++)
                trace+=(double)matrix(i,i);
            return trace;
        };

        double ComputeTrace(const CPURuntime& cudart, const Vector<double>& diag)
        {
            double trace = 0;
            for(size_t i = 0 ; i < diag.size() ; i++)
                trace+=(double)diag[i];
            return trace;
        };

        template<typename T>
        T FrobeniusNorm(const Matrix<T>& mat)
        {
            T norm ; 
            for (int i = 0 ; i < mat.size() ; i++)
            {
                norm += mat.data()[i]*mat.data()[i];
            };
            norm = std::sqrt(norm);
            return norm;
        }

        template<typename T>
        T FrobeniusNorm(const CPURuntime& rt, const Matrix<T>& mat)
        {
            return FrobeniusNorm(mat);
        };

        template float FrobeniusNorm<float>(const Matrix<float>& mat);
        template double FrobeniusNorm<double>(const Matrix<double>& mat);
        template double FrobeniusNorm<double>(const CPURuntime& rt, const Matrix<double>& mat);
        template float FrobeniusNorm<float>(const CPURuntime& rt, const Matrix<float>& mat);

    } // namespace cpu
    
} // namespace tcgmtensor
