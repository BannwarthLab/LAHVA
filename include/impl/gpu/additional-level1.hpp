#ifndef TCGMTENSOR_ADD_LEVEL1_HPP
#define TCGMTENSOR_ADD_LEVEL1_HPP
#include "linalg.hpp"
#include "runtime.hpp"

namespace tcgmtensor{
    namespace gpu
    {
        //Currently only floar 
        template<typename T>
        void Hadamard(const CudaRuntime& cudart, const GPUTensor<T>& vecin, GPUTensor<T>& vecinout);
        template<typename T>
        void Hadamard(const CudaRuntime& cudart, const GPUTensor<T>& vecin, const GPUTensor<T>& vecin2, GPUTensor<T>& vecout);
        template<typename T>
        void Hadamard(const CudaRuntime& cudart, const Matrix<T>& vecin, const Vector<T>& vecin2, Matrix<T>& vecout);
        template<typename T>
        double ComputeTrace(const CudaRuntime& cudart, const Matrix<T>&, bool use_diag = false);
        double ComputeTrace(const CudaRuntime& cudart, const Vector<double>& diag);
        template<typename T>
        void SymmetrizeMatrix(const CudaRuntime& cudart, Matrix<T>&);

    } // namespace gpu
    
}   
#endif