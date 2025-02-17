#pragma once
#include "impl/blas/gpu/additional-level1.hpp"
#include "impl/blas/gpu/level3.hpp"
#include "impl/blas/gpu/additional-level3.hpp"
#include "custom-kernel/mixed-precision.h"
#include "../../gpu-utils/utils.hpp"
#include "timer.hpp"
#include "impl/blas/gpu/additional-level1.hpp"
#include "impl/blas/gpu/level3.hpp"

namespace lahva
{
    namespace gpu
    {
        template <typename T1, typename T2>
        void launch_wmma_mm(T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n,
                    uint32_t k, bool is_A_transpose, bool is_B_transpose,
                    cudaStream_t stream, float alpha = 1.0, float beta = 0.0);

        template <typename Allocator, typename GPUAllocator>
        void MatrixMatrixProductMP(const CudaRuntime &cudart, const Matrix<float, Allocator, GPUAllocator> &A,
                                   const Matrix<float, Allocator, GPUAllocator> &B, Matrix<float, Allocator, GPUAllocator> &C,
                                   const float alpha = 1.0, const float beta = 0.0, const char *Ta = "N", const char *Tb = "N")
        {

            // fiuture improvement k must be divisible by 8
            CPUTimer timer;
            timer.push("Allocation");
            using MyMatrix = Matrix<__half>;

            MyMatrix A1(A.shape(), cudart, A.get_gpuallocator());
            A1.allocateGPU(cudart);
            MyMatrix A2(A.shape(), cudart);
            A2.allocateGPU(cudart);
            MyMatrix B1(B.shape(), cudart);
            B1.allocateGPU(cudart);
            MyMatrix B2(B.shape(), cudart);
            B2.allocateGPU(cudart);
            //Matrix<float> C1(C.shape(), cudart, C.get_gpuallocator());
            //C1.allocateGPU(cudart);
            timer.pop();
            timer.push("Decompose");
            DecomposeVector2MP(cudart, A, A1, A2);
            DecomposeVector2MP(cudart, B, B1, B2);
            timer.pop();
            timer.push("Product");

            launch_wmma_mm(A1.gpu_data(), B1.gpu_data(), C.gpu_data(), C.shape().first, C.shape().second, A.shape().second, false, false, cudart.getStream(), alpha, beta);
            launch_wmma_mm(A2.gpu_data(), B1.gpu_data(), C.gpu_data(), C.shape().first, C.shape().second, A.shape().second, false, false, cudart.getStream(), alpha, 1.0);
            launch_wmma_mm(A1.gpu_data(), B2.gpu_data(), C.gpu_data(), C.shape().first, C.shape().second, A.shape().second, false, false, cudart.getStream(), alpha, 1.0);
            //MatrixMatrixProductFP16(cudart, A1, B1, C, float(alpha), beta, Ta, Tb);
            //MatrixMatrixProductFP16(cudart, A2, B2, C1, float(alpha), 1.0, Ta, Tb);
            //MatrixMatrixProductFP16(cudart, A2, B1, C, float(alpha), 1.0, Ta, Tb);
            //MatrixMatrixProductFP16(cudart, A1, B2, C, float(alpha), 1.0, Ta, Tb);
          
            timer.print_entries();
        };

    } // namespace gpu
}