#pragma once
#include "impl/blas/gpu/additional-level1.hpp"
#include "impl/blas/gpu/level3.hpp"
#include "impl/blas/gpu/additional-level3.hpp"
#include "custom-kernel/mixed-precision.h"
#include "../gpu-utils/utils.hpp"
#include "timer.hpp"
#include "impl/blas/gpu/additional-level1.hpp"
#include "impl/blas/gpu/level1.hpp"

namespace lahva
{
    namespace gpu
    {
        template <typename T1, typename T2>
        void launch_wmma_mm(T1 const *A, T1 const *B, T2 *C, uint32_t m, uint32_t n,
                            uint32_t k, bool is_A_transpose, bool is_B_transpose,
                            cudaStream_t stream, float alpha = 1.0, float beta = 0.0);
        
        template <typename Allocator, typename GPUAllocator, typename Allocator2, typename GPUAllocator2>
        void MatrixMatrixProductBatchFP16(const CudaRuntime &cudart, const std::vector<Matrix<__half, Allocator, GPUAllocator>> &a_array, 
            const std::vector<Matrix<__half, Allocator, GPUAllocator>> &b_array, const std::vector<Matrix<float, Allocator2, GPUAllocator2>> &c_array,
            const float alpha, const float beta, const char *Ta, const char *Tb)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            assert(a_array.size() * b_array.size() == c_array.size());

            auto [m, n, k] = check_size_mm(a_array[0], b_array[0], c_array[0], transa, transb);

            cudaDataType_t sp_type = CUDA_R_32F;
            cudaDataType_t half_type = CUDA_R_16F;
            cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;

            size_t lda = get_leading(m, k);
            size_t ldb = get_leading(k, n);
            size_t ldc = get_leading(m, n);
            
            __half** a_ptrs = new  __half*[c_array.size()];
            __half** b_ptrs = new  __half*[c_array.size()];
            float** c_ptrs = new float*[c_array.size()];
            for (size_t i = 0; i < a_array.size(); ++i)
            {
                for (size_t j = 0; j < b_array.size(); ++j)
                {
                    a_ptrs[i * b_array.size() + j] = a_array[i].gpu_data();
                    b_ptrs[i * b_array.size() + j] = b_array[j].gpu_data();
                    c_ptrs[i * b_array.size() + j] = c_array[i * b_array.size() + j].gpu_data();
                }
            }

            __half** da_ptrs;
            __half** db_ptrs;
            float** dc_ptrs;

            get_cuda_error(cudaMallocAsync((void**)&da_ptrs, sizeof(__half*) * c_array.size(), cudart.getStream()));
            get_cuda_error(cudaMallocAsync((void**)&db_ptrs, sizeof(__half*) * c_array.size(), cudart.getStream()));
            get_cuda_error(cudaMallocAsync((void**)&dc_ptrs, sizeof(float*) * c_array.size(), cudart.getStream()));

            get_cuda_error(cudaMemcpyAsync(da_ptrs, a_ptrs, sizeof(__half*) * c_array.size(), cudaMemcpyHostToDevice, cudart.getStream()));
            get_cuda_error(cudaMemcpyAsync(db_ptrs, b_ptrs, sizeof(__half*) * c_array.size(), cudaMemcpyHostToDevice, cudart.getStream()));
            get_cuda_error(cudaMemcpyAsync(dc_ptrs, c_ptrs, sizeof(float*) * c_array.size(), cudaMemcpyHostToDevice, cudart.getStream()));

            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasGemmBatchedEx(cudart.handle, transa, transb, m, n, k, &alpha, (void**)da_ptrs, half_type, lda, 
                    (const void**)db_ptrs, half_type, ldb, &beta, (void**)dc_ptrs, sp_type, ldc, (size_t)c_array.size(), computeType, CUBLAS_GEMM_DEFAULT);
            get_cublas_error(istat);

            get_cuda_error(cudaFreeAsync(da_ptrs, cudart.getStream()));
            get_cuda_error(cudaFreeAsync(db_ptrs, cudart.getStream()));
            get_cuda_error(cudaFreeAsync(dc_ptrs, cudart.getStream()));

            delete[] a_ptrs;
            delete[] b_ptrs;
            delete[] c_ptrs;
        };

        template <typename Allocator, typename GPUAllocator>
        void MatrixMatrixProductMP(const CudaRuntime &cudart, const Matrix<float, Allocator, GPUAllocator> &A,
                                   const Matrix<float, Allocator, GPUAllocator> &B, Matrix<float, Allocator, GPUAllocator> &C,
                                   const float alpha = 1.0, const float beta = 0.0, const char *Ta = "N", const char *Tb = "N")
        {

            // fiuture improvement k must be divisible by 8
            CPUTimer timer;
            timer.push("Allocation");
            using MyMatrix = Matrix<__half>;

            Vector<int> va(2, 0);
            va.copy2device(cudart);
            MyMatrix A1(A.shape(), cudart, A.get_gpuallocator());
            A1.allocateGPU(cudart);
            MyMatrix A2(A.shape(), cudart, A.get_gpuallocator());
            A2.allocateGPU(cudart);

            if (A.data() == B.data())
            {
                timer.pop();
                timer.push("Decompose");
                DecomposeVector2MP(cudart, A, A1, A2, va);
                timer.pop();
                timer.push("Product");
                // largest terms
                MatrixMatrixProductFP16(cudart, A1, A1, C, float(alpha * scalbn(1.0, va[0] + va[0])), beta, Ta, Tb);
                // cross terms
                MatrixMatrixProductFP16(cudart, A1, A2, C, float(alpha * scalbn(1.0, va[0] + va[1])), 1.0, Ta, Tb);
                MatrixMatrixProductFP16(cudart, A2, A1, C, float(alpha * scalbn(1.0, va[1] + va[0])), 1.0, Ta, Tb);
                // smallest terms
                MatrixMatrixProductFP16(cudart, A2, A2, C, float(alpha * scalbn(1.0, va[1] + va[1])), 1.0, Ta, Tb);
                timer.pop();
            }
            else
            {
                Vector<int> vb(2, 0);
                vb.copy2device(cudart);
                MyMatrix B1(B.shape(), cudart);
                B1.allocateGPU(cudart);
                MyMatrix B2(B.shape(), cudart);
                B2.allocateGPU(cudart);
                // Matrix<float> C1(C.shape(), cudart, C.get_gpuallocator());
                // C1.allocateGPU(cudart);
                timer.pop();
                timer.push("Decompose");
                DecomposeVector2MP(cudart, A, A1, A2, va);
                DecomposeVector2MP(cudart, B, B1, B2, vb);
                std::cout << "va: " << va[0] << " " << va[1] << std::endl;
                std::cout << "vb: " << vb[0] << " " << vb[1] << std::endl;
                timer.pop();
                timer.push("Product");

                MatrixMatrixProductFP16(cudart, A1, B1, C, float(alpha * scalbn(1.0, va[0] + vb[0])), beta, Ta, Tb);
                MatrixMatrixProductFP16(cudart, A2, A2, C, float(alpha * scalbn(1.0, va[1] + vb[1])), 1.0, Ta, Tb);
                MatrixMatrixProductFP16(cudart, A1, B2, C, float(alpha * scalbn(1.0, va[0] + vb[1])), 1.0, Ta, Tb);
                MatrixMatrixProductFP16(cudart, A2, B1, C, float(alpha * scalbn(1.0, va[1] + vb[0])), 1.0, Ta, Tb);
            }
            // timer.print_entries();
        };

        template <typename high, typename Allocator, typename GPUAllocator>
        void MatrixMatrixProduct(const CudaRuntime &cudart, const MixedPrecisionMatrix<high, Allocator, GPUAllocator> &A,
                                 const MixedPrecisionMatrix<high, Allocator, GPUAllocator> &B, MixedPrecisionMatrix<high, Allocator, GPUAllocator> &C,
                                 const high alpha = 1.0, const high beta = 0.0, const char *Ta = "N", const char *Tb = "N")
        {

            // fiuture improvement k must be divisible by 8
            CPUTimer timer;
            int maxsplit = 0;
            if constexpr (std::is_same<high, float>::value)
            {
                maxsplit = 2;
            }
            else if constexpr (std::is_same<high, double>::value)
            {
                maxsplit = 4;
            }
            
            bool batch = true;
            bool fast = false;

            if (A.data() == B.data())
            {
                timer.push("Decompose");
                A.template split<__half>(cudart, maxsplit);
                timer.pop();
                timer.push("Prepare");
                size_t asize = A.template splitSize<__half>();
                size_t combinations = asize * asize;
                high* alpha_arr = new high[combinations];

                for (size_t i = 0; i < asize; ++i)
                {
                    for (size_t j = 0; j < asize; ++j)
                    {
                        alpha_arr[i * asize + j] = scalbn(alpha, A.getSplitExponent(i) + A.getSplitExponent(j));
                        //if (!batch )
                        //{
                            //if (i == 0 and j == 0)
                            //{
                                //MatrixMatrixProductFP16(cudart, A.getSplitMatrices()[i], A.getSplitMatrices()[j], C, alpha_arr[i * A.splitSize() + j], beta, Ta, Tb);
                            //}
                            //else
                            //{
                                //MatrixMatrixProductFP16(cudart, A.getSplitMatrices()[i], A.getSplitMatrices()[j], C, alpha_arr[i * A.splitSize() + j], 1.0, Ta, Tb);
                            //}
                        //}
                    }
                }

                if (batch)
                {
                    timer.pop();
                    timer.push("Product");
                    C.template createSplitMatrices<float>(cudart, combinations);
                    MatrixMatrixProductBatchFP16(cudart, A.template getSplitMatrices<__half>(), A.template getSplitMatrices<__half>(), C.template getSplitMatrices<float>(), 1.0, 0.0, Ta, Tb);
                    timer.pop();
                    timer.push("Merge");
                    C.merge(cudart, alpha_arr, beta);
                }
               
                
                
                delete[] alpha_arr;
                timer.pop();
            }
            else
            {
                timer.push("Decompose");
                A.template split<__half>(cudart, maxsplit);
                B.template split<__half>(cudart, maxsplit);
                timer.pop();
                timer.push("Product");
                size_t asize = A.template splitSize<__half>();
                size_t bsize = B.template splitSize<__half>();
                size_t combinations = asize * bsize;
                high* alpha_arr = new high[combinations];

                for (size_t i = 0; i < asize; ++i)
                {
                    for (size_t j = 0; j < bsize; ++j)
                    {
                        alpha_arr[i * bsize + j] = scalbn(alpha, A.getSplitExponent(i) + B.getSplitExponent(j));
                        
                        if (!batch)
                        {
                            C.template createSplitMatrices<float>(cudart, 1);
                            //std::cout << "alpha[" << i << "," << j << "] = " << alpha_arr[i * A.splitSize() + j] << std::endl;
                            if (i == 0 and j == 0)
                            {
                                MatrixMatrixProductFP16(cudart, A.template getSplitMatrices<__half>()[i], B.template getSplitMatrices<__half>()[j], C.template getSplitMatrices<float>(), alpha_arr[i * A.splitSize() + j], beta, Ta, Tb);
                            }
                            else
                            {
                                MatrixMatrixProductFP16(cudart, A.template getSplitMatrices<__half>()[i], B.template getSplitMatrices<__half>()[j], C.template getSplitMatrices<float>(), alpha_arr[i * A.splitSize() + j], 1.0, Ta, Tb);
                            }
                        }
                    }
                }
                if (batch)
                {
                    C.template createSplitMatrices<float>(cudart, combinations);
                    MatrixMatrixProductBatchFP16(cudart, A.template getSplitMatrices<__half>(), B.template getSplitMatrices<__half>(), C.template getSplitMatrices<float>(), 1.0, 0.0, Ta, Tb);
                    timer.pop();
                    timer.push("Merge");
                    C.merge(cudart, alpha_arr, beta);
                    
                }
                delete[] alpha_arr;
                timer.pop();
                
            }
            
        };

    } // namespace gpu
}