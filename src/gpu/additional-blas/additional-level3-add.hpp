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
        
        
        template <typename Allocator, typename GPUAllocator, typename Allocator2, typename GPUAllocator2>
        void MatrixMatrixProductBatchFP16(const CudaRuntime &cudart, const std::vector<Matrix<__half, Allocator, GPUAllocator>> &a_array, 
            const std::vector<Matrix<__half, Allocator, GPUAllocator>> &b_array, const std::vector<Matrix<float, Allocator2, GPUAllocator2>> &c_array,
            const float alpha, const float beta, const char *Ta, const char *Tb, bool fast = false)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            //assert(a_array.size() * b_array.size() == c_array.size());

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
            int index = 0;
            if (fast)
            {
            for (size_t i = 0; i < a_array.size(); ++i)
            {
                for (size_t j = 0; j < b_array.size(); ++j)
                {
                    if  (i+j > a_array.size() -1)
                    {
                        continue;
                    }
                    check_device_alloc(cudart, a_array[i]);
                    check_device_alloc(cudart, b_array[j]);
                    check_device_alloc(cudart, c_array[index]);
                    a_ptrs[index] = a_array[i].gpu_data();
                    b_ptrs[index] = b_array[j].gpu_data();
                    c_ptrs[index] = c_array[index].gpu_data();
                    index++;
                }
            }
            }
            else
            {
                for (size_t i = 0; i < a_array.size(); ++i)
                {
                    for (size_t j = 0; j < b_array.size(); ++j)
                    {
                        check_device_alloc(cudart, a_array[i]);
                        check_device_alloc(cudart, b_array[j]);
                        check_device_alloc(cudart, c_array[index]);
                        a_ptrs[index] = a_array[i].gpu_data();
                        b_ptrs[index] = b_array[j].gpu_data();
                        c_ptrs[index] = c_array[index].gpu_data();
                        index++;
                    }
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

        template <typename high, typename Allocator, typename GPUAllocator>
        void SymMatrixMatrixProduct(const CudaRuntime &cudart, const MPRuntime& mp_rt, const MixedPrecisionMatrix<high, Allocator, GPUAllocator> &A,
                                 const MixedPrecisionMatrix<high, Allocator, GPUAllocator> &B, Matrix<high, Allocator, GPUAllocator> &C,
                                 const high alpha = 1.0, const high beta = 0.0, const char *Ta = "N", const char *Tb = "N")
        {

            
            if (A.data() == B.data())
            {
               
                A.template split<__half>(cudart, mp_rt.template getMaxSplits<high>(), 
                    mp_rt.template getMPBuffer<high>(cudart, A.shape()));
                
                MatrixMatrixProductFP16(cudart, A.template getSplitMatrices<__half>()[1], A.template getSplitMatrices<__half>()[0],
                    mp_rt.template getMPBuffer<high>(cudart, A.shape()), scalbn(alpha, A.getSplitExponent(1) + A.getSplitExponent(0)) , 0.0, Ta, Tb);
                AddMatrices(cudart, mp_rt.template getMPBuffer<high>(cudart, A.shape()), 
                    mp_rt.template getMPBuffer<high>(cudart, A.shape()), C, 1.0f, 1.0f, "N", "T");
                MatrixMatrixProductFP16(cudart, A.template getSplitMatrices<__half>()[0], A.template getSplitMatrices<__half>()[0], C, 
                    scalbn(alpha, A.getSplitExponent(0) + A.getSplitExponent(0)) , 1.0, Ta, Tb);
                if (!mp_rt.fast_mode)
                {
                    MatrixMatrixProductFP16(cudart, A.template getSplitMatrices<__half>()[1], A.template getSplitMatrices<__half>()[1], C, 
                        scalbn(alpha, A.getSplitExponent(1) + A.getSplitExponent(1)) , 1.0, Ta, Tb);
                }   
            }
            else
            {
                MatrixMatrixProduct(cudart, mp_rt, A, B, C, alpha, beta, Ta, Tb);
            }
        };


        template <typename high, typename Allocator, typename GPUAllocator>
        void MatrixMatrixProduct(const CudaRuntime &cudart, const MPRuntime &mp_rt, const MixedPrecisionMatrix<high, Allocator, GPUAllocator> &A,
                                 const MixedPrecisionMatrix<high, Allocator, GPUAllocator> &B, MixedPrecisionMatrix<high, Allocator, GPUAllocator> &C,
                                 const high alpha = 1.0, const high beta = 0.0, const char *Ta = "N", const char *Tb = "N")
        {

            bool fast = mp_rt.fast_mode;
            bool batch = mp_rt.batch_mode;
            size_t maxsplit = mp_rt.template getMaxSplits<high>();
            if (A.data() == B.data())
            {
                
                if (beta == 0.0)
                {
                    A.template split<__half>(cudart, maxsplit, C);
                }
                else
                {
                    A.template split<__half>(cudart, maxsplit, mp_rt.template getMPBuffer<high>(cudart, A.shape()));
                }
                size_t asize = A.template splitSize<__half>(); 
                size_t index = 0;
                size_t combinations = 0;
                if (fast)
                {
                    combinations = (asize * (asize + 1)) / 2;
                    batch = true;
                }
                else
                {
                    combinations = asize * asize;
                }
                high* alpha_arr = new high[combinations];
                if (!batch)
                {
                    C.template createSplitMatrices<float>(cudart, combinations);
                }
                if (fast)
                {
                for (size_t i = 0; i < asize; ++i)
                {
                    for (size_t j = 0; j < asize; ++j)
                    {
                        
                        if (i+j > asize-1)
                        {
                            continue;
                        }
                        else
                        {
                            alpha_arr[index] = scalbn(alpha, A.getSplitExponent(i) + A.getSplitExponent(j));
                            if (!batch)
                            {
                                MatrixMatrixProductFP16(cudart, A.template getSplitMatrices<__half>()[i], A.template getSplitMatrices<__half>()[j], 
                                C.template getSplitMatrices<float>()[index], 1.0, 0.0, Ta, Tb);
                            }
                        }
                        index++;
                        }
                    }
                }
                else
                {
                    for (size_t i = 0; i < asize; ++i)
                    {
                        for (size_t j = 0; j < asize; ++j)
                        {
                            alpha_arr[index] = scalbn(alpha, A.getSplitExponent(i) + B.getSplitExponent(j));
                            if (!batch)
                            {
                                MatrixMatrixProductFP16(cudart, A.template getSplitMatrices<__half>()[i], A.template getSplitMatrices<__half>()[j], 
                                C.template getSplitMatrices<float>()[index], alpha_arr[index], 0.0, Ta, Tb);
                            }
                            index++;
                        }
                    }
                }
                
                if (batch)
                {
                    C.template createSplitMatrices<float>(cudart, combinations);
                    MatrixMatrixProductBatchFP16(cudart, A.template getSplitMatrices<__half>(), A.template getSplitMatrices<__half>(), 
                    C.template getSplitMatrices<float>(), 1.0, 0.0, Ta, Tb, fast);
                    C.merge(cudart, alpha_arr, beta); 
                }
                else
                {
                    C.merge(cudart);
                }
                
                delete[] alpha_arr;
            }
            else
            {
                if (beta == 0.0)
                {
                    A.template split<__half>(cudart, maxsplit, C);
                    B.template split<__half>(cudart, maxsplit, C);
                }
                else
                {
                    A.template split<__half>(cudart, maxsplit, mp_rt.template getMPBuffer<high>(cudart, A.shape()));
                    B.template split<__half>(cudart, maxsplit, mp_rt.template getMPBuffer<high>(cudart, B.shape()));
                }
                size_t asize = A.template splitSize<__half>();
                size_t bsize = B.template splitSize<__half>();
                size_t combinations = 0;
                if (fast)
                {
                    combinations = asize*(bsize + 1) / 2;
                    batch = true;
                }
                else
                {
                    combinations = asize * bsize;
                }
                high* alpha_arr = new high[combinations];
                if (!batch)
                {
                    C.template createSplitMatrices<float>(cudart, combinations);
                }
                int index = 0;
                if (fast)
                {
                for (size_t i = 0; i < asize; ++i)
                {
                    for (size_t j = 0; j < bsize; ++j)
                    {
                        
                        if (i+j > asize-1)
                        {
                            continue;
                        }
                        else
                        {
                            
                            alpha_arr[index] = scalbn(alpha, A.getSplitExponent(i) + B.getSplitExponent(j));
                        
                            if (!batch)
                            {
                                MatrixMatrixProductFP16(cudart, A.template getSplitMatrices<__half>()[i], B.template getSplitMatrices<__half>()[j], 
                                C.template getSplitMatrices<float>()[index], 1.0, 0.0, Ta, Tb);
                            }
                            }
                            index++;
                        }
                    }
                }
                else
                {
                    for (size_t i = 0; i < asize; ++i)
                    {
                        for (size_t j = 0; j < bsize; ++j)
                        {
                            alpha_arr[index] = scalbn(alpha, A.getSplitExponent(i) + B.getSplitExponent(j));
                            
                            if (!batch)
                            {
                                MatrixMatrixProductFP16(cudart, A.template getSplitMatrices<__half>()[i], B.template getSplitMatrices<__half>()[j], 
                                C.template getSplitMatrices<float>()[index], alpha_arr[index], 0.0, Ta, Tb);
                            }
                            index++;
                        }
                    }
                }
                if (!batch)                    
                {   
                    C.merge(cudart);
                }
                if (batch)
                {
                    C.template createSplitMatrices<float>(cudart, combinations);
                    MatrixMatrixProductBatchFP16(cudart, A.template getSplitMatrices<__half>(), B.template getSplitMatrices<__half>(), 
                    C.template getSplitMatrices<float>(), 1.0, 0.0, Ta, Tb, fast);
                    C.merge(cudart, alpha_arr, beta);
                    
                }
                
                delete[] alpha_arr;
            }
            
        };

    } // namespace gpu
}