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
            const float alpha, const float beta, const char *Ta, const char *Tb)
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
        void SymMatrixMatrixProduct(const CudaRuntime &cudart, const MixedPrecisionMatrix<high, Allocator, GPUAllocator> &A,
                                 const MixedPrecisionMatrix<high, Allocator, GPUAllocator> &B, Matrix<high, Allocator, GPUAllocator> &C,
                                 Matrix<high, Allocator, GPUAllocator> &buffer,
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
            //Matrix<high, Allocator, GPUAllocator> buffer(A.shape(), cudart, A.get_gpuallocator());
            bool batch = true;
            bool fast = false;
            
            if (A.data() == B.data())
            {
                timer.push("Decompose");
                A.template split<__half>(cudart, maxsplit, buffer);
                timer.pop();
                timer.push("Multiplication");
                size_t asize = A.template splitSize<__half>();
                size_t combinations = asize * asize;
                
                MatrixMatrixProductFP16(cudart, A.template getSplitMatrices<__half>()[1], A.template getSplitMatrices<__half>()[0], buffer, scalbn(alpha, A.getSplitExponent(1) + A.getSplitExponent(0)) , 0.0, Ta, Tb);
                AddMatrices(cudart, buffer, buffer, C, 1.0f, 1.0f, "N", "T");
                MatrixMatrixProductFP16(cudart, A.template getSplitMatrices<__half>()[0], A.template getSplitMatrices<__half>()[0], C, scalbn(alpha, A.getSplitExponent(0) + A.getSplitExponent(0)) , 1.0, Ta, Tb);
                if (!cudart.fast_mp)
                {
                    MatrixMatrixProductFP16(cudart, A.template getSplitMatrices<__half>()[1], A.template getSplitMatrices<__half>()[1], C, scalbn(alpha, A.getSplitExponent(1) + A.getSplitExponent(1)) , 1.0, Ta, Tb);
                }
               
                timer.pop();
            }
            
            
        };

        //template <typename high, typename Allocator, typename GPUAllocator>
        //void MatrixMatrixProduct(const CudaRuntime &cudart, const MixedPrecisionMatrix<high, Allocator, GPUAllocator> &A,
                                 //const MixedPrecisionMatrix<high, Allocator, GPUAllocator> &B, MixedPrecisionMatrix<high, Allocator, GPUAllocator> &C,
                                 //const high alpha = 1.0, const high beta = 0.0, const char *Ta = "N", const char *Tb = "N")
        //{

            //// fiuture improvement k must be divisible by 8
            //CPUTimer timer;
            //int maxsplit = 0;
            //Matrix<high, Allocator, GPUAllocator> buffer(A.shape(), cudart, A.get_gpuallocator());
            //if constexpr (std::is_same<high, float>::value)
            //{
                //maxsplit = 2;
            //}
            //else if constexpr (std::is_same<high, double>::value)
            //{
                //maxsplit = 4;
            //}
            
            //bool batch = true;
            //bool fast = false;

            //if (A.data() == B.data())
            //{
                //timer.push("Decompose");
                //A.template split<__half>(cudart, maxsplit, buffer);
                //timer.pop();
                //timer.push("Prepare");
                //size_t asize = A.template splitSize<__half>();
                //size_t combinations = asize * asize;
                //high* alpha_arr = new high[combinations];

                //for (size_t i = 0; i < asize; ++i)
                //{
                    //for (size_t j = 0; j < asize; ++j)
                    //{
                        //alpha_arr[i * asize + j] = scalbn(alpha, A.getSplitExponent(i) + A.getSplitExponent(j));
                        ////if (!batch )
                        ////{
                            ////if (i == 0 and j == 0)
                            ////{
                                ////MatrixMatrixProductFP16(cudart, A.getSplitMatrices()[i], A.getSplitMatrices()[j], C, alpha_arr[i * A.splitSize() + j], beta, Ta, Tb);
                            ////}
                            ////else
                            ////{
                                ////MatrixMatrixProductFP16(cudart, A.getSplitMatrices()[i], A.getSplitMatrices()[j], C, alpha_arr[i * A.splitSize() + j], 1.0, Ta, Tb);
                            ////}
                        ////}
                    //}
                //}

                //if (batch)
                //{
                    //timer.pop();
                    //timer.push("Product");
                    //C.template createSplitMatrices<float>(cudart, combinations);
                    //MatrixMatrixProductBatchFP16(cudart, A.template getSplitMatrices<__half>(), A.template getSplitMatrices<__half>(), C.template getSplitMatrices<float>(), 1.0, 0.0, Ta, Tb);
                    //timer.pop();
                    //timer.push("Merge");
                    //C.merge(cudart, alpha_arr, beta);
                //}
               
                
                
                //delete[] alpha_arr;
                //timer.pop();
            //}
            //else
            //{
                //timer.push("Decompose");
                //A.template split<__half>(cudart, maxsplit, buffer);
                //B.template split<__half>(cudart, maxsplit, buffer);
                //timer.pop();
                //timer.push("Product");
                //size_t asize = A.template splitSize<__half>();
                //size_t bsize = B.template splitSize<__half>();
                //size_t combinations = asize * bsize;
                //high* alpha_arr = new high[combinations];

                //for (size_t i = 0; i < asize; ++i)
                //{
                    //for (size_t j = 0; j < bsize; ++j)
                    //{
                        //alpha_arr[i * bsize + j] = scalbn(alpha, A.getSplitExponent(i) + B.getSplitExponent(j));
                        
                        //if (!batch)
                        //{
                            //C.template createSplitMatrices<float>(cudart, 1);
                            ////std::cout << "alpha[" << i << "," << j << "] = " << alpha_arr[i * A.splitSize() + j] << std::endl;
                            //if (i == 0 and j == 0)
                            //{
                                //MatrixMatrixProductFP16(cudart, A.template getSplitMatrices<__half>()[i], B.template getSplitMatrices<__half>()[j], C.template getSplitMatrices<float>()[0], alpha_arr[i * A.template splitSize<__half>() + j], beta, Ta, Tb);
                            //}
                            //else
                            //{
                                //MatrixMatrixProductFP16(cudart, A.template getSplitMatrices<__half>()[i], B.template getSplitMatrices<__half>()[j], C.template getSplitMatrices<float>()[0], alpha_arr[i * A.template splitSize<__half>() + j], 1.0, Ta, Tb);
                            //}
                        //}
                    //}
                //}
                //if (batch)
                //{
                    //C.template createSplitMatrices<float>(cudart, combinations);
                    //MatrixMatrixProductBatchFP16(cudart, A.template getSplitMatrices<__half>(), B.template getSplitMatrices<__half>(), C.template getSplitMatrices<float>(), 1.0, 0.0, Ta, Tb);
                    //timer.pop();
                    //timer.push("Merge");
                    //C.merge(cudart, alpha_arr, beta);
                    
                //}
                //delete[] alpha_arr;
                //timer.pop();
                //std::cout << timer.print_entries() << std::endl;
            //}
            
        //};

        template <typename high, typename Allocator, typename GPUAllocator>
        void MatrixMatrixProduct(const CudaRuntime &cudart, const MixedPrecisionMatrix<high, Allocator, GPUAllocator> &A,
                                 const MixedPrecisionMatrix<high, Allocator, GPUAllocator> &B, MixedPrecisionMatrix<high, Allocator, GPUAllocator> &C,
                                 Matrix<high, Allocator, GPUAllocator> &buffer,
                                 const high alpha = 1.0, const high beta = 0.0, const char *Ta = "N", const char *Tb = "N")
        {

            // fiuture improvement k must be divisible by 8
            CPUTimer timer;
            int maxsplit = 0;
            if (buffer.size() != A.size())
            {
                buffer = Matrix<high, Allocator, GPUAllocator>(A.shape(), cudart, A.get_gpuallocator());
            }
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
                A.template split<__half>(cudart, maxsplit, buffer);
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
                        {
                            //if (i == 0 and j == 0)
                            //{
                                //MatrixMatrixProductFP16(cudart, A.getSplitMatrices()[i], A.getSplitMatrices()[j], C, alpha_arr[i * A.splitSize() + j], beta, Ta, Tb);
                            //}
                            //else
                            //{
                                //MatrixMatrixProductFP16(cudart, A.getSplitMatrices()[i], A.getSplitMatrices()[j], C, alpha_arr[i * A.splitSize() + j], 1.0, Ta, Tb);
                            //}
                        }
                    }
                }
 
                
                delete[] alpha_arr;
                timer.pop();
            }
            else
            {
                timer.push("Decompose");
                A.template split<__half>(cudart, maxsplit, buffer);
                B.template split<__half>(cudart, maxsplit, buffer);
                timer.pop();
                timer.push("Product");
                size_t asize = A.template splitSize<__half>();
                size_t bsize = B.template splitSize<__half>();
                size_t combinations = 10;
                high* alpha_arr = new high[combinations];
                if (!batch)
                {
                    C.template createSplitMatrices<float>(cudart, combinations);
                }
                int index = 0;
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
                    
                
                if (!batch)
                    {   

                        timer.pop();
                        timer.push("Merge");
                        C.merge(cudart, alpha_arr, beta);
                
                    }
                if (batch)
                {
                    C.template createSplitMatrices<float>(cudart, combinations);
                    MatrixMatrixProductBatchFP16(cudart, A.template getSplitMatrices<__half>(), B.template getSplitMatrices<__half>(), 
                    C.template getSplitMatrices<float>(), 1.0, 0.0, Ta, Tb);
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