#include "linalg.hpp"
#include "runtime.hpp"
#include "../../gpu-utils/utils.hpp"
#include "impl/blas/gpu/level1.hpp"
#include "impl/blas/gpu/additional-level1.hpp"
#include "impl/blas/gpu/additional-level2.hpp"
#include "common.h"
#include "timer.hpp"
#include "impl/tensor/allocators.hpp"
#include <vector>
namespace lahva
{
    namespace gpu
    {
        __global__ static void SymmetrizedDiagonalMatrixMatrixProductKernel_Mixed(unsigned long long ndim, const double scale, const double *diag,
                                                                                  const float *matrixIn, float *matrixOut)
        {
            // unsigned long long id = blockDim.x < ndim ? (blockIdx.x*blockDim.x)%ndim+threadIdx.x : threadIdx.x  ;
            // unsigned long long jd = blockDim.x < ndim ? (blockIdx.x*blockDim.x)/ndim : blockIdx.x ;
            // A.1 B.1 C.1
            // A.2 B.2 C.2
            // A.3 B.3 C.3
            unsigned long long id = blockIdx.y;
            unsigned long long jd = threadIdx.x + blockIdx.x * blockDim.x;
            // 0.5*(matrix * diag + matrix^T * diag)
            if (id < ndim && jd < ndim)
            {
                matrixOut[id * ndim + jd] += ((float)(0.5 * scale * (diag[id] + diag[jd])) * matrixIn[id * ndim + jd]);
            }
        }

        template <typename inprec, typename outprec>
        __global__ void CastMatrix(const unsigned long long ntot, const inprec *MatIn, outprec *MatOut)
        {
            unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < ntot)
                MatOut[id] = (outprec)(MatIn[id]); // fast_float2double(MatIn[id]);
            __syncthreads();
        }

        template <typename inprec, typename outprec>
        __global__ void IncrVectors(const unsigned long long ntot, const inprec *MatIn, outprec *MatOut)
        {
            unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < ntot)
                MatOut[id] += (outprec)(MatIn[id]); // fast_float2double(MatIn[id]);
            __syncthreads();
        }

        template <typename T>
        __device__ T truncateSignificantDigits(T value, int significantDigits)
        {

            // Handle the case where the value is zero
            if (value == 0.0)
            {
                return 0.0;
            }

            // Calculate the factor based on the number of significant digits
            T factor = pow(10.0, significantDigits);
            return trunc(value * factor) / factor;
        }

        template <>
        __global__ void IncrVectors(const unsigned long long ntot, const float *MatIn, double *MatOut)
        {
            unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < ntot)
                MatOut[id] += truncateSignificantDigits(MatIn[id], 8); // fast_float2double(MatIn[id]);
            __syncthreads();
        }

        template <typename inprec, typename outprec, size_t blockSize, size_t chunkSize = 1>
        __global__ void DecomposeMatrixKernel(const unsigned long long ntot, inprec *MatIn, outprec *SplitOut, int tau, inprec sigma)
        {
            unsigned long long idx = blockIdx.x * blockSize + threadIdx.x;

            size_t tid = threadIdx.x * chunkSize;
            __shared__ inprec split[blockSize*chunkSize];
            __shared__ inprec input[blockSize*chunkSize];
            
            #pragma unroll chunkSize
            for (int i = 0; i < chunkSize; i++)
            {
                const size_t iidx = idx + i * blockSize * gridDim.x;
                
                if (iidx < ntot)
                {
                    input[tid + i] = MatIn[iidx];
                    split[tid + i] = (input[tid+i] + sigma) - sigma;
                }
            }
                
            
            
            //__syncthreads();
            
            #pragma unroll chunkSize
            for (int i = 0; i < chunkSize; i++)
            {
                const size_t iidx = idx + i * blockSize * gridDim.x;
                if (iidx < ntot)
                {
                    SplitOut[iidx] = (scalbnf(split[tid + i], -tau));
                    MatIn[iidx] = input[tid+i] - split[tid + i];
                }; 
            }
            //__syncthreads();
        }

        void SymmetrizedON2ScalingProductGPU(const CudaRuntime &cudart, const Vector_<double> &diag1, const Matrix_<float> &matrix1,
                                             const Vector_<double> &diag2, const Matrix_<float> &matrix2, Matrix_<float> &matrix_out)

        {
            int n = diag1.size();

            check_device_alloc(cudart, diag1);
            check_device_alloc(cudart, diag2);
            check_device_alloc(cudart, matrix1);
            check_device_alloc(cudart, matrix2);
            check_device_alloc(cudart, matrix_out);

            dim3 nThreadsPerBlock(cudart.blockSize(), 1);
            dim3 nBlocksPerGrid(ceil((double)(n) / double(cudart.blockSize())), n);
            // diagonal * offdiagonal
            if (diag1.gpu_data() == diag2.gpu_data() && matrix1.gpu_data() == matrix2.gpu_data())
            {
                double two = 2.0;
                SymmetrizedDiagonalMatrixMatrixProductKernel_Mixed<<<nBlocksPerGrid, nThreadsPerBlock, 0, cudart.getStream()>>>(n, two, diag1.gpu_data(), matrix1.gpu_data(), matrix_out.gpu_data());
            }
            else
            {
                double one = 1.0;
                SymmetrizedDiagonalMatrixMatrixProductKernel_Mixed<<<nBlocksPerGrid, nThreadsPerBlock, 0, cudart.getStream()>>>(n, one, diag1.gpu_data(), matrix2.gpu_data(), matrix_out.gpu_data());
                SymmetrizedDiagonalMatrixMatrixProductKernel_Mixed<<<nBlocksPerGrid, nThreadsPerBlock, 0, cudart.getStream()>>>(n, one, diag2.gpu_data(), matrix1.gpu_data(), matrix_out.gpu_data());
            }
        }

        template <typename in, typename out>
        void AddVectors(const CudaRuntime &cudart, const GPUTensor_<in> &X, GPUTensor_<out> &Y)
        {
            assert(X.size() == Y.size());
            check_device_alloc(cudart, Y);
            check_device_alloc(cudart, X);
            unsigned long long n = X.size();
            IncrVectors<in, out><<<cudart.gridSize(n, 1), cudart.blockSize(), 0, cudart.getStream()>>>(n, X.gpu_data(), Y.gpu_data());
        }

        void CopyVectors(const CudaRuntime &cudart, const GPUTensor_<double> &X, GPUTensor_<float> &Y)
        {
            assert(X.size() == Y.size());
            check_device_alloc(cudart, Y);
            check_device_alloc(cudart, X);

            unsigned long long n = X.size();
            CastMatrix<double, float><<<cudart.gridSize(n, 1), cudart.blockSize(), 0, cudart.getStream()>>>(n, X.gpu_data(), Y.gpu_data());
        }

        void CopyVectors(const CudaRuntime &cudart, const GPUTensor_<float> &X, GPUTensor_<double> &Y)
        {
            assert(X.size() == Y.size());
            unsigned long long n = X.size();
            check_device_alloc(cudart, X);
            check_device_alloc(cudart, Y);

            CastMatrix<float, double><<<cudart.gridSize(n, 1), cudart.blockSize(), 0, cudart.getStream()>>>(n, X.gpu_data(), Y.gpu_data());
        }

        template <typename inprec, typename outprec>
        void DecomposeVector2MP(const CudaRuntime &cudart, const GPUTensor_<inprec> &in, GPUTensor_<outprec> &out1, GPUTensor_<outprec> &out2, GPUTensor_<int> &coeff)
        {
            assert(in.size() == out1.size());
            assert(in.size() == out2.size());
            unsigned long long n = in.size();
            check_device_alloc(cudart, in);
            check_device_alloc(cudart, out1);
            check_device_alloc(cudart, out2);
            Vector<inprec> CopyIn(in.size());
            CopyVectors(cudart, in, CopyIn);
            check_device_alloc(cudart, coeff);
            check_device_alloc(cudart, CopyIn);

            int rho = ceil(getEpse<inprec>() - (getEpse<outprec>() - log2(2)) / 2);
            // std::cout << "rho: " << rho << std::endl;
            int maxsplit = 2;
            for (int i = 0; i < maxsplit; i++)
            {

                int idmax;
                get_cublas_error(cublasIsamax(cudart.handle, n, CopyIn.gpu_data(), 1, &idmax));
                CopyIn.copy2host(cudart);
                cudart.synchronize();
                inprec mu = abs(CopyIn[idmax - 1]);
                CopyIn.copy2device(cudart);
                int tau = ceil(log2(mu));
                inprec sigma = scalbnf(1.0, rho + tau);
                coeff[i] = tau;
                if (i == 0)
                {
                    DecomposeMatrixKernel<inprec, outprec, 512><<<cudart.gridSize(n, 1), cudart.blockSize(), 0, cudart.getStream()>>>
                (n, CopyIn.gpu_data(), out1.gpu_data(), tau, sigma);
                }
                else
                {
                    DecomposeMatrixKernel<inprec, outprec, 512><<<cudart.gridSize(n, 1), cudart.blockSize(), 0, cudart.getStream()>>>
                (n, CopyIn.gpu_data(), out2.gpu_data(), tau, sigma);
                }
            }
        }

        template <typename inprec, typename outprec, typename Allocator, typename GPUAllocator>
        void SplitMatrix(const CudaRuntime &cudart, Matrix_<inprec> &in, std::vector<Matrix<outprec, Allocator, GPUAllocator>>& out, GPUTensor_<int> &coeff, int maxsplit)
        {
            CPUTimer timer;

            unsigned long long n = in.size();
            
            check_device_alloc(cudart, in);

            int rho = ceil(getEpse<inprec>() - (getEpse<outprec>() - log2(2)) / 2);
            cudart.setblockSize(256);
           
            for (int i = 0; i < maxsplit; i++)
            {
                check_device_alloc(cudart, out[i]);
                assert(out[i].size() == in.size());
                timer.push("Amax");
                int idmax;
                if constexpr (std::is_same<inprec, float>::value)
                {
                    // For mixed precision, we use the original in data
                    cudart.cublasSetStream_();
                    get_cublas_error(cublasIsamax(cudart.handle, n, in.gpu_data(), 1, &idmax));
                }
                else
                {
                    cudart.cublasSetStream_();
                    get_cublas_error(cublasIdamax(cudart.handle, n, in.gpu_data(), 1, &idmax));
                }
                
                timer.pop();
                timer.push("Copy");
                inprec max;
                get_cuda_error(cudaMemcpyAsync(&max, &in.gpu_data()[idmax-1],sizeof(inprec), cudaMemcpyDeviceToHost, cudart.getStream()));    
                //cudart.synchronize();
                inprec mu = abs(max);
                timer.pop();
                int tau = ceil(log2(mu));
                inprec sigma = scalbnf(1.0, rho + tau);
                coeff[i] = tau;
                timer.push("Decompose");
                int computeperThread = 2;
                DecomposeMatrixKernel<inprec, outprec, 256, 2><<<cudart.gridSize(n/computeperThread, 1), cudart.blockSize(), 0, cudart.getStream()>>>
                (n, in.gpu_data(), out[i].gpu_data(), tau, sigma);
                timer.pop();
            }
            cudart.setblockSize(512);
        }

        template void AddVectors<float, double>(const CudaRuntime &cudart, const GPUTensor_<float> &X, GPUTensor_<double> &Y);
        template void AddVectors<double, float>(const CudaRuntime &cudart, const GPUTensor_<double> &X, GPUTensor_<float> &Y);
        // template void DecomposeVector2MP<double, float>(const CudaRuntime &, const GPUTensor_<double> &in, GPUTensor_<float> &out1, GPUTensor_<float> &out2, GPUTensor_<double>& coeff);
        template void DecomposeVector2MP<float, __half>(const CudaRuntime &cudart, const GPUTensor_<float> &in, GPUTensor_<__half> &out1, GPUTensor_<__half> &out2, GPUTensor_<int> &coeff);
        //template void SplitMatrix<float, __half>(const CudaRuntime &cudart, Matrix_<float> &in, std::vector<Matrix<__half, CudaHostAllocator<__half>, CudaDeviceAllocator<__half>>& out, GPUTensor_<int> &coeff, int maxsplit);
        template void SplitMatrix<float, __half, CudaHostAllocator<__half>, CudaDeviceAllocator<__half>>(const CudaRuntime &cudart, Matrix_<float> &in, std::vector<Matrix<__half, CudaHostAllocator<__half>, CudaDeviceAllocator<__half>>> &out, GPUTensor_<int> &coeff, int maxsplit);
        template void SplitMatrix<float, __half, CudaHostAllocator<__half>, CudaDeviceAsyncAllocator<__half>>(const CudaRuntime &cudart, Matrix_<float> &in, std::vector<Matrix<__half, CudaHostAllocator<__half>, CudaDeviceAsyncAllocator<__half>>> &out, GPUTensor_<int> &coeff, int maxsplit);
        //template void SplitMatrix<double, __half>(const CudaRuntime &cudart, Matrix_<double> &in, Matrix_<__half>& out1, Matrix_<__half>& out2, GPUTensor_<int> &coeff, int maxsplit);
        //template void SplitMatrix<double, float, CudaHostAllocator<float>, CudaDeviceAllocator<float>>(const CudaRuntime &cudart, Matrix_<double> &in, std::vector<Matrix<float, CudaHostAllocator<float>, CudaDeviceAllocator<float>>> &out, GPUTensor_<int> &coeff, int maxsplit);
    } // namespace gpu

} // namespace lahva
