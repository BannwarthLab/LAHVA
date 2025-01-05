#include "../../../gpu-utils/utils.hpp"
#include "impl/blas/gpu/level1.hpp"
#include "impl/blas/gpu/additional-level1.hpp"
#include "impl/blas/gpu/additional-level2.hpp"
//#include "mixed-precision.h"
#include "impl/blas/gpu/level1.hpp"
#include "common.h"
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
            T factor = pow(10.0, significantDigits - ceil(log10(fabs(value))));
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


        template <typename inprec, typename outprec>
        __global__ void DecomposeMatrixKernel(const unsigned long long ntot, const inprec *MatIn, outprec *MatOut1, outprec *MatOut2)
        {
            unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < ntot)
            {
                inprec trunc_ = truncateSignificantDigits(MatIn[id], getDig<outprec>());
                MatOut1[id] = trunc_;
                MatOut2[id] = getSub(MatIn[id], trunc_);
            }; // fast_float2double(MatIn[id]);
            //__syncthreads();
        }

        template <>
        __global__ void DecomposeMatrixKernel(const unsigned long long ntot, const float *MatIn, __half *MatOut1, __half *MatOut2)
        {
            unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < ntot)
            {
                MatOut1[id] = __float2half(MatIn[id]);
                MatOut2[id] = getSub<float>(MatIn[id], MatOut1[id]);
            }; // fast_float2double(MatIn[id]);
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
        void DecomposeVector2MP(const CudaRuntime &cudart, const GPUTensor_<inprec> &in, GPUTensor_<outprec> &out1, GPUTensor_<outprec> &out2)
        {

            assert(in.size() == out1.size());
            assert(in.size() == out2.size());
            unsigned long long n = in.size();
            check_device_alloc(cudart, in);
            check_device_alloc(cudart, out1);
            check_device_alloc(cudart, out2);

            DecomposeMatrixKernel<inprec, outprec><<<cudart.gridSize(n, 1), cudart.blockSize(), 0, cudart.getStream()>>>(n, in.gpu_data(), out1.gpu_data(), out2.gpu_data());
        }

        template void AddVectors<float, double>(const CudaRuntime &cudart, const GPUTensor_<float> &X, GPUTensor_<double> &Y);
        template void AddVectors<double, float>(const CudaRuntime &cudart, const GPUTensor_<double> &X, GPUTensor_<float> &Y);
        template void DecomposeVector2MP<double, float>(const CudaRuntime &, const GPUTensor_<double> &in, GPUTensor_<float> &out1, GPUTensor_<float> &out2);
        template void DecomposeVector2MP<float, __half>(const CudaRuntime &, const GPUTensor_<float> &in, GPUTensor_<__half> &out1, GPUTensor_<__half> &out2);
    } // namespace gpu

} // namespace lahva
