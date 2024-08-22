#include "../../gpu-utils/utils.hpp"
#include "impl/gpu/level1.hpp"
#include "mixed-precision.h"

namespace tcgmtensor
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

        __global__  static  void CastMatrixSPToDP(const unsigned long long ntot, const float *MatIn, double *MatOut)
        {
          unsigned long long id = blockIdx.x*blockDim.x+threadIdx.x;
          if(id < ntot) MatOut[id]=(double) MatIn[id]; //fast_float2double(MatIn[id]);
          __syncthreads();
        }

        __global__  static  void CastMatrixDPToSP(const unsigned long long ntot, const double *MatIn, float *MatOut)
        {
          unsigned long long id = blockIdx.x*blockDim.x+threadIdx.x;
          if(id < ntot) MatOut[id]=(double) MatIn[id]; //fast_float2double(MatIn[id]);
          __syncthreads();
        }

        void SymmetrizedON2ScalingProductGPU(const CudaRuntime& cudart, const Vector<double>& diag1, const Matrix<float>& matrix1,
                                             const Vector<double>& diag2, const Matrix<float>& matrix2, Matrix<float>& matrix_out)
        
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
                SymmetrizedDiagonalMatrixMatrixProductKernel_Mixed<<<nBlocksPerGrid, nThreadsPerBlock>>>(n, two, diag1.gpu_data(), matrix1.gpu_data(), matrix_out.gpu_data());
            }
            else
            {
                double one = 1.0;
                SymmetrizedDiagonalMatrixMatrixProductKernel_Mixed<<<nBlocksPerGrid, nThreadsPerBlock>>>(n, one, diag1.gpu_data(), matrix2.gpu_data(), matrix_out.gpu_data());
                SymmetrizedDiagonalMatrixMatrixProductKernel_Mixed<<<nBlocksPerGrid, nThreadsPerBlock>>>(n, one, diag2.gpu_data(), matrix1.gpu_data(), matrix_out.gpu_data());
            }
        }

        void CopyVectors(const CudaRuntime& cudart, const GPUTensor<double>& X, GPUTensor<float> &Y)
        {
            assert(X.size() == Y.size());
            check_device_alloc(cudart, Y);
            check_device_alloc(cudart, X);
            
            unsigned long long n = X.size();
            CastMatrixDPToSP<<<cudart.gridSize(n,1),cudart.blockSize()>>>(n, X.gpu_data(), Y.gpu_data());
        }

        void CopyVectors(const CudaRuntime& cudart, const GPUTensor<float>& X, GPUTensor<double> &Y)
        {
            assert(X.size() == Y.size());
            unsigned long long n = X.size();
            check_device_alloc(cudart, X);
            check_device_alloc(cudart, Y);

            CastMatrixSPToDP<<<cudart.gridSize(n,1),cudart.blockSize()>>>(n, X.gpu_data(), Y.gpu_data());
        }
    } // namespace gpu

    
    
} // namespace tcgmtensor
