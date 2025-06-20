/* purification.cu */
/* Routines for  purification includin  purification including purifier class */
#include "impl/blas/gpu/additional-level1.hpp"
#include "../../gpu-utils/utils.hpp"

namespace lahva
{
    namespace gpu
    {

        // GPU kernels to take a matrix trace
        // expects that trace is zero at beginning (we must memset externally)
        // compute trace of matrix on the GPU
        __global__ static void MatrixTrace(const double *vecin, const unsigned long long ndim, double *trace)
        {
            // get the global id and thread id (in vector)
            unsigned long long tid = threadIdx.x;
            unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
            // Each block gets its own copy in shared memory
            __shared__ double temp[THREADS_PER_BLOCK];
            double tmp = 0.0;
            while (id < ndim)
            {
                tmp += vecin[id * ndim + id];
                id += blockDim.x * gridDim.x;
            }
            temp[tid] = tmp;
            __syncthreads();

            // for reductions, threadsPerBlock must be a power of 2
            // because of the following code
            int i = blockDim.x / 2;
            while (i != 0)
            {
                if (tid < i)

                    temp[tid] += temp[tid + i];

                __syncthreads();

                i /= 2;
            }
            // Thread 0 adds partial sums to overall sum
            if (tid == 0)
                trace[blockIdx.x] = temp[0]; // per block, we have incremented everything to the first element
        }

        // compute trace of matrix on the GPU (starting from single precision)
        __global__ static void MatrixTrace(const float *vecin, const unsigned long long ndim, float *trace)
        {
            // get the global id and thread id (in vector)
            unsigned long long tid = threadIdx.x;
            unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
            // Each block gets its own copy in shared memory
            __shared__ float temp[THREADS_PER_BLOCK];
            float tmp = 0.0;
            while (id < ndim)
            {
                tmp += vecin[id * ndim + id]; // fast_float2double(vecin[id*ndim+id]);
                id += blockDim.x * gridDim.x;
            }
            temp[tid] = tmp;
            __syncthreads();

            // for reductions, threadsPerBlock must be a power of 2
            // because of the following code
            int i = blockDim.x / 2;
            while (i != 0)
            {
                if (tid < i)

                    temp[tid] += temp[tid + i];

                __syncthreads();

                i /= 2;
            }
            // Thread 0 adds partial sums to overall sum
            if (tid == 0)
                trace[blockIdx.x] = temp[0]; // per block, we have incremented everything to the first element
        }

          // compute trace of matrix on the GPU based on separately stored diagonal
        __global__ static void MatrixTraceFromDiagonal(const float *diag, const unsigned long long ndim, float *trace)
        {
            // get the global id and thread id (in vector)
            unsigned long long tid = threadIdx.x;
            unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
            // Each block gets its own copy in shared memory
            __shared__ float temp[THREADS_PER_BLOCK];
            float tmp = 0.0;
            while (id < ndim)
            {
                tmp += diag[id];
                id += blockDim.x * gridDim.x;
            }
            temp[tid] = tmp;
            __syncthreads();

            // for reductions, threadsPerBlock must be a power of 2
            // because of the following code
            int i = blockDim.x / 2;
            while (i != 0)
            {
                if (tid < i)

                    temp[tid] += temp[tid + i];

                __syncthreads();

                i /= 2;
            }
            // Thread 0 adds partial sums to overall sum
            if (tid == 0)
                trace[blockIdx.x] = temp[0]; // per block, we have incremented everything to the first element
        }

        // compute trace of matrix on the GPU based on separately stored diagonal
        __global__ static void MatrixTraceFromDiagonal(const double *diag, const unsigned long long ndim, double *trace)
        {
            // get the global id and thread id (in vector)
            unsigned long long tid = threadIdx.x;
            unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
            // Each block gets its own copy in shared memory
            __shared__ double temp[THREADS_PER_BLOCK];
            double tmp = 0.0;
            while (id < ndim)
            {
                tmp += diag[id];
                id += blockDim.x * gridDim.x;
            }
            temp[tid] = tmp;
            __syncthreads();

            // for reductions, threadsPerBlock must be a power of 2
            // because of the following code
            int i = blockDim.x / 2;
            while (i != 0)
            {
                if (tid < i)

                    temp[tid] += temp[tid + i];

                __syncthreads();

                i /= 2;
            }
            // Thread 0 adds partial sums to overall sum
            if (tid == 0)
                trace[blockIdx.x] = temp[0]; // per block, we have incremented everything to the first element
        }

        // symmetrize matrix
        __global__ static void SymmetrizeMatrix(unsigned long long ndim, float *matrix)
        {
            unsigned long long id = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned long long jd = threadIdx.x + blockIdx.x * blockDim.x;
            // 0.5*(matrix * diag + matrix^T * diag)
            if (id < ndim && jd < ndim)
            {
                float avg = 0.5 * (matrix[id * ndim + jd] + matrix[jd * ndim + id]);
                matrix[id * ndim + jd] = avg;
                matrix[jd * ndim + id] = avg;
            }
        }
        

        // symmetrize matrix
        __global__ static void SymmetrizeMatrix(unsigned long long ndim, double *matrix)
        {
            unsigned long long id = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned long long jd = threadIdx.x + blockIdx.x * blockDim.x;
            // 0.5*(matrix * diag + matrix^T * diag)
            if (id < ndim && jd < ndim)
            {
                double  avg = 0.5 * (matrix[id * ndim + jd] + matrix[jd * ndim + id]);
                matrix[id * ndim + jd] = avg;
                matrix[jd * ndim + id] = avg;
            }
        }
        
        __global__ static void dGetDiagonal(unsigned long long ndim, const double *matrix, double* diag)
        {
            unsigned long long id = threadIdx.x + blockIdx.x * blockDim.x;
            
            if (id < ndim)
            {
                diag[id] = matrix[id * ndim + id];
            }
        }
         __global__ static void sGetDiagonal(unsigned long long ndim, const float *matrix, float* diag)
        {
            unsigned long long id = threadIdx.x + blockIdx.x * blockDim.x;
            
            if (id < ndim)
            {
                diag[id] = matrix[id * ndim + id];
            }
        }

         __global__ static void dSetDiagonal(unsigned long long ndim, const double* diag, double *matrix)
        {
            unsigned long long id = threadIdx.x + blockIdx.x * blockDim.x;
            
            if (id < ndim)
            {
                matrix[id * ndim + id] = diag[id];
            }
        }
         __global__ static void sSetDiagonal(unsigned long long ndim, const float* diag, float *matrix)
        {
            unsigned long long id = threadIdx.x + blockIdx.x * blockDim.x;
            
            if (id < ndim)
            {
                matrix[id * ndim + id] = diag[id];
            }
        }
        


        template<>        
        void TraceKernelDiag(const CudaRuntime &cudart, unsigned long long ndim, const float* diag, float* vec)
        {
            // Number of blocks in grid;
            int gridS = cudart.gridSize(ndim, 1);
            MatrixTraceFromDiagonal<<<gridS, cudart.blockSize(), cudart.blockSize()*sizeof(double), cudart.getStream()>>>(diag, ndim, vec);
        }

        template<>        
        void TraceKernelDiag(const CudaRuntime &cudart, unsigned long long ndim, const double* diag, double* vec)
        {
            // Number of blocks in grid;
            int gridS = cudart.gridSize(ndim, 1);
            MatrixTraceFromDiagonal<<<gridS, cudart.blockSize(), cudart.blockSize()*sizeof(double), cudart.getStream()>>>(diag, ndim, vec);
        }

        template<>        
        void TraceKernel(const CudaRuntime &cudart, unsigned long long ndim, const float* diag, float* vec)
        {
            // Number of blocks in grid;
            int gridS = cudart.gridSize(ndim, 1);
            MatrixTrace<<<gridS, cudart.blockSize(), cudart.blockSize()*sizeof(double), cudart.getStream()>>>(diag, ndim, vec);
        }

        template<>        
        void TraceKernel(const CudaRuntime &cudart, unsigned long long ndim, const double* diag, double* vec)
        {
            // Number of blocks in grid;
            int gridS = cudart.gridSize(ndim, 1);
            MatrixTrace<<<gridS, cudart.blockSize(), cudart.blockSize()*sizeof(double), cudart.getStream()>>>(diag, ndim, vec);
        }

        template <>
        void SymmetrizeMatrix<float>(const CudaRuntime &cudart, Matrix_<float> &m)
        {
            // Number of blocks in grid;
            int N = m.shape().first;
            dim3 blockSize(16, 16); // 16x16 threads per block
            dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
            check_device_alloc(cudart, m);
            SymmetrizeMatrix<<<gridSize, blockSize, 0, cudart.getStream()>>>(m.shape().first, m.gpu_data()); // compute trace
        }

        template <>
        void SymmetrizeMatrix<double>(const CudaRuntime &cudart, Matrix_<double> &m)
        {
            // Number of blocks in grid;
            int N = m.shape().first;
            dim3 blockSize(16, 16); // 16x16 threads per block
            dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
            check_device_alloc(cudart, m);
            SymmetrizeMatrix<<<gridSize, blockSize, 0, cudart.getStream()>>>(m.shape().first, m.gpu_data()); // compute trace
        }

        template<>
        void GetDiagonal<double>(const CudaRuntime& cudart, const Matrix_<double>& m, GPUTensor_<double>& diag)
        {
            int gridS = cudart.gridSize(diag.size(), 1);
            check_device_alloc(cudart, diag);
            check_device_alloc(cudart, m);
            dGetDiagonal<<<gridS, cudart.blockSize(), 0, cudart.getStream()>>>(diag.size(), m.gpu_data(), diag.gpu_data());

        }
        
        template<>
        void GetDiagonal<float>(const CudaRuntime& cudart, const Matrix_<float>& m, GPUTensor_<float>& diag)
        {
            int gridS = cudart.gridSize(diag.size(), 1);
            check_device_alloc(cudart, diag);
            check_device_alloc(cudart, m);
            sGetDiagonal<<<gridS, cudart.blockSize(), 0, cudart.getStream()>>>(diag.size(), m.gpu_data(), diag.gpu_data());

        }

        template<>
        void SetDiagonal<double>(const CudaRuntime& cudart, const GPUTensor_<double>& diag, Matrix_<double>& m)
        {
            int gridS = cudart.gridSize(diag.size(), 1);
            check_device_alloc(cudart, diag);
            check_device_alloc(cudart, m);
            dSetDiagonal<<<gridS, cudart.blockSize(), 0, cudart.getStream()>>>(diag.size(), diag.gpu_data(), m.gpu_data());

        }

        template<>
        void SetDiagonal<float>(const CudaRuntime& cudart, const GPUTensor_<float>& diag, Matrix_<float>& m)
        {
            int gridS = cudart.gridSize(diag.size(), 1);
            check_device_alloc(cudart, diag);
            check_device_alloc(cudart, m);
            sSetDiagonal<<<gridS, cudart.blockSize(), 0, cudart.getStream()>>>(diag.size(), diag.gpu_data(), m.gpu_data());

        }

    } // namespace Purification_kernel
} // namespace Purification