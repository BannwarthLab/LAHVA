/// @file trace.cu
/// @brief GPU kernels for matrix trace, symmetrization, and diagonal operations.
///
/// Provides CUDA kernels for computing matrix traces, symmetrizing matrices,
/// and extracting/setting diagonal elements efficiently on GPU.

#include "impl/blas/gpu/additional-level1.hpp"
#include "../../gpu-utils/utils.hpp"

namespace lahva
{
    namespace gpu
    {
        /// @brief GPU kernel to compute matrix trace for double-precision matrices.
        ///
        /// Computes the sum of diagonal elements using parallel reduction.
        /// Output trace must be zero-initialized; partial sums are written per-block.
        /// Block size must be a power of 2 for the reduction algorithm.
        ///
        /// @param vecin Input matrix in row-major order.
        /// @param ndim Matrix dimension (square matrix).
        /// @param trace Output vector of partial sums (one per block).
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

        /// @brief GPU kernel to compute matrix trace for single-precision matrices.
        ///
        /// Computes the sum of diagonal elements using parallel reduction.
        /// Output trace must be zero-initialized; partial sums are written per-block.
        /// Block size must be a power of 2 for the reduction algorithm.
        ///
        /// @param vecin Input matrix in row-major order.
        /// @param ndim Matrix dimension (square matrix).
        /// @param trace Output vector of partial sums (one per block).
        __global__ static void MatrixTrace(const float *vecin, const unsigned long long ndim, float *trace)
        {
            unsigned long long tid = threadIdx.x;
            unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
            __shared__ float temp[THREADS_PER_BLOCK];
            float tmp = 0.0;
            while (id < ndim)
            {
                tmp += vecin[id * ndim + id];
                id += blockDim.x * gridDim.x;
            }
            temp[tid] = tmp;
            __syncthreads();

            int i = blockDim.x / 2;
            while (i != 0)
            {
                if (tid < i)
                    temp[tid] += temp[tid + i];

                __syncthreads();

                i /= 2;
            }
            if (tid == 0)
                trace[blockIdx.x] = temp[0];
        }

        /// @brief GPU kernel to compute matrix trace from pre-computed diagonal vector (single-precision).
        ///
        /// Sums diagonal elements stored separately, more efficient when diagonal is already extracted.
        /// Uses parallel reduction with per-block partial sums.
        ///
        /// @param diag Pre-extracted diagonal vector.
        /// @param ndim Vector dimension.
        /// @param trace Output vector of partial sums (one per block).
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

        /// @brief GPU kernel to compute matrix trace from pre-computed diagonal vector (double-precision).
        ///
        /// Sums diagonal elements stored separately, more efficient when diagonal is already extracted.
        /// Uses parallel reduction with per-block partial sums.
        ///
        /// @param diag Pre-extracted diagonal vector.
        /// @param ndim Vector dimension.
        /// @param trace Output vector of partial sums (one per block).
        __global__ static void MatrixTraceFromDiagonal(const double *diag, const unsigned long long ndim, double *trace)
        {
            unsigned long long tid = threadIdx.x;
            unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
            __shared__ double temp[THREADS_PER_BLOCK];
            double tmp = 0.0;
            while (id < ndim)
            {
                tmp += diag[id];
                id += blockDim.x * gridDim.x;
            }
            temp[tid] = tmp;
            __syncthreads();

            int i = blockDim.x / 2;
            while (i != 0)
            {
                if (tid < i)
                    temp[tid] += temp[tid + i];

                __syncthreads();

                i /= 2;
            }
            if (tid == 0)
                trace[blockIdx.x] = temp[0];
        }

        /// @brief GPU kernel to symmetrize single-precision matrix in-place.
        ///
        /// Makes matrix symmetric by averaging each element with its transpose:
        /// M[i,j] = M[j,i] = 0.5 * (M[i,j] + M[j,i]).
        /// Uses 2D grid for better thread-to-element mapping.
        ///
        /// @param ndim Matrix dimension (square matrix).
        /// @param matrix Input/output single-precision matrix in row-major order.
        __global__ static void SymmetrizeMatrix(unsigned long long ndim, float *matrix)
        {
            unsigned long long id = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned long long jd = threadIdx.x + blockIdx.x * blockDim.x;

            if (id < ndim && jd < ndim)
            {
                float avg = 0.5 * (matrix[id * ndim + jd] + matrix[jd * ndim + id]);
                matrix[id * ndim + jd] = avg;
                matrix[jd * ndim + id] = avg;
            }
        }
        

        /// @brief GPU kernel to symmetrize double-precision matrix in-place.
        ///
        /// Makes matrix symmetric by averaging each element with its transpose:
        /// M[i,j] = M[j,i] = 0.5 * (M[i,j] + M[j,i]).
        /// Uses 2D grid for better thread-to-element mapping.
        ///
        /// @param ndim Matrix dimension (square matrix).
        /// @param matrix Input/output double-precision matrix in row-major order.
        __global__ static void SymmetrizeMatrix(unsigned long long ndim, double *matrix)
        {
            unsigned long long id = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned long long jd = threadIdx.x + blockIdx.x * blockDim.x;
            if (id < ndim && jd < ndim)
            {
                double  avg = 0.5 * (matrix[id * ndim + jd] + matrix[jd * ndim + id]);
                matrix[id * ndim + jd] = avg;
                matrix[jd * ndim + id] = avg;
            }
        }

        /// @brief GPU kernel to extract diagonal from double-precision matrix.
        ///
        /// Copies diagonal elements to a separate vector: diag[i] = matrix[i,i].
        ///
        /// @param ndim Matrix dimension.
        /// @param matrix Input double-precision matrix in row-major order.
        /// @param diag Output diagonal vector.
        __global__ static void dGetDiagonal(unsigned long long ndim, const double *matrix, double* diag)
        {
            unsigned long long id = threadIdx.x + blockIdx.x * blockDim.x;
            if (id < ndim)
            {
                diag[id] = matrix[id * ndim + id];
            }
        }

        /// @brief GPU kernel to extract diagonal from single-precision matrix.
        ///
        /// Copies diagonal elements to a separate vector: diag[i] = matrix[i,i].
        ///
        /// @param ndim Matrix dimension.
        /// @param matrix Input single-precision matrix in row-major order.
        /// @param diag Output diagonal vector.
        __global__ static void sGetDiagonal(unsigned long long ndim, const float *matrix, float* diag)
        {
            unsigned long long id = threadIdx.x + blockIdx.x * blockDim.x;
            if (id < ndim)
            {
                diag[id] = matrix[id * ndim + id];
            }
        }

        /// @brief GPU kernel to set diagonal of double-precision matrix.
        ///
        /// Sets diagonal elements from a vector: matrix[i,i] = diag[i].
        ///
        /// @param ndim Matrix dimension.
        /// @param diag Input diagonal vector.
        /// @param matrix Input/output double-precision matrix in row-major order.
        __global__ static void dSetDiagonal(unsigned long long ndim, const double* diag, double *matrix)
        {
            unsigned long long id = threadIdx.x + blockIdx.x * blockDim.x;
            if (id < ndim)
            {
                matrix[id * ndim + id] = diag[id];
            }
        }

        /// @brief GPU kernel to set diagonal of single-precision matrix.
        ///
        /// Sets diagonal elements from a vector: matrix[i,i] = diag[i].
        ///
        /// @param ndim Matrix dimension.
        /// @param diag Input diagonal vector.
        /// @param matrix Input/output single-precision matrix in row-major order.
        __global__ static void sSetDiagonal(unsigned long long ndim, const float* diag, float *matrix)
        {
            unsigned long long id = threadIdx.x + blockIdx.x * blockDim.x;
            if (id < ndim)
            {
                matrix[id * ndim + id] = diag[id];
            }
        }
        


        /// @brief Host wrapper for matrix trace from diagonal (single-precision).
        ///
        /// Launches GPU kernel to compute trace from pre-extracted diagonal vector.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param ndim Vector dimension.
        /// @param diag Pre-extracted diagonal vector.
        /// @param vec Output partial sums (one per block).
        template<>
        void TraceKernelDiag(const CudaRuntime &cudart, unsigned long long ndim, const float* diag, float* vec)
        {
            int gridS = cudart.gridSize(ndim, 1);
            MatrixTraceFromDiagonal<<<gridS, cudart.blockSize(), cudart.blockSize()*sizeof(double), cudart.getStream()>>>(diag, ndim, vec);
        }

        /// @brief Host wrapper for matrix trace from diagonal (double-precision).
        ///
        /// Launches GPU kernel to compute trace from pre-extracted diagonal vector.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param ndim Vector dimension.
        /// @param diag Pre-extracted diagonal vector.
        /// @param vec Output partial sums (one per block).
        template<>
        void TraceKernelDiag(const CudaRuntime &cudart, unsigned long long ndim, const double* diag, double* vec)
        {
            int gridS = cudart.gridSize(ndim, 1);
            MatrixTraceFromDiagonal<<<gridS, cudart.blockSize(), cudart.blockSize()*sizeof(double), cudart.getStream()>>>(diag, ndim, vec);
        }

        /// @brief Host wrapper for matrix trace (single-precision).
        ///
        /// Launches GPU kernel to compute trace from full matrix.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param ndim Matrix dimension.
        /// @param diag Input matrix (treated as pointer to data).
        /// @param vec Output partial sums (one per block).
        template<>
        void TraceKernel(const CudaRuntime &cudart, unsigned long long ndim, const float* diag, float* vec)
        {
            int gridS = cudart.gridSize(ndim, 1);
            MatrixTrace<<<gridS, cudart.blockSize(), cudart.blockSize()*sizeof(double), cudart.getStream()>>>(diag, ndim, vec);
        }

        /// @brief Host wrapper for matrix trace (double-precision).
        ///
        /// Launches GPU kernel to compute trace from full matrix.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param ndim Matrix dimension.
        /// @param diag Input matrix (treated as pointer to data).
        /// @param vec Output partial sums (one per block).
        template<>
        void TraceKernel(const CudaRuntime &cudart, unsigned long long ndim, const double* diag, double* vec)
        {
            int gridS = cudart.gridSize(ndim, 1);
            MatrixTrace<<<gridS, cudart.blockSize(), cudart.blockSize()*sizeof(double), cudart.getStream()>>>(diag, ndim, vec);
        }

        /// @brief Host wrapper to symmetrize single-precision matrix in-place.
        ///
        /// Launches GPU kernel with 2D grid to make matrix symmetric.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param m Input/output single-precision matrix (replaced with symmetric result).
        template <>
        void SymmetrizeMatrix<float>(const CudaRuntime &cudart, Matrix_<float> &m)
        {
            int N = m.shape().first;
            dim3 blockSize(16, 16);
            dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
            check_device_alloc(cudart, m);
            SymmetrizeMatrix<<<gridSize, blockSize, 0, cudart.getStream()>>>(m.shape().first, m.gpu_data());
        }

        /// @brief Host wrapper to symmetrize double-precision matrix in-place.
        ///
        /// Launches GPU kernel with 2D grid to make matrix symmetric.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param m Input/output double-precision matrix (replaced with symmetric result).
        template <>
        void SymmetrizeMatrix<double>(const CudaRuntime &cudart, Matrix_<double> &m)
        {
            int N = m.shape().first;
            dim3 blockSize(16, 16);
            dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
            check_device_alloc(cudart, m);
            SymmetrizeMatrix<<<gridSize, blockSize, 0, cudart.getStream()>>>(m.shape().first, m.gpu_data());
        }

        /// @brief Host wrapper to extract diagonal from double-precision matrix.
        ///
        /// Launches GPU kernel to copy diagonal elements to a separate vector.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param m Input double-precision matrix.
        /// @param diag Output diagonal vector.
        template<>
        void GetDiagonal<double>(const CudaRuntime& cudart, const Matrix_<double>& m, GPUTensor_<double>& diag)
        {
            int gridS = cudart.gridSize(diag.size(), 1);
            check_device_alloc(cudart, diag);
            check_device_alloc(cudart, m);
            dGetDiagonal<<<gridS, cudart.blockSize(), 0, cudart.getStream()>>>(diag.size(), m.gpu_data(), diag.gpu_data());
        }

        /// @brief Host wrapper to extract diagonal from single-precision matrix.
        ///
        /// Launches GPU kernel to copy diagonal elements to a separate vector.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param m Input single-precision matrix.
        /// @param diag Output diagonal vector.
        template<>
        void GetDiagonal<float>(const CudaRuntime& cudart, const Matrix_<float>& m, GPUTensor_<float>& diag)
        {
            int gridS = cudart.gridSize(diag.size(), 1);
            check_device_alloc(cudart, diag);
            check_device_alloc(cudart, m);
            sGetDiagonal<<<gridS, cudart.blockSize(), 0, cudart.getStream()>>>(diag.size(), m.gpu_data(), diag.gpu_data());
        }

        /// @brief Host wrapper to set diagonal of double-precision matrix.
        ///
        /// Launches GPU kernel to set diagonal elements from a vector.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param diag Input diagonal vector.
        /// @param m Input/output double-precision matrix (diagonal updated).
        template<>
        void SetDiagonal<double>(const CudaRuntime& cudart, const GPUTensor_<double>& diag, Matrix_<double>& m)
        {
            int gridS = cudart.gridSize(diag.size(), 1);
            check_device_alloc(cudart, diag);
            check_device_alloc(cudart, m);
            dSetDiagonal<<<gridS, cudart.blockSize(), 0, cudart.getStream()>>>(diag.size(), diag.gpu_data(), m.gpu_data());
        }

        /// @brief Host wrapper to set diagonal of single-precision matrix.
        ///
        /// Launches GPU kernel to set diagonal elements from a vector.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param diag Input diagonal vector.
        /// @param m Input/output single-precision matrix (diagonal updated).
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