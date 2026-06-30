/// @file mixed-precision.cu
/// @brief GPU kernels for mixed-precision matrix decomposition and iterative refinement.
///
/// Provides CUDA kernels for matrix decomposition into lower-precision components,
/// mixed-precision arithmetic operations, and type conversion for iterative refinement algorithms.

#include "linalg.hpp"
#include "runtime.hpp"
#include "../../gpu-utils/utils.hpp"
#include "impl/blas/gpu/level1.hpp"
#include "impl/blas/gpu/additional-level1.hpp"
#include "impl/blas/gpu/additional-level2.hpp"
#include "common.h"
#include "impl/tensor/allocators.hpp"
#include <vector>

namespace lahva
{
    namespace gpu
    {
        /// @brief GPU kernel for symmetrized diagonal-scaled matrix product with mixed precision.
        ///
        /// Computes element-wise scaled matrix operation equivalent to:
        /// matrixOut += 0.5 * (matrix * diag + matrix^T * diag)
        /// This creates a symmetrized result by averaging the scaled matrix with its transpose scaled.
        /// Implemented as element-wise: matrixOut[i,j] += 0.5 * scale * (diag[i] + diag[j]) * matrixIn[i,j].
        ///
        /// @param ndim Matrix dimension.
        /// @param scale Scaling factor applied to the operation.
        /// @param diag Double-precision diagonal scaling vector.
        /// @param matrixIn Input single-precision matrix.
        /// @param matrixOut Output single-precision matrix (accumulated).
        __global__ static void SymmetrizedDiagonalMatrixMatrixProductKernel_Mixed(unsigned long long ndim, const double scale, const double *diag,
                                                                                  const float *matrixIn, float *matrixOut)
        {
            unsigned long long id = blockIdx.y;
            unsigned long long jd = threadIdx.x + blockIdx.x * blockDim.x;

            if (id < ndim && jd < ndim)
            {
                matrixOut[id * ndim + jd] += ((float)(0.5 * scale * (diag[id] + diag[jd])) * matrixIn[id * ndim + jd]);
            }
        }

        /// @brief GPU kernel for type-conversion matrix copy with mixed precision.
        ///
        /// Converts and copies matrix elements from input to output type.
        /// MatOut[i] = (outprec)(MatIn[i]).
        ///
        /// @tparam inprec Input precision type.
        /// @tparam outprec Output precision type.
        /// @param ntot Total number of elements.
        /// @param MatIn Input matrix in input precision.
        /// @param MatOut Output matrix in output precision.
        template <typename inprec, typename outprec>
        __global__ void CastMatrix(const unsigned long long ntot, const inprec *MatIn, outprec *MatOut)
        {
            unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < ntot)
                MatOut[id] = (outprec)(MatIn[id]);
            __syncthreads();
        }

        /// @brief GPU kernel for mixed-precision vector accumulation.
        ///
        /// Accumulates scaled values with type conversion: MatOut[i] += (outprec)(MatIn[i]).
        ///
        /// @tparam inprec Input precision type.
        /// @tparam outprec Output precision type.
        /// @param ntot Total number of elements.
        /// @param MatIn Input matrix in input precision.
        /// @param MatOut Output matrix in output precision (accumulated).
        template <typename inprec, typename outprec>
        __global__ void IncrVectors(const unsigned long long ntot, const inprec *MatIn, outprec *MatOut)
        {
            unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < ntot)
                MatOut[id] += (outprec)(MatIn[id]);
            __syncthreads();
        }

        /// @brief GPU kernel for mixed-precision scaled vector addition (AXPY) with FMA.
        ///
        /// Computes scaled vector addition with fused multiply-add: MatOut[i] = alpha*MatIn[i] + MatOut[i].
        /// Uses correctly rounded fused multiply-add (FMA) operations for precision preservation.
        ///
        /// @tparam inprec Input precision type.
        /// @tparam outprec Output precision type.
        /// @param ntot Total number of elements.
        /// @param alpha Scalar factor for input vector.
        /// @param MatIn Input matrix in input precision.
        /// @param MatOut Output matrix in output precision (accumulated).
        template <typename inprec, typename outprec>
        __global__ void AxpyVectors(const unsigned long long ntot, const inprec alpha, const inprec *__restrict__ MatIn, outprec *MatOut)
        {
            const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t stride = blockDim.x * gridDim.x;
#pragma unroll
            for (size_t i = id; i < ntot; i += stride)
            {
                MatOut[i] = getFMA<outprec>((outprec)(MatIn[i]), (outprec)(alpha), MatOut[i]); 
            }
        }

        /// @brief Truncates value to specified number of significant decimal digits.
        ///
        /// Performs precision truncation by keeping only the specified number of
        /// significant decimal digits for precision control in iterative refinement.
        ///
        /// @tparam T Floating-point type.
        /// @param value Input value to truncate.
        /// @param significantDigits Number of significant digits to keep.
        /// @return Truncated value.
        template <typename T>
        __device__ T truncateSignificantDigits(T value, int significantDigits)
        {
            if (value == 0.0)
            {
                return 0.0;
            }

            T factor = pow(10.0, significantDigits);
            return trunc(value * factor) / factor;
        }

        /// @brief Specialized GPU kernel for mixed-precision vector accumulation (float to double).
        ///
        /// Accumulates float values into double with truncation for precision control:
        /// MatOut[i] += truncateSignificantDigits(MatIn[i], 8).
        /// Specialized for float-to-double conversion with digit truncation.
        ///
        /// @param ntot Total number of elements.
        /// @param MatIn Input single-precision matrix.
        /// @param MatOut Output double-precision matrix (accumulated).
        template <>
        __global__ void IncrVectors(const unsigned long long ntot, const float *MatIn, double *MatOut)
        {
            unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < ntot)
                MatOut[id] += truncateSignificantDigits(MatIn[id], 8);
            __syncthreads();
        }

        /// @brief GPU kernel for matrix decomposition into split components (1D iteration).
        ///
        /// Decomposes matrix elements into high-precision mantissa and lower-precision components
        /// for iterative refinement. Updates both the split output and residual input.
        ///
        /// @tparam inprec Input/residual precision type.
        /// @tparam outprec Output split component precision type.
        /// @tparam blockSize CUDA block size
        /// @param ntot Total number of elements.
        /// @param MatIn Input/output matrix (contains residuals after decomposition).
        /// @param SplitOut Output split components in lower precision.
        /// @param tau Exponent for scaling split components.
        /// @param sigma Rounding constant for precision control.
        template <typename inprec, typename outprec, size_t blockSize>
        __global__ void DecomposeMatrixKernel(const unsigned long long ntot, inprec *__restrict__ MatIn, outprec *__restrict__ SplitOut, const int tau, const inprec sigma)
        {
            size_t const idx = blockIdx.x * blockSize + threadIdx.x;
            size_t const stride = blockSize * gridDim.x;
            const int tauneg = -tau;

#pragma unroll
            for (size_t i{idx}; i < ntot; i += stride)
            {
                const inprec inp = MatIn[i];
                const inprec split = (inp + sigma) - sigma;
                outprec scaled;
                if constexpr (std::is_same<inprec, float>::value)
                    scaled = scalbnf(split, tauneg);
                else
                    scaled = scalbn(split, tauneg);
                SplitOut[i] = scaled;
                MatIn[i] = inp - split;
            }
        }

        /// @brief GPU kernel for matrix decomposition into split components (2D tiling).
        ///
        /// Decomposes matrix elements into high-precision mantissa and lower-precision components
        /// using 2D block tiling for better memory access patterns and locality.
        ///
        /// @tparam inprec Input/residual precision type.
        /// @tparam outprec Output split component precision type.
        /// @tparam TILE_SIZE 2D tile size
        /// @param nrows Number of rows in the matrix.
        /// @param ncols Number of columns in the matrix.
        /// @param MatIn Input/output matrix (contains residuals after decomposition).
        /// @param SplitOut Output split components in lower precision.
        /// @param tau Exponent for scaling split components.
        /// @param sigma Rounding constant for precision control.
        template <typename inprec, typename outprec, size_t TILE_SIZE>
        __global__ void DecomposeMatrixKernel2D(const unsigned long long nrows, const unsigned long long ncols,
                                                inprec *__restrict__ MatIn, outprec *__restrict__ SplitOut,
                                                const int tau, const inprec sigma)
        {
            unsigned long long row = blockIdx.y * TILE_SIZE + threadIdx.y;
            unsigned long long col = blockIdx.x * TILE_SIZE + threadIdx.x;
            unsigned long long idx = row * ncols + col;
            const int tauneg = -tau;

            if (row < nrows && col < ncols)
            {
                const inprec inp = MatIn[idx];
                const inprec split = ((inp + sigma) - sigma);
                outprec scaled;
                if constexpr (std::is_same<inprec, float>::value)
                    scaled = scalbnf(split, tauneg);
                else
                    scaled = scalbn(split, tauneg);
                SplitOut[idx] = scaled;
                MatIn[idx] = inp - split;
            }
        }

        /// @brief Host wrapper for symmetrized diagonal-scaled matrix product with mixed precision.
        ///
        /// Computes matrix product with diagonal scaling and symmetrization:
        /// When matrices match: applies 2.0 * scale factor.
        /// When matrices differ: applies two separate 1.0 * scale operations.
        /// Result is accumulated into matrix_out.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param diag1 First diagonal scaling vector (double-precision).
        /// @param matrix1 First input matrix (single-precision).
        /// @param diag2 Second diagonal scaling vector (double-precision).
        /// @param matrix2 Second input matrix (single-precision).
        /// @param matrix_out Output matrix (single-precision, accumulated).
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

        /// @brief Host wrapper for mixed-precision vector accumulation.
        ///
        /// Launches GPU kernel to accumulate values with type conversion: Y[i] += (out)(X[i]).
        ///
        /// @tparam in Input precision type.
        /// @tparam out Output precision type.
        /// @param cudart CUDA runtime instance.
        /// @param X Input tensor.
        /// @param Y Input/output tensor (accumulated).
        template <typename in, typename out>
        void AddVectors(const CudaRuntime &cudart, const GPUTensor_<in> &X, GPUTensor_<out> &Y)
        {
            assert(X.size() == Y.size());
            check_device_alloc(cudart, Y);
            check_device_alloc(cudart, X);
            unsigned long long n = X.size();
            IncrVectors<in, out><<<cudart.gridSize(n, 1), cudart.blockSize(), 0, cudart.getStream()>>>(n, X.gpu_data(), Y.gpu_data());
        }

        /// @brief Host wrapper for scaled vector addition from half-precision to single-precision.
        ///
        /// Launches GPU kernel to compute: Y[i] = alpha*X[i] + Y[i] with FMA.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param alpha Scalar factor for half-precision input.
        /// @param X Input half-precision tensor.
        /// @param Y Input/output single-precision tensor.
        void AddVectors(const CudaRuntime &cudart, float alpha, const GPUTensor_<__half> &X, GPUTensor_<float> &Y)
        {
            assert(X.size() == Y.size());
            check_device_alloc(cudart, Y);
            check_device_alloc(cudart, X);
            unsigned long long n = X.size();
            AxpyVectors<__half, float><<<cudart.gridSize(n, 1) / 8, cudart.blockSize(), 0, cudart.getStream()>>>(n, alpha, X.gpu_data(), Y.gpu_data());
        }

        /// @brief Host wrapper for scaled vector addition from half-precision to double-precision.
        ///
        /// Launches GPU kernel to compute: Y[i] = alpha*X[i] + Y[i] with FMA.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param alpha Scalar factor for half-precision input.
        /// @param X Input half-precision tensor.
        /// @param Y Input/output double-precision tensor.
        void AddVectors(const CudaRuntime &cudart, double alpha, const GPUTensor_<__half> &X, GPUTensor_<double> &Y)
        {
            assert(X.size() == Y.size());
            check_device_alloc(cudart, Y);
            check_device_alloc(cudart, X);
            unsigned long long n = X.size();
            AxpyVectors<__half, double><<<cudart.gridSize(n, 1) / 2, cudart.blockSize(), 0, cudart.getStream()>>>(n, alpha, X.gpu_data(), Y.gpu_data());
        }

        /// @brief Host wrapper for mixed-precision matrix copy (double to float).
        ///
        /// Launches GPU kernel to convert and copy: Y[i] = (float)(X[i]).
        ///
        /// @param cudart CUDA runtime instance.
        /// @param X Input double-precision tensor.
        /// @param Y Output single-precision tensor.
        void CopyVectors(const CudaRuntime &cudart, const GPUTensor_<double> &X, GPUTensor_<float> &Y)
        {
            assert(X.size() == Y.size());
            check_device_alloc(cudart, Y);
            check_device_alloc(cudart, X);

            unsigned long long n = X.size();
            CastMatrix<double, float><<<cudart.gridSize(n, 1), cudart.blockSize(), 0, cudart.getStream()>>>(n, X.gpu_data(), Y.gpu_data());
        }

        /// @brief Host wrapper for mixed-precision matrix copy (float to double).
        ///
        /// Launches GPU kernel to convert and copy: Y[i] = (double)(X[i]).
        ///
        /// @param cudart CUDA runtime instance.
        /// @param X Input single-precision tensor.
        /// @param Y Output double-precision tensor.
        void CopyVectors(const CudaRuntime &cudart, const GPUTensor_<float> &X, GPUTensor_<double> &Y)
        {
            assert(X.size() == Y.size());
            unsigned long long n = X.size();
            check_device_alloc(cudart, X);
            check_device_alloc(cudart, Y);

            CastMatrix<float, double><<<cudart.gridSize(n, 1), cudart.blockSize(), 0, cudart.getStream()>>>(n, X.gpu_data(), Y.gpu_data());
        }

        /// @brief Host wrapper for vector decomposition into two mixed-precision components.
        ///
        /// Decomposes an input vector into two lower-precision components for iterative refinement.
        /// Uses dynamic exponent selection based on maximum element magnitude.
        ///
        /// @tparam inprec Input precision type (higher precision).
        /// @tparam outprec Output precision type (lower precision for components).
        /// @param cudart CUDA runtime instance.
        /// @param in Input vector to decompose.
        /// @param out1 First decomposed component (lower precision).
        /// @param out2 Second decomposed component (lower precision).
        /// @param coeff Exponent coefficients used in decomposition (output).
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
                    DecomposeMatrixKernel<inprec, outprec, 512><<<cudart.gridSize(n, 1), cudart.blockSize(), 0, cudart.getStream()>>>(n, CopyIn.gpu_data(), out1.gpu_data(), tau, sigma);
                }
                else
                {
                    DecomposeMatrixKernel<inprec, outprec, 512><<<cudart.gridSize(n, 1), cudart.blockSize(), 0, cudart.getStream()>>>(n, CopyIn.gpu_data(), out2.gpu_data(), tau, sigma);
                }
            }
        }

        /// @brief Host wrapper for matrix decomposition into multiple mixed-precision components.
        ///
        /// Iteratively decomposes an input matrix into multiple lower-precision components
        /// for iterative refinement. Each iteration selects the maximum element and computes
        /// the appropriate exponent for that split. Supports adaptive grid sizing based on block size.
        ///
        /// @tparam inprec Input precision type (higher precision).
        /// @tparam outprec Output precision type (lower precision for components).
        /// @tparam Allocator Host memory allocator type for output matrices.
        /// @tparam GPUAllocator Device memory allocator type for output matrices.
        /// @param cudart CUDA runtime instance.
        /// @param in Input matrix to decompose (updated with residuals).
        /// @param out Vector of output matrices with decomposed components.
        /// @param coeff Vector of exponent coefficients (one per split component).
        /// @param maxsplit Maximum number of split components to create.
        template <typename inprec, typename outprec, typename Allocator, typename GPUAllocator>
        void SplitMatrix(const CudaRuntime &cudart, Matrix_<inprec> &in, std::vector<Matrix<outprec, Allocator, GPUAllocator>> &out, GPUTensor_<int> &coeff, int maxsplit)
        {
            unsigned long long n = in.size();
            size_t numel = std::sqrt(n);

            check_device_alloc(cudart, in);
            int rho;
            if constexpr (std::is_same<inprec, float>::value)
            {
                rho = ceil(getEpse<inprec>() - (getEpse<outprec>() - log2(2)) / 2);
            }
            else
            {
                if (numel > 3500)
                    rho = ceil(getEpse<inprec>() - (getEpse<outprec>() - log2(32)) / 2);
                else
                    rho = ceil(getEpse<inprec>() - (getEpse<outprec>() - log2(32)) / 2);
            }
            for (int i = 0; i < maxsplit; i++)
            {
                check_device_alloc(cudart, out[i]);
                assert(out[i].size() == in.size());
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

                inprec max;
                get_cuda_error(cudaMemcpyAsync(&max, &in.gpu_data()[idmax - 1], sizeof(inprec), cudaMemcpyDeviceToHost, cudart.getStream()));
                cudart.synchronize();
                inprec mu = abs(max);
                int tau = ceil(log2(mu));
                inprec sigma = scalbn(1.0, rho + tau)*0.75;
                coeff[i] = tau;
                n = in.size();
                dim3 blockSize(cudart.blockSize(), 1);
                int computeperThread = 1;
                dim3 gridSize(cudart.gridSize(n, 1) / computeperThread, 1);
                switch (blockSize.x)
                {
                case 1024:
                    DecomposeMatrixKernel<inprec, outprec, 1024><<<gridSize, blockSize, 0, cudart.getStream()>>>(n, in.gpu_data(), out[i].gpu_data(), tau, sigma);
                    break;
                case 512:
                    DecomposeMatrixKernel<inprec, outprec, 512><<<gridSize, blockSize, 0, cudart.getStream()>>>(n, in.gpu_data(), out[i].gpu_data(), tau, sigma);
                    break;
                case 256:
                    DecomposeMatrixKernel<inprec, outprec, 256><<<gridSize, blockSize, 0, cudart.getStream()>>>(n, in.gpu_data(), out[i].gpu_data(), tau, sigma);
                    break;
                case 128:
                    DecomposeMatrixKernel<inprec, outprec, 128><<<gridSize, blockSize, 0, cudart.getStream()>>>(n, in.gpu_data(), out[i].gpu_data(), tau, sigma);
                    break;
                case 64:
                    DecomposeMatrixKernel<inprec, outprec, 64><<<gridSize, blockSize, 0, cudart.getStream()>>>(n, in.gpu_data(), out[i].gpu_data(), tau, sigma);
                    break;
                case 32:
                    DecomposeMatrixKernel<inprec, outprec, 32><<<gridSize, blockSize, 0, cudart.getStream()>>>(n, in.gpu_data(), out[i].gpu_data(), tau, sigma);
                    break;

                default:
                    break;
                }
            }
        }

        template void AddVectors<float, double>(const CudaRuntime &cudart, const GPUTensor_<float> &X, GPUTensor_<double> &Y);
        template void AddVectors<double, float>(const CudaRuntime &cudart, const GPUTensor_<double> &X, GPUTensor_<float> &Y);
        template void AddVectors<__half, double>(const CudaRuntime &cudart, const GPUTensor_<__half> &X, GPUTensor_<double> &Y);
        template void AddVectors<__half, float>(const CudaRuntime &cudart, const GPUTensor_<__half> &X, GPUTensor_<float> &Y);

        template void DecomposeVector2MP<float, __half>(const CudaRuntime &cudart, const GPUTensor_<float> &in, GPUTensor_<__half> &out1, GPUTensor_<__half> &out2, GPUTensor_<int> &coeff);
        template void SplitMatrix<float, __half, CudaHostAllocator<__half>, CudaDeviceAllocator<__half>>(const CudaRuntime &cudart, Matrix_<float> &in, std::vector<Matrix<__half, CudaHostAllocator<__half>, CudaDeviceAllocator<__half>>> &out, GPUTensor_<int> &coeff, int maxsplit);
        template void SplitMatrix<float, __half, CudaHostAllocator<__half>, CudaDeviceAsyncAllocator<__half>>(const CudaRuntime &cudart, Matrix_<float> &in, std::vector<Matrix<__half, CudaHostAllocator<__half>, CudaDeviceAsyncAllocator<__half>>> &out, GPUTensor_<int> &coeff, int maxsplit);
        template void SplitMatrix<double, __half, CudaHostAllocator<__half>, CudaDeviceAllocator<__half>>(const CudaRuntime &cudart, Matrix_<double> &in, std::vector<Matrix<__half, CudaHostAllocator<__half>, CudaDeviceAllocator<__half>>> &out, GPUTensor_<int> &coeff, int maxsplit);
        template void SplitMatrix<double, __half, CudaHostAllocator<__half>, CudaDeviceAsyncAllocator<__half>>(const CudaRuntime &cudart, Matrix_<double> &in, std::vector<Matrix<__half, CudaHostAllocator<__half>, CudaDeviceAsyncAllocator<__half>>> &out, GPUTensor_<int> &coeff, int maxsplit);
    } // namespace gpu

} // namespace lahva
