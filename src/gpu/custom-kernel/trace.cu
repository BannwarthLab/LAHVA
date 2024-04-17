/* purification.cu */
/* Routines for  purification includin  purification including purifier class */

#include <stdio.h>
#include <cstring>
#include "linalg.hpp"
#include "additional-level1.hpp"
#include "../../gpu-utils/utils.hpp"
//#include "printmat.h" //DEBUG
#include <cuda_runtime.h>
#include "cublas_v2.h"

namespace tcgmtensor
{
  namespace gpu
  {
  

    // GPU kernels to take a matrix trace 
    // expects that trace is zero at beginning (we must memset externally)
    // compute trace of matrix on the GPU
    __global__ static void MatrixTrace(const double* vecin,const unsigned long long ndim,double* trace)
    {
      // get the global id and thread id (in vector)
      unsigned long long tid= threadIdx.x;
      unsigned long long id = blockIdx.x*blockDim.x+threadIdx.x;
      //Each block gets its own copy in shared memory
      __shared__ double temp[THREADS_PER_BLOCK];
      double tmp=0.0;
      while (id < ndim)
      {
          tmp += vecin[id*ndim+id];
          id += blockDim.x * gridDim.x;
      }
      temp[tid]= tmp;
      __syncthreads();
    
    
      // for reductions, threadsPerBlock must be a power of 2
      // because of the following code
      int i = blockDim.x/2;
      while (i != 0)
      {
              if (tid < i)
            
              temp[tid] += temp[tid + i];
            
             __syncthreads();
            
        i /= 2;
      }
      // Thread 0 adds partial sums to overall sum
      if(tid==0) 
        trace[blockIdx.x]=temp[0]; // per block, we have incremented everything to the first element
    }


    // compute trace of matrix on the GPU (starting from single precision)
    __global__ static void MatrixTrace(const float* vecin,const unsigned long long ndim,double* trace)
    {
      // get the global id and thread id (in vector)
      unsigned long long tid= threadIdx.x;
      unsigned long long id = blockIdx.x*blockDim.x+threadIdx.x;
      //Each block gets its own copy in shared memory
      __shared__ double temp[THREADS_PER_BLOCK];
      double tmp=0.0;
      while (id < ndim)
      {
          tmp += (double)vecin[id*ndim+id]; //fast_float2double(vecin[id*ndim+id]);
          id += blockDim.x * gridDim.x;
      }
      temp[tid]= tmp;
      __syncthreads();
    
    
      // for reductions, threadsPerBlock must be a power of 2
      // because of the following code
      int i = blockDim.x/2;
      while (i != 0)
      {
              if (tid < i)
            
              temp[tid] += temp[tid + i];
            
             __syncthreads();
            
        i /= 2;
      }
      // Thread 0 adds partial sums to overall sum
      if(tid==0) 
        trace[blockIdx.x]=temp[0]; // per block, we have incremented everything to the first element
    }

    // compute trace of matrix on the GPU based on separately stored diagonal 
    __global__ static void MatrixTraceFromDiagonal(const double* diag,const unsigned long long ndim,double* trace)
    {
      // get the global id and thread id (in vector)
      unsigned long long tid= threadIdx.x;
      unsigned long long id = blockIdx.x*blockDim.x+threadIdx.x;
      //Each block gets its own copy in shared memory
      __shared__ double temp[THREADS_PER_BLOCK];
      double tmp=0.0;
      while (id < ndim)
      {
          tmp += diag[id];
          id += blockDim.x * gridDim.x;
      }
      temp[tid]= tmp;
      __syncthreads();
    
    
      // for reductions, threadsPerBlock must be a power of 2
      // because of the following code
      int i = blockDim.x/2;
      while (i != 0)
      {
              if (tid < i)
            
              temp[tid] += temp[tid + i];
            
             __syncthreads();
            
        i /= 2;
      }
      // Thread 0 adds partial sums to overall sum
      if(tid==0) 
        trace[blockIdx.x]=temp[0]; // per block, we have incremented everything to the first element
    }

    
    // computes scale*0.5*(d*A+A*d) 
    __global__ static void SymmetrizedDiagonalMatrixMatrixProductKernel_Mixed(unsigned long long ndim,const double scale,const double* diag,
                                                                              const float* matrixIn,float *matrixOut)
    {
      // unsigned long long id = blockDim.x < ndim ? (blockIdx.x*blockDim.x)%ndim+threadIdx.x : threadIdx.x  ;
      // unsigned long long jd = blockDim.x < ndim ? (blockIdx.x*blockDim.x)/ndim : blockIdx.x ;
      // A.1 B.1 C.1
      // A.2 B.2 C.2
      // A.3 B.3 C.3
      unsigned long long id = blockIdx.y; 
      unsigned long long jd = threadIdx.x+blockIdx.x*blockDim.x;
      // 0.5*(matrix * diag + matrix^T * diag)
      if (id < ndim && jd < ndim)
      {
        matrixOut[id*ndim+jd]+=((float) (0.5*scale*(diag[id]+diag[jd]))*matrixIn[id*ndim+jd]);
      }
    }

    // symmetrize matrix 
    __global__ static void SymmetrizeMatrix(unsigned long long ndim,float *matrix)
    {
      unsigned long long id = blockIdx.y; 
      unsigned long long jd = threadIdx.x+blockIdx.x*blockDim.x;
      // 0.5*(matrix * diag + matrix^T * diag)
      if (id < ndim && jd < ndim)
      {
        matrix[id*ndim+jd]=0.5*(matrix[id*ndim+jd]+matrix[jd*ndim+id]);
      }
    }

    // symmetrize matrix 
    __global__ static void SymmetrizeMatrix(unsigned long long ndim,double *matrix)
    {
      unsigned long long id = blockIdx.y; 
      unsigned long long jd = threadIdx.x+blockIdx.x*blockDim.x;
      // 0.5*(matrix * diag + matrix^T * diag)
      if (id < ndim && jd < ndim)
      {
        matrix[id*ndim+jd]=0.5*(matrix[id*ndim+jd]+matrix[jd*ndim+id]);
      }
    }
 
    // compute trace on GPU wrapper routine (FP64)
    template<>
    double ComputeTrace<double>(const CudaRuntime& cudart, const Matrix<double>& m, bool use_diag)
    {
      // Number of blocks in grid;
      int gridS = cudart.gridSize(m.size(), 1);
      Vector<double> v(gridS);
      v.copy2device(cudart);
      if (use_diag)
      {
        Vector<double> diag = m.get_diagonal();
        diag.copy2device(cudart);
        MatrixTraceFromDiagonal<<<gridS,cudart.blockSize()>>>(diag.gpu_data(),diag.size(),v.gpu_data());
        
      }
      else 
      {
        check_device_alloc(cudart, m);
        MatrixTrace<<<gridS,cudart.blockSize()>>>(m.gpu_data(),m.shape().first ,v.gpu_data()); // compute trace
      }
      
      v.copy2host(cudart);
      double trace=0.0;
      // final summation on CPU
      for(int i = 0 ; i < gridS ; i++) trace+=v[i];
      return trace;
    }

    double ComputeTrace(const CudaRuntime& cudart, const Vector<double>& diag)
    {
      // Number of blocks in grid;
      int gridS = cudart.gridSize(diag.size(), 1);
      Vector<double> v(gridS);
      v.copy2device(cudart);
      check_device_alloc(cudart, diag);
      MatrixTraceFromDiagonal<<<gridS,cudart.blockSize()>>>(diag.gpu_data(),diag.size() ,v.gpu_data());
      
      v.copy2host(cudart);
      double trace=0.0;
      // final summation on CPU
      for(int i = 0 ; i < gridS ; i++) trace+=v[i];
      return trace;
    }

    
    // compute trace on GPU wrapper routine (FP32)
    template<>
    double ComputeTrace<float>(const CudaRuntime& cudart, const Matrix<float>& m, bool use_diag)
    {
      // Number of blocks in grid;
      int gridS =  cudart.gridSize(m.size(), 1);
      Vector<double> v(gridS);
      v.copy2device(cudart);
      check_device_alloc(cudart, m);
      MatrixTrace<<<gridS,cudart.blockSize()>>>(m.gpu_data(),m.shape().first ,v.gpu_data()); // compute trace
      v.copy2host(cudart);
      double trace=0.0;
      // final summation on CPU
      for(int i = 0 ; i < gridS ; i++) trace+=v[i];
      return trace;
    }

    template<>
    void SymmetrizeMatrix<float>(const CudaRuntime& cudart, Matrix<float>& m)
    {
      // Number of blocks in grid;
      int gridS =  cudart.gridSize(m.size(), 1);
      check_device_alloc(cudart, m);
      SymmetrizeMatrix<<<gridS,cudart.blockSize()>>>(m.shape().first,m.gpu_data()); // compute trace
    }

    template<>
    void SymmetrizeMatrix<double>(const CudaRuntime& cudart, Matrix<double>& m)
    {
      // Number of blocks in grid;
      int gridS =  cudart.gridSize(m.size(), 1);
      check_device_alloc(cudart, m);
      SymmetrizeMatrix<<<gridS,cudart.blockSize()>>>(m.shape().first,m.gpu_data()); // compute trace
    }

  } // namespace Purification_kernel
} // namespace Purification