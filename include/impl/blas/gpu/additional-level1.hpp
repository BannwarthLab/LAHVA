#ifndef LAHVA_ADD_LEVEL1_HPP
#define LAHVA_ADD_LEVEL1_HPP
#include "linalg.hpp"
#include "runtime.hpp"
#include "../../../../src/gpu-utils/utils.hpp"
namespace lahva{
    namespace gpu
    {
        //Currently only floar 
        template<typename T>
        void Hadamard(const CudaRuntime& cudart, const GPUTensor_<T>& vecin, GPUTensor_<T>& vecinout);
        template<typename T>
        void Hadamard(const CudaRuntime& cudart, const GPUTensor_<T>& vecin, const GPUTensor_<T>& vecin2, GPUTensor_<T>& vecout);
        template<typename T>
        void Hadamard(const CudaRuntime& cudart, const Matrix_<T>& vecin, const GPUTensor_<T>& vecin2, Matrix_<T>& vecout);
        template<typename T>
        void TraceKernelDiag(const CudaRuntime &cudart, unsigned long long ndim, const T* diag, T* vec);
        template<typename U, typename V>
        double ComputeTrace(const CudaRuntime& cudart, const GPUTensor<double, U, V>& diag)
        {
            int gridS = cudart.gridSize(diag.size(), 1);
            Vector<double, U, V> v(gridS);
            v.copy2device(cudart); 
            TraceKernelDiag(cudart, diag.size(), diag.gpu_data(), v.gpu_data());
            v.copy2host(cudart);
            cudart.synchronize();
            return (double)v.sum();
        };
        template<typename U, typename V>
        double ComputeTrace(const CudaRuntime& cudart, const GPUTensor<double, U, V>& diag, Vector<double, U, V>& v)
        {
            v.copy2device(cudart); 
            TraceKernelDiag(cudart, diag.size(), diag.gpu_data(), v.gpu_data());
            v.copy2host(cudart);
            cudart.synchronize();
            return (double)v.sum();
        };

        template<typename T>
        void SymmetrizeMatrix(const CudaRuntime& cudart, Matrix_<T>&);

        template<typename T>
        void FrobeniusKernel(const CudaRuntime& cudart, const unsigned long long ndim, const T* mat, T* vec);
        
        template<typename T, typename U, typename V>
        T FrobeniusNorm(const CudaRuntime& cudart, const GPUTensor<T, U, V>& mat)
        {
            check_device_alloc(cudart, mat);
            Vector<T, U, V> vec(cudart.gridSize(mat.size(), 1), 0.0);
            ScaleVector(cudart, 0.0, vec);
            FrobeniusKernel<T>(cudart, mat.size(), mat.gpu_data(), vec.gpu_data());
            vec.copy2host(cudart);
            cudart.synchronize();
            float norm = vec[0];
            return std::sqrt(norm);
        };

        template<typename T, typename U, typename V>
        T FrobeniusNorm(const CudaRuntime& cudart, const GPUTensor<T, U, V>& mat, Vector<T, U, V>& vec)
        {
            check_device_alloc(cudart, mat);
            ScaleVector(cudart, 0.0, vec);
            FrobeniusKernel(cudart, mat.size(), mat.gpu_data(), vec.gpu_data());
            vec.copy2host(cudart);
            cudart.synchronize();
            float norm = vec.sum();
            return std::sqrt(norm);
        };


        template<typename T>
        void GetDiagonal(const CudaRuntime& cudart, const Matrix_<T>& mat, GPUTensor_<T>& vec);

        template<typename T>
        void SetDiagonal(const CudaRuntime& cudart, const GPUTensor_<T>& vec, Matrix_<T>& m);

        template<typename T>
        void TraceKernel(const CudaRuntime &cudart, unsigned long long ndim, const T* diag, T* vec);

        template<typename T, typename U, typename V>
        double ComputeTrace(const CudaRuntime& cudart, const Matrix<T, U, V>& m, bool use_diag = false)
        {
            // Number of blocks in grid;
            int gridS = cudart.gridSize(m.shape().first, 1);
            Vector<T, U, V> v(gridS);
            v.copy2device(cudart);
            if (use_diag)
            {
                Vector<T, U, V> diag(m.shape().first, cudart);
                GetDiagonal(cudart, m, diag);
                TraceKernelDiag(cudart, diag.size(), diag.gpu_data(), v.gpu_data());
            }
            else
            {
                check_device_alloc(cudart, m);
                TraceKernel(cudart, m.shape().first, m.gpu_data(), v.gpu_data());
                
            }
            v.copy2host(cudart);
            cudart.synchronize();
    
            return (double)v.sum();
        }
     

        template<typename T, typename U, typename V>
        double ComputeTrace(const CudaRuntime& cudart, const Matrix<T, U, V>& m, Vector<T, U, V>& v, bool use_diag = false)
        {
            v.copy2device(cudart);
            if (use_diag)
            {
                Vector<T, U, V> diag(m.shape().first, cudart);
                GetDiagonal(cudart, m, diag);
                TraceKernelDiag(cudart, diag.size(), diag.gpu_data(), v.gpu_data());
            }
            else
            {
                check_device_alloc(cudart, m);
                TraceKernel(cudart, m.shape().first, m.gpu_data(), v.gpu_data());
                
            }
            v.copy2host(cudart);
            cudart.synchronize();
    
            return (double)v.sum();
        }

        template<typename inprec, typename outprec>
        void DecomposeVector2MP(const CudaRuntime& cudart, const GPUTensor_<inprec>& in, GPUTensor_<outprec> &out1, GPUTensor_<outprec> &out2);

        template<typename T, typename U, typename V, typename Tout>
        void DecomposeMatrix(const CudaRuntime& cudart, const GPUTensor<T, U, V>& min, Vector<Tout, U, V>& mout1, Vector<Tout, U, V>& mout2)
        {
            if (min.size() != mout1.size())
                Vector<Tout, U, V> mout1(min.size(), cudart);
            
            if (min.size() != mout2.size())
                Vector<Tout, U, V> mout2(min.size(), cudart);

            DecomposeVector2MP<T, Tout>(cudart, min, mout1, mout2);

        }

    } // namespace gpu
    
}   
#endif