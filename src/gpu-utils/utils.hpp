#ifndef TCGMBLAS_GPU_UTILS_HPP
#define TCGMBLAS_GPU_UTILS_HPP 
#include "linalg.hpp"
#include "runtime.hpp"

namespace tcgmtensor {
    namespace gpu
    {
        template<typename T>
        void check_device_alloc(const CudaRuntime& cudart, const Vector<T>& gpu_vec){
            if (!gpu_vec.alloc_on_device())
            {
                gpu_vec.copy2device(cudart);
            }
        };

        template<typename T>
        void check_device_alloc(const CudaRuntime& cudart, const Matrix<T>& gpu_vec){
            if (!gpu_vec.alloc_on_device())
            {
                gpu_vec.copy2device(cudart);
            }
        };
        template<typename T>
        void check_device_alloc(const CudaRuntime& cudart, const GPUTensor<T>& gpu_vec){
            if (!gpu_vec.alloc_on_device())
            {
                gpu_vec.copy2device(cudart);
            }
        };

        template<typename T>
        std::tuple<size_t, size_t> check_size_mv(const Matrix<T>& m, const Vector<T>& vmult, const Vector<T>& vres, cublasOperation_t trans = CUBLAS_OP_N){
        Shape s = m.shape();
        size_t nrow = s.first;
        size_t ncol = s.second;
        if (trans == CUBLAS_OP_N) {
            assert(nrow == vres.size());
            assert(ncol == vmult.size());
        }
        else {
            assert(nrow == vmult.size());
            assert(ncol == vres.size()); 
        }
        
        return std::make_tuple(nrow, ncol);
        };

        template<typename T>
        std::tuple<size_t, size_t> check_size_mv(const LowTriMatrix<T>& m, const Vector<T>& vmult, const Vector<T>& vres, cublasOperation_t trans = CUBLAS_OP_N){
        Shape s = m.shape();
        size_t nrow = s.first;
        size_t ncol = s.second;
        if (trans == CUBLAS_OP_N) {
            assert(nrow == vres.size());
            assert(ncol == vmult.size());
        }
        else {
            assert(nrow == vmult.size());
            assert(ncol == vres.size()); 
        }
        
        return std::make_tuple(nrow, ncol);
        };

        size_t get_leading(size_t nrow, size_t ncol);

        cublasOperation_t get_trans(const char* T);

        template<typename T>
        std::tuple<size_t, size_t, size_t>  check_size_mm(const Matrix<T>& a, const Matrix<T>& b, 
        const Matrix<T>& c, cublasOperation_t transa = CUBLAS_OP_N, cublasOperation_t transb = CUBLAS_OP_N){
        
        Shape sa = a.shape();
        size_t nrowa = sa.first;
        size_t ncola = sa.second;

        Shape sb = b.shape();
        size_t nrowb = sb.first;
        size_t ncolb = sb.second;

        Shape sc = c.shape();
        size_t nrowc = sc.first;
        size_t ncolc = sc.second;

        size_t k;

        if (transa == CUBLAS_OP_N) {
            if (transb == CUBLAS_OP_N)
            {
                assert(ncola == nrowb);
                assert(nrowa == nrowc);
                assert(ncolb == ncolc);
                k = ncola;
            }
            else // B is transposed
            {
                assert(ncola == ncolb);
                assert(nrowa == nrowc);
                assert(nrowb == ncolc);
                k = ncola;
            }   
        }
        else // A is transposed 
        {
           if (transb == CUBLAS_OP_N)
            {
                assert(nrowa == nrowb);
                assert(ncola == nrowc);
                assert(ncolb == ncolc);
                k = nrowa;
            }
            else // A and B are transposed
            {
                assert(nrowa == ncolb);
                assert(ncola == nrowc);
                assert(nrowb == ncolc);
                k = ncola;
            }    
        }
        

        return std::make_tuple(nrowc, ncolc, k);
    };

    } // namespace gpu
    
}
#endif