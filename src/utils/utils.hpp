#pragma once
#include "linalg.hpp"
#include "const.h"
#include <tuple>

#include <assert.h>

namespace lahva{
    template<typename T, typename U>
    void check_equal_size(const Tensor<T>& v1, const Tensor<U>& v2){
        assert(v1.size() == v2.size());
    };
    namespace cpu
    {
    
    

    template<typename T>
    std::tuple<BLAS_INT, BLAS_INT> check_size_mv(const Matrix_<T>& m, const Tensor<T>& vmult, const Tensor<T>& vres, CBLAS_TRANSPOSE trans = CblasNoTrans){
        Shape s = m.shape();
        BLAS_INT nrow = s.first;
        BLAS_INT ncol = s.second;
        if (trans == CblasNoTrans) {
            assert(nrow == (BLAS_INT)vres.size());
            assert(ncol == (BLAS_INT)vmult.size());
            
        }
        else {
            assert(nrow == (BLAS_INT)vmult.size());
            assert(ncol == (BLAS_INT)vres.size());
            
        }
        
        return std::make_tuple(nrow, ncol);
    };

    template<typename T>
    std::tuple<BLAS_INT, BLAS_INT> check_size_mv(const LowTriMatrix_<T>& m, const Tensor<T>& vmult, const Tensor<T>& vres, CBLAS_TRANSPOSE trans = CblasNoTrans){
        Shape s = m.shape();
        BLAS_INT nrow = s.first;
        BLAS_INT ncol = s.second;
        if (trans == CblasNoTrans) {
            assert(nrow == (BLAS_INT)vres.size());
            assert(ncol == (BLAS_INT)vmult.size());
            
        }
        else {
            assert(nrow == (BLAS_INT)vmult.size());
            assert(ncol == (BLAS_INT)vres.size());
            
        }
        
        return std::make_tuple(nrow, ncol);
    };

    template<typename T>
    std::tuple<BLAS_INT, BLAS_INT, BLAS_INT> check_size_mm(const Matrix_<T>& a, const Matrix_<T>& b, 
        const Matrix_<T>& c, CBLAS_TRANSPOSE transa = CblasNoTrans, CBLAS_TRANSPOSE transb = CblasNoTrans){
        
        Shape sa = a.shape();
        BLAS_INT nrowa = sa.first;
        BLAS_INT ncola = sa.second;

        Shape sb = b.shape();
        BLAS_INT nrowb = sb.first;
        BLAS_INT ncolb = sb.second;

        Shape sc = c.shape();
        BLAS_INT nrowc = sc.first;
        BLAS_INT ncolc = sc.second;

        BLAS_INT k;

        if (transa == CblasNoTrans) {
            if (transb == CblasNoTrans)
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
           if (transb == CblasNoTrans)
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

    template<typename T>
    std::tuple<BLAS_INT, BLAS_INT> check_same_shape_mm(const Matrix_<T>& a, const Matrix_<T>& b, 
        const Matrix_<T>& c, CBLAS_TRANSPOSE transa = CblasNoTrans, CBLAS_TRANSPOSE transb = CblasNoTrans){
        
        Shape sa = a.shape();
        BLAS_INT nrowa = sa.first;
        BLAS_INT ncola = sa.second;

        Shape sb = b.shape();
        BLAS_INT nrowb = sb.first;
        BLAS_INT ncolb = sb.second;

        Shape sc = c.shape();
        BLAS_INT nrowc = sc.first;
        BLAS_INT ncolc = sc.second;

        if (transa == CblasNoTrans) {
            if (transb == CblasNoTrans)
            {
                assert(ncola == ncolb);
                assert(nrowa == nrowb);
                assert(ncolb == ncolc);
                assert(nrowb == nrowc);
            }
            else // B is transposed
            {
                assert(ncola == nrowb);
                assert(nrowa == ncolb);
                assert(ncola == ncolc);
                assert(nrowa == nrowc);
            }   
        }
        else // A is transposed 
        {
           if (transb == CblasNoTrans)
            {
                assert(nrowa == ncolb);
                assert(ncola == nrowb);
                assert(ncolb == ncolc);
                assert(nrowb == nrowc);
            }
            else // A and B are transposed
            {
                assert(nrowa == nrowb);
                assert(ncola == ncolb);
                assert(ncolb == nrowc);
                assert(nrowa == ncolc);
            }    
        }
        

        return std::make_tuple(nrowc, ncolc);
    };

    template<typename T>
    std::tuple<BLAS_INT, BLAS_INT> check_size_mv(const LowTriMatrix_<T>& m, const Tensor<T>& vmult)
    {
        Shape s = m.shape();
        BLAS_INT nrow = s.first;
        BLAS_INT ncol = s.second;

        assert(ncol == (BLAS_INT)vmult.size());
       
        return std::make_tuple(nrow, ncol);
    };

    BLAS_INT get_leading(BLAS_INT nrow, BLAS_INT ncol, CBLAS_TRANSPOSE trans = CblasNoTrans);
    CBLAS_TRANSPOSE get_trans(const char* T);
    } // namespace cpu
}