#pragma once
#include "linalg.hpp"
#include "const.h"
#include <tuple>

#include <assert.h>

namespace tcgmtensor{
    template<typename T>
    void check_equal_size(const Vector<T>& v1, const Vector<T>& v2){
        assert(v1.size() == v2.size());
    };

    template<typename T>
    void check_equal_size(const Tensor<T>& v1, const Tensor<T>& v2){
        assert(v1.size() == v2.size());
    };

    template<typename T>
    std::tuple<BLAS_INT, BLAS_INT> check_size_mv(const Matrix<T>& m, const Vector<T>& vmult, const Vector<T>& vres, CBLAS_TRANSPOSE trans = CblasNoTrans){
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
    std::tuple<BLAS_INT, BLAS_INT> check_size_mv(const LowTriMatrix<T>& m, const Vector<T>& vmult, const Vector<T>& vres, CBLAS_TRANSPOSE trans = CblasNoTrans){
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
    std::tuple<BLAS_INT, BLAS_INT, BLAS_INT> check_size_mm(const Matrix<T>& a, const Matrix<T>& b, 
        const Matrix<T>& c, CBLAS_TRANSPOSE transa = CblasNoTrans, CBLAS_TRANSPOSE transb = CblasNoTrans){
        
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
    std::tuple<BLAS_INT, BLAS_INT> check_size_mv(const LowTriMatrix<T> m, const Vector<T> vmult){
        Shape s = m.shape();
        BLAS_INT nrow = s.first;
        BLAS_INT ncol = s.second;

        assert(ncol == (BLAS_INT)vmult.size());
       
        return std::make_tuple(nrow, ncol);
    };

    BLAS_INT get_leading(BLAS_INT nrow, BLAS_INT ncol);
    CBLAS_TRANSPOSE get_trans(const char* T);

}