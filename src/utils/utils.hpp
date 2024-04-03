#pragma once
#include "linalg.hpp"
#include "tcgmblas.hpp"
#include "const.h"
#include <tuple>

#include <assert.h>

namespace tcgmtensor{
    template<typename T>
    void check_equal_size(const vector<T> v1, const vector<T> v2){
        assert(v1.size() == v2.size());
    };

    template<typename T>
    std::tuple<BLAS_INT, BLAS_INT> check_size_mv(const Matrix<T> m, const vector<T> vmult, const vector<T> vres, CBLAS_TRANSPOSE trans = CblasNoTrans){
        Shape s = m.shape();
        BLAS_INT nrow = s.first;
        BLAS_INT ncol = s.second;
        if (trans == CblasNoTrans) {
            assert(ncol == vres.size());
            assert(nrow == vmult.size());
        }
        else {
            assert(ncol == vmult.size());
            assert(nrow == vres.size()); 
        }
        
        return std::make_tuple(nrow, ncol);
    };

    template<typename T>
    std::tuple<BLAS_INT, BLAS_INT, BLAS_INT> check_size_mm(const Matrix<T> a, const Matrix<T> b, 
        const Matrix<T> c, CBLAS_TRANSPOSE transa = CblasNoTrans, CBLAS_TRANSPOSE transb = CblasNoTrans){
        
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
    std::tuple<BLAS_INT, BLAS_INT> check_size_mv(const LowTriMatrix<T> m, const vector<T> vmult){
        Shape s = m.shape();
        BLAS_INT nrow = s.first;
        BLAS_INT ncol = s.second;

        assert(ncol == vmult.size());
       
        return std::make_tuple(nrow, ncol);
    };

    BLAS_INT get_leading(BLAS_INT nrow, BLAS_INT ncol);
    CBLAS_TRANSPOSE get_trans(const char* T);

}