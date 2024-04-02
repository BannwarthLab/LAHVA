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
        if (trans) {
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