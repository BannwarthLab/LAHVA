#include "linalg.hpp"
#include "const.h"
#include "tcgmblas.h"
#include "tcgmblas.hpp"
#include "utils.hpp"
#include <numeric>
#include <iostream>

#define M 10
#define N 5 

using namespace tcgmtensor::cpu;
using namespace tcgmtensor;

const double thr2 = 5.0e-15;
const float thr = 5.0e-7;
float vf[9] = {1.0, 4.0, 5.0, 0.0, 2.0, 6.0, 0.0, 0.0, 3.0};
double vd[9] = {1.0, 4.0, 5.0, 0.0, 2.0, 6.0, 0.0, 0.0, 3.0};
float *pf = vf;
double *pd = vd;
float vftri[6] = {1.0, 4.0, 5.0, 2.0, 6.0, 3.0};
double vdtri[6] = {1.0, 4.0, 5.0, 2.0, 6.0, 3.0};
float *pft = vftri;
double *pdt = vdtri;
Vector<float> pvf({1.0, 2.0, 3.0}) ;

int test_dgemv_zero_v_cpp(){
    int stat_ = 0;
    Shape s(M,N);
    Matrix<double> A(s, 1.0);
    Vector<double> x(N, 0.0);
    Vector<double> y(M, 1.0);

    MatrixVectorProduct("N", 1.0, A, x, 1, 0.0, y, 1);

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1; 

    y = Vector<double>(M, 1.0);
    
    MatrixVectorProduct("N", 1.0, A, x, 1, 1.0, y, 1);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    MatrixVectorProduct(A, x, y, "N", 1.0d, 1.0d);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    A = Matrix<double>(Shape(N,M), 1.0);

    MatrixVectorProduct(A, x, y, "T"); //Check transpose

    sum_ = sum(M, y.data());

    if (!check(sum_, 0.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    return stat_;
}

int test_sgemv_zero_v_cpp(){
    int stat_ = 0;
    Shape s(M,N);
    Matrix<float> A(s, 1.0);
    Vector<float> x(N, 0.0);
    Vector<float> y(M, 1.0);

    MatrixVectorProduct("N", 1.0, A, x, 1, 0.0, y, 1);

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1; 

    y = Vector<float>(M, 1.0);
    
    MatrixVectorProduct("N", 1.0, A, x, 1, 1.0, y, 1);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    MatrixVectorProduct(A, x, y, "N", 1.0d, 1.0d);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    A = Matrix<float>(Shape(N,M), 1.0);

    MatrixVectorProduct(A, x, y, "T"); //Check transpose

    sum_ = sum(M, y.data());

    if (!check(sum_, 0.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    return stat_;
}

int test_dgemv_v_cpp(){
    int stat_ = 0;
    Shape s(3,3);
    Matrix<double> A(s, pd, false);
    Vector<double> x({1.0, 2.0, 3.0});
    Vector<double> y(3, 0.0);

    MatrixVectorProduct("N", 1.0, A, x, 1, 0.0, y, 1);

    Vector<double> vres({1.0, 8.0, 26.0}); 

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1; 
    
    y = Vector<double>(3, 1.0);
    
    MatrixVectorProduct("N", 1.0, A, x, 1, 1.0, y, 1);

    vres = Vector<double>({2.0, 9.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    MatrixVectorProduct(A, x, y, "N", 2.0d, 0.0d);

    vres = Vector<double>({2.0, 16.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

int test_sgemv_v_cpp(){
    int stat_ = 0;
    Shape s(3,3);
    Matrix<float> A(s, pf, false);
    Vector<float> x({1.0, 2.0, 3.0});
    Vector<float> y(3, 0.0);

    MatrixVectorProduct("N", 1.0, A, x, 1, 0.0, y, 1);

    Vector<float> vres({1.0, 8.0, 26.0}); 

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1; 
    
    y = Vector<float>(3, 1.0);
    
    MatrixVectorProduct("N", 1.0, A, x, 1, 1.0, y, 1);

    vres = Vector<float>({2.0, 9.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    MatrixVectorProduct(A, x, y, "N", 2.0d, 0.0d);

    vres = Vector<float>({2.0, 16.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

int test_dsymv_zero_v_cpp(){
    int stat_ = 0;
    Shape s(M,M);
    Matrix<double> A(s, 1.0);
    Vector<double> x(M, 0.0);
    Vector<double> y(M, 1.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 0.0, y, 1);

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1; 

    y = Vector<double>(M, 1.0);
    
    SymMatrixVectorProduct(1.0, A, x, 1, 1.0, y, 1);
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(A, x, y, 1.0d, 1.0d);
    
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    return stat_;
}

int test_ssymv_zero_v_cpp(){
    int stat_ = 0;
    Shape s(M,M);
    Matrix<float> A(s, 1.0);
    Vector<float> x(M, 0.0);
    Vector<float> y(M, 1.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 0.0, y, 1);

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1; 

    y = Vector<float>(M, 1.0);
    
    SymMatrixVectorProduct(1.0, A, x, 1, 1.0, y, 1);
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(A, x, y, 1.0d, 1.0d);  
    
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    return stat_;
}

int test_dsymv_v_cpp(){
    int stat_ = 0;
    Shape s(3,3);
    Matrix<double> A(s, pd, false);
    Vector<double> x({1.0, 2.0, 3.0});
    Vector<double> y(3, 0.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 0.0, y, 1);

    Vector<double> vres({24.0, 26.0, 26.0}); 

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1; 
    
    
    y = Vector<double>(3, 1.0);
    
    SymMatrixVectorProduct(1.0, A, x, 1, 1.0, y, 1);

    vres = Vector<double>({25.0, 27.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(A, x, y, 2.0d, 0.0d);

    vres = Vector<double>({48.0, 52.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

int test_ssymv_v_cpp(){
    int stat_ = 0;
    Shape s(3,3);
    Matrix<float> A(s, pf, false);
    Vector<float> x({1.0, 2.0, 3.0});
    Vector<float> y(3, 0.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 0.0, y, 1);

    Vector<float> vres({24.0, 26.0, 26.0}); 

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1; 

    
    y = Vector<float>(3, 1.0);
    
    SymMatrixVectorProduct(1.0, A, x, 1, 1.0, y, 1);

    vres = Vector<float>({25.0, 27.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(A, x, y, 2.0d, 0.0d);

    vres = Vector<float>({48.0, 52.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

int test_dsymv_zero_lowtri_cpp(){
    int stat_ = 0;
    Shape s(M,M);
    LowTriMatrix<double> A(s, 1.0);
    Vector<double> x(M, 0.0);
    Vector<double> y(M, 1.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 0.0, y, 1);

    double sum_ = sum(M, y.data());

    if (!check(sum_, 0.0d, thr, "Error when using Matrix Multiplication with a zero vector. LowTri1")) stat_ += 1; 

    y = Vector<double>(M, 1.0);
    
    SymMatrixVectorProduct(1.0, A, x, 1, 1.0, y, 1);
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector. LowTri2")) stat_ += 1;

    SymMatrixVectorProduct(A, x, y, 1.0d, 1.0d);
    
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector. LowTri3")) stat_ += 1;

    return stat_;
}

int test_ssymv_zero_lowtri_cpp(){
    int stat_ = 0;
    Shape s(M,M);
    LowTriMatrix<float> A(s, 1.0);
    Vector<float> x(M, 0.0);
    Vector<float> y(M, 1.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 0.0, y, 1);

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0d, thr, "Error when using Matrix Multiplication with a zero vector. LowTri1")) stat_ += 1; 

    y = Vector<float>(M, 1.0);
    
    SymMatrixVectorProduct(1.0, A, x, 1, 1.0, y, 1);
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector. LowTri2")) stat_ += 1;

    SymMatrixVectorProduct(A, x, y, 1.0d, 1.0d);  
    
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector. LowTRi3")) stat_ += 1;

    return stat_;
}

int test_dsymv_lowtri_cpp(){
    int stat_ = 0;
    double* vdtri_ = new double[6] {1.0, 4.0, 5.0, 2.0, 6.0, 3.0};
    LowTriMatrix<double> A(3, vdtri_);
    Vector<double> x({1.0, 2.0, 3.0});
    Vector<double> y(3, 0.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 0.0, y, 1);

    Vector<double> vres({24.0, 26.0, 26.0}); 
    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1; 
    y = Vector<double>(3, 1.0);
    
    SymMatrixVectorProduct(1.0, A, x, 1, 1.0, y, 1);

    vres = Vector<double>({25.0, 27.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(A, x, y, 2.0d, 0.0d);

    vres = Vector<double>({48.0, 52.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

int test_ssymv_lowtri_cpp(){
    int stat_ = 0;
    float* vdtri_ = new float[6] {1.0, 4.0, 5.0, 2.0, 6.0, 3.0};
    LowTriMatrix<float> A(3, vdtri_, true);
    Vector<float> x({1.0, 2.0, 3.0});
    Vector<float> y(3, 0.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 0.0, y, 1);

    Vector<float> vres({24.0, 26.0, 26.0}); 

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1; 
    
    y = Vector<float>(3, 1.0);
    
    SymMatrixVectorProduct(1.0, A, x, 1, 1.0, y, 1);

    vres = Vector<float>({25.0, 27.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(A, x, y, 2.0d, 0.0d);

    vres = Vector<float>({48.0, 52.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

int test_dtpmv_cpp(){
    int stat_ = 0;
    double* vdtri_ = new double[6] {1.0, 4.0, 5.0, 2.0, 6.0, 3.0};
    LowTriMatrix<double> A(3, vdtri_);
    Vector<double> x({1.0, 2.0, 3.0});

    LowTriMatrixVectorProduct("N", CblasNonUnit, A, x, 1);

    Vector<double> vres({1.0, 8.0, 26.0}); 

    if (!check(x.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    x = Vector<double>({1.0, 2.0, 3.0});
    LowTriMatrixVectorProduct("N", CblasNonUnit, A, x, 1);

    if (!check(x.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    x = Vector<double>({1.0, 2.0, 3.0});
    LowTriMatrixVectorProduct("T", CblasNonUnit, A, x, 1);

    vres = Vector<double>({24.0, 22.0, 9.0}); 

    if (!check(x.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

int test_stpmv_cpp(){
    int stat_ = 0;
    float* vdtri_ = new float[6] {1.0, 4.0, 5.0, 2.0, 6.0, 3.0};
    LowTriMatrix<float> A(3, vdtri_);
    Vector<float> x({1.0, 2.0, 3.0});

    LowTriMatrixVectorProduct("N", CblasNonUnit, A, x, 1);

    Vector<float> vres({1.0, 8.0, 26.0}); 

    if (!check(x.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    x = Vector<float>({1.0, 2.0, 3.0});
    LowTriMatrixVectorProduct("N", CblasNonUnit, A, x, 1);

    if (!check(x.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    x = Vector<float>({1.0, 2.0, 3.0});
    LowTriMatrixVectorProduct("T", CblasNonUnit, A, x, 1);

    vres = Vector<float>({24.0, 22.0, 9.0}); 

    if (!check(x.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}



int test_dgemv_zero_v_c(){
    int stat_ = 0;
    Shape s(M,N);
    Matrix<double> A(s, 1.0);
    Vector<double> x(N, 0.0);
    Vector<double> y(M, 1.0);

    MatrixVectorProduct("N", M, N, 1.0, A.data(), x.data(), 1, 0.0, y.data(), 1);

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1; 

    y = Vector<double>(M, 1.0);
    
    MatrixVectorProduct("N", M, N, 1.0, A.data(), x.data(), 1, 1.0, y.data(), 1);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    MatrixVectorProduct(M, N, A.data(), x.data(), y.data(), "N", 1.0d, 1.0d);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    A = Matrix<double>(Shape(N,M), 1.0);

    MatrixVectorProduct(N, M, A.data(), x.data(), y.data(), "T"); //Check transpose

    sum_ = sum(M, y.data());

    if (!check(sum_, 0.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    return stat_;
}

int test_sgemv_zero_v_c(){
    int stat_ = 0;
    Shape s(M,N);
    Matrix<float> A(s, 1.0);
    Vector<float> x(N, 0.0);
    Vector<float> y(M, 1.0);

    MatrixVectorProduct("N", M, N, 1.0, A.data(), x.data(), 1, 0.0, y.data(), 1);

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1; 

    y = Vector<float>(M, 1.0);
    
    MatrixVectorProduct("N", M, N, 1.0, A.data(), x.data(), 1, 1.0, y.data(), 1);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    MatrixVectorProduct(M, N, A.data(), x.data(), y.data(), "N", 1.0d, 1.0d);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    A = Matrix<float>(Shape(N,M), 1.0);

    MatrixVectorProduct(N, M, A.data(), x.data(), y.data(), "T"); //Check transpose

    sum_ = sum(M, y.data());

    if (!check(sum_, 0.0d, thr, "Error when using Matrix Multiplication with a zero vector. Transpose")) stat_ += 1;

    return stat_;
}

int test_dgemv_v_c(){
    int stat_ = 0;
    Shape s(3,3);
    Matrix<double> A(s, pd, false);
    Vector<double> x({1.0, 2.0, 3.0});
    Vector<double> y(3, 0.0);

    MatrixVectorProduct("N", 3, 3, 1.0, A.data(), x.data(), 1, 0.0, y.data(), 1);

    Vector<double> vres({1.0, 8.0, 26.0}); 

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1; 
    return stat_;
    
    y = Vector<double>(M, 1.0);
    
    MatrixVectorProduct("N", 3, 3, 1.0, A.data(), x.data(), 1, 1.0, y.data(), 1);

    vres = Vector<double>({2.0, 9.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    MatrixVectorProduct(3, 3, A.data(), x.data(), y.data(), "N", 2.0d, 0.0d);

    vres = Vector<double>({2.0, 16.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

int test_sgemv_v_c(){
    int stat_ = 0;
    Shape s(3,3);
    Matrix<float> A(s, pf, false);
    Vector<float> x({1.0, 2.0, 3.0});
    Vector<float> y(3, 0.0);

    MatrixVectorProduct("N", 3, 3, 1.0, A.data(), x.data(), 1, 0.0, y.data(), 1);

    Vector<float> vres({1.0, 8.0, 26.0}); 

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1; 
    return stat_;
    
    y = Vector<float>(M, 1.0);
    
    MatrixVectorProduct("N", 3, 3, 1.0, A.data(), x.data(), 1, 1.0, y.data(), 1);

    vres = Vector<float>({2.0, 9.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    MatrixVectorProduct(3, 3, A.data(), x.data(), y.data(), "N", 2.0d, 0.0d);

    vres = Vector<float>({2.0, 16.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

int test_dsymv_zero_v_c(){
    int stat_ = 0;
    Shape s(M,M);
    Matrix<double> A(s, 1.0);
    Vector<double> x(M, 0.0);
    Vector<double> y(M, 1.0);

    SymMatrixVectorProduct(M, 1.0, A.data(), x.data(), 1, 0.0, y.data(), 1);

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1; 

    y = Vector<double>(M, 1.0);
    
    SymMatrixVectorProduct(M, 1.0, A.data(), x.data(), 1, 1.0, y.data(), 1);
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(M, A.data(), x.data(), y.data(), 1.0d, 1.0d);
    
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    return stat_;
}

int test_ssymv_zero_v_c(){
    int stat_ = 0;
    Shape s(M,M);
    Matrix<float> A(s, 1.0);
    Vector<float> x(M, 0.0);
    Vector<float> y(M, 1.0);

    SymMatrixVectorProduct(M, (float) 1.0, A.data(), x.data(), 1, (float) 0.0, y.data(), 1);

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1; 

    y = Vector<float>(M, 1.0);
    
    SymMatrixVectorProduct(M, 1.0, A.data(), x.data(), 1, 1.0, y.data(), 1);
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(M, A.data(), x.data(), y.data(), 1.0d, 1.0d);  
    
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0d, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    return stat_;
}

int test_dsymv_v_c(){
    int stat_ = 0;
    Shape s(3,3);
    Matrix<double> A(s, pd, false);
    Vector<double> x({1.0, 2.0, 3.0});
    Vector<double> y(3, 0.0);

    SymMatrixVectorProduct(3, 1.0, A.data(), x.data(), 1, 0.0, y.data(), 1);

    Vector<double> vres({24.0, 26.0, 26.0}); 

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1; 
    return stat_;
    
    y = Vector<double>(M, 1.0);
    
    SymMatrixVectorProduct(3, 1.0, A.data(), x.data(), 1, 1.0, y.data(), 1);

    vres = Vector<double>({25.0, 27.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(3, A.data(), x.data(), y.data(), 2.0d, 0.0d);

    vres = Vector<double>({48.0, 52.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

int test_ssymv_v_c(){
    int stat_ = 0;
    Shape s(3,3);
    Matrix<float> A(s, pf, false);
    Vector<float> x({1.0, 2.0, 3.0});
    Vector<float> y(3, 0.0);

    SymMatrixVectorProduct(3, 1.0, A.data(), x.data(), 1, 0.0, y.data(), 1);

    Vector<float> vres({24.0, 26.0, 26.0}); 

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1; 
    return stat_;
    
    y = Vector<float>(M, 1.0);
    
    SymMatrixVectorProduct(3, 1.0, A.data(), x.data(), 1, 1.0, y.data(), 1);

    vres = Vector<float>({25.0, 27.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(3, A.data(), x.data(), y.data(), 2.0d, 0.0d);

    vres = Vector<float>({48.0, 52.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

int main(){
    int stat = 0;
    stat += test_dgemv_zero_v_cpp();
    stat += test_sgemv_zero_v_cpp();
    stat += test_dsymv_zero_v_cpp();
    stat += test_ssymv_zero_v_cpp();
    stat += test_dsymv_zero_lowtri_cpp();
    stat += test_ssymv_zero_lowtri_cpp();
    stat += test_sgemv_v_cpp();
    stat += test_dgemv_v_cpp();
    stat += test_ssymv_v_cpp();
    stat += test_dsymv_v_cpp();
    stat += test_ssymv_lowtri_cpp();
    stat += test_dsymv_lowtri_cpp();
    stat += test_dtpmv_cpp();
    stat += test_stpmv_cpp();
    stat += test_dgemv_zero_v_c();
    stat += test_sgemv_zero_v_c();
    stat += test_dsymv_zero_v_c();
    stat += test_ssymv_zero_v_c();
    stat += test_sgemv_v_c();
    stat += test_dgemv_v_c();
    stat += test_ssymv_v_c();
    stat += test_dsymv_v_c();
    return stat;
};
