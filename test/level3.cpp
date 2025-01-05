#include "common.h"
using namespace lahva::cpu;
#define M 10
#define N 5 
#define K 3

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

int test_dgemm_zero_v_cpp(){
    int stat_ = 0;
    Shape sres(M,N);
    Shape sa(M,K);
    Shape sb(K,N);
    Matrix<double> A(sa, 1.0);
    Matrix<double> B(sb, 0.0);
    Matrix<double> C(sres, 1.0);

    MatrixMatrixProduct("N", "N", 1.0, A, B, 0.0, C);

    Matrix<double> Mres(sres, 0.0);

    if (!check(C.data(), Mres.data(), thr2, M*N,"Error when using Matrix Multiplication with a zero vector.")) stat_ += 1; 
    return stat_;

    return stat_;
}

int test_sgemm_zero_v_cpp(){
    int stat_ = 0;
    Shape sres(M,N);
    Shape sa(M,K);
    Shape sb(K,N);
    Matrix<float> A(sa, 1.0);
    Matrix<float> B(sb, 0.0);
    Matrix<float> C(sres, 1.0);

    MatrixMatrixProduct("N", "N", 1.0, A, B, 0.0, C);

    Matrix<float> Mres(sres, 0.0);

    if (!check(C.data(), Mres.data(), thr2, M*N,"Error when using Matrix Multiplication with a zero vector.")) stat_ += 1; 

    return stat_;
}



int main(){
    int stat = 0;
    stat += test_dgemm_zero_v_cpp();
    //stat += test_sgemm_zero_v_cpp();
    
    return stat;
};
