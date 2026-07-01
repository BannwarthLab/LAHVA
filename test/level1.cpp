#include "common.h"

using namespace lahva::cpu;
const double thr2 = 5.0e-15;
const double thr = 5.0e-7;

int test_v_addition_c(){

    int stat_ = 0;

    double v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    double v2[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    double* p = v1;
    double* s = v2;

    AddVectors(5, 1.0, p, s);

    auto sum_ = sum(5, s);

    if (!check(sum_, 3.0*5, thr2, "Error when adding up two Vectors without scaling.")) stat_ += 1; 
    
    AddVectors(5, 1.0, p, 1, s, 1);

    auto sum__ = sum(5, s);

    if (!check(sum__, 5.0*5, thr2, "Error when adding up two Vectors without scaling.")) stat_ += 1;      

    return stat_;
};

int test_v_addition_and_scale_c(){

    int stat_ = 0; 

    double v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    double v2[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    double* p = v1;
    double* s = v2;

    AddVectors(5, 2.0, p, s);

    auto sum_ = sum(5, s);

    if (!check(sum_, 5.0*5, thr2, "Error when adding up two Vectors without scaling.")) stat_ += 1;
    
    AddVectors(5, 1.0, p, 1, s, 1);

    auto sum__ = sum(5, s);

    if (!check(sum__, 7.0*5, thr2, "Error when adding up two Vectors without scaling.")) stat_ += 1;       

    return stat_;
};

int test_vf_addition_c(){

    int stat_ = 0;

    float v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    float v2[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    float* p = v1;
    float* s = v2;

    AddVectors(5, 1.0, p, s);

    auto sum_ = sum(5, s);

    if (!check(sum_, 3.0*5, thr, "Error when adding up two Vectors without scaling.")) stat_ += 1; 
    
    AddVectors(5, 1.0, p, 1, s, 1);

    auto sum__ = sum(5, s);

    if (!check(sum__, 5.0*5, thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;      

    return stat_;
};

int test_vf_addition_and_scale_c(){

    int stat_ = 0; 

    float v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    float v2[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    float* p = v1;
    float* s = v2;

    AddVectors(5, 2.0, p, s);

    auto sum_ = sum(5, s);

    if (!check(sum_, 5.0*5, thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;
    
    AddVectors(5, 1.0, p, 1, s, 1);

    auto sum__ = sum(5, s);

    if (!check(sum__, 7.0*5, thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;       

    return stat_;
};

int test_copy_v_c(){

    int stat_ = 0;

    double v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    double v2[5] ;
    double* p = v1;
    double* s = v2;

    CopyVectors(5, p, s);

    auto sum_ = sum(5, s);
    auto sum_p = sum(5, p);

    if (!check(sum_, sum_p, thr2, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    CopyVectors(5, p, 1, s, 1); 

    auto sum__ = sum(5, s);

    if (!check(sum__, sum_p, thr2, "Error when adding up two Vectors without scaling.")) stat_ += 1; 

    return stat_;
};

int test_copy_vf_c(){

    int stat_ = 0;

    float v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    float v2[5] ;
    float* p = v1;
    float* s = v2;

    CopyVectors(5, p, s);

    auto sum_ = sum(5, s);
    auto sum_p = sum(5, p);

    if (!check(sum_, sum_p, thr2, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    CopyVectors(5, p, 1, s, 1); 

    auto sum__ = sum(5, s);

    if (!check(sum__, sum_p, thr2, "Error when adding up two Vectors without scaling.")) stat_ += 1; 

    return stat_;
};

int test_swap_v_c(){

    int stat_ = 0;

    double v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    double v2[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    double* p = v1;
    double* s = v2;

    SwapVectors(5, p, s);

    auto sum_s = sum(5, s);
    auto sum_p = sum(5, p);

    if (!check(sum_s, 2.0*5, thr2, "Error when swapping two Vectors.")) stat_ += 1;
    if (!check(sum_p, 1.0*5, thr2, "Error when swapping two Vectors.")) stat_ += 1;

    SwapVectors(5, p, 1, s, 1); 

    auto sum_s_ = sum(5, s);
    auto sum_p_ = sum(5, p); 

    if (!check(sum_s_, 1.0*5, thr2, "Error when swapping two Vectors.")) stat_ += 1;
    if (!check(sum_p_, 2.0*5, thr2, "Error when swapping two Vectors.")) stat_ += 1;

    return stat_;
};

int test_swap_vf_c(){

    int stat_ = 0;

    float v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    float v2[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    float* p = v1;
    float* s = v2;

    SwapVectors(5, p, s);

    auto sum_s = sum(5, s);
    auto sum_p = sum(5, p);

    if (!check(sum_s, 2.0*5, thr, "Error when swapping two Vectors.")) stat_ += 1;
    if (!check(sum_p, 1.0*5, thr, "Error when swapping two Vectors.")) stat_ += 1;

    SwapVectors(5, p, 1, s, 1); 

    auto sum_s_ = sum(5, s);
    auto sum_p_ = sum(5, p); 

    if (!check(sum_s_, 1.0*5, thr, "Error when swapping two Vectors.")) stat_ += 1;
    if (!check(sum_p_, 2.0*5, thr, "Error when swapping two Vectors.")) stat_ += 1;

    return stat_;
};

int test_scale_v_c(){

    int stat_ = 0;

    double v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    double* p = v1;

    ScaleVector(5, 2.0, p);

    auto sum_p = sum(5, p);

    if (!check(sum_p, 4.0*5, thr2, "Error when scaling a Vector.")) stat_ += 1;

    ScaleVector(5, 0.5, p); 

    auto sum_p_ = sum(5, p); 

    if (!check(sum_p_, 2.0*5, thr2, "Error when scaling a Vector.")) stat_ += 1;

    return stat_;
};

int test_scale_vf_c(){

    int stat_ = 0;

    float v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    float* p = v1;

    ScaleVector(5, 2.0, p);

    auto sum_p = sum(5, p);

    if (!check(sum_p, 4.0*5, thr, "Error when scaling a Vector.")) stat_ += 1;

    ScaleVector(5, 0.5, p); 

    auto sum_p_ = sum(5, p); 

    if (!check(sum_p_, 2.0*5, thr, "Error when scaling a Vector.")) stat_ += 1;

    return stat_;
};

int test_v_addition_cpp(){

    int stat_ = 0;
    
    Vector<double> p(5, 2.0);
    Vector<double> s(5, 1.0);

    AddVectors(1.0, p, s);

    auto sum = std::accumulate(s.begin(), s.end(), 0.0);

    if (!check(sum, 3.0*p.size(), thr2, "Error when adding up two Vectors without scaling.")) stat_ +=1 ;

    AddVectors(1.0, p, 1, s, 1);

    auto sum_ = std::accumulate(s.begin(), s.end(), 0.0);

    if (!check(sum_, 5.0*p.size(), thr2, "Error when adding up two Vectors without scaling.")) stat_ +=1 ; 

    return stat_;

};

int test_v_addition_and_scale_cpp(){
    int stat_ = 0;
    Vector<double> p(5, 2.0);
    Vector<double> s(5, 1.0);

    AddVectors(2.0, p, s);

    auto sum = std::accumulate(s.begin(), s.end(), 0.0);

    if (!check(sum, 5.0*p.size(), thr2, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    AddVectors(2.0, p, 1, s, 1);

    auto sum_ = std::accumulate(s.begin(), s.end(), 0.0);
    if (!check(sum_, 9.0*p.size(), thr2, "Error when adding up two Vectors without scaling.")) stat_ += 1;
    

    return stat_;

};

int test_inner_v_prod_c(){

    int stat_ = 0;

    double v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    double v2[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    double* p = v1;
    double* s = v2;

    auto prod = InnerVectorProduct(5, p, s);

    if (!check(prod, 2.0*5, thr2, "Error when computing inner product of two Vectors.")) stat_ += 1;

    auto prod_ = InnerVectorProduct(5, p, 1, s, 1); 

    if (!check(prod_, prod, thr2, "Error when computing inner product of two Vectors.")) stat_ += 1;

    return stat_;
};

int test_inner_vf_prod_c(){

    int stat_ = 0;

    float v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    float v2[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    float* p = v1;
    float* s = v2;

    auto prod = InnerVectorProduct(5, p, s);

    if (!check<float>(prod, 2.0*5, thr, "Error when computing inner product of two Vectors.")) stat_ += 1;

    auto prod_ = InnerVectorProduct(5, p, 1, s, 1); 

    if (!check<float>(prod_, prod, thr, "Error when computing inner product of two Vectors.")) stat_ += 1;

    return stat_;
};

int test_vf_addition_cpp(){

    int stat_ = 0;
    
    Vector<float> p(5, 2.0);
    Vector<float> s(5, 1.0);

    AddVectors(1.0, p, s);

    auto sum = std::accumulate(s.begin(), s.end(), 0.0);

    if (!check(sum, 3.0*p.size(), thr, "Error when adding up two Vectors without scaling.")) stat_ +=1 ;

    AddVectors(1.0, p, 1, s, 1);

    auto sum_ = std::accumulate(s.begin(), s.end(), 0.0);

    if (!check(sum_, 5.0*p.size(), thr, "Error when adding up two Vectors without scaling.")) stat_ +=1 ; 

    return stat_;

};

int test_vf_addition_and_scale_cpp(){
    int stat_ = 0;
    Vector<float> p(5, 2.0);
    Vector<float> s(5, 1.0);

    AddVectors(2.0, p, s);

    auto sum = std::accumulate(s.begin(), s.end(), 0.0);

    if (!check(sum, 5.0*p.size(), thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    AddVectors(2.0, p, 1, s, 1);

    auto sum_ = std::accumulate(s.begin(), s.end(), 0.0);
    if (!check(sum_, 9.0*p.size(), thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;
    

    return stat_;

};

int test_copy_v_cpp(){

    int stat_ = 0;
    
    Vector<double> p(5, 2.0);
    Vector<double> s(5);

    CopyVectors(p, s);

    auto sum_s = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s, sum_p, thr2, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    CopyVectors(p, 1, s, 1);

    auto sum_s_ = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p_ = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s_, sum_p_, thr2, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    return stat_;
};

int test_copy_vf_cpp(){

    int stat_ = 0;
    
    Vector<float> p(5, 2.0);
    Vector<float> s(5);

    CopyVectors(p, s);

    auto sum_s = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s, sum_p, thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    CopyVectors(p, 1, s, 1);

    auto sum_s_ = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p_ = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s_, sum_p_, thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    return stat_;
};

int test_swap_v_cpp(){

    int stat_ = 0;
    
    Vector<double> p(5, 2.0);
    Vector<double> s(5, 1.0);

    SwapVectors(p, s);

    auto sum_s = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s, 2.0*5, thr2, "Error when swapping two Vectors.")) stat_ += 1;
    if (!check(sum_p, 1.0*5, thr2, "Error when swapping two Vectors.")) stat_ += 1;
    SwapVectors(p, 1, s, 1);

    auto sum_s_ = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p_ = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s_, 1.0*5, thr2, "Error when swapping up two Vectors.")) stat_ += 1;
    if (!check(sum_p_, 2.0*5, thr2, "Error when swapping up two Vectors.")) stat_ += 1;

    return stat_;
};

int test_swap_vf_cpp(){

    int stat_ = 0;
    
    Vector<float> p(5, 2.0);
    Vector<float> s(5, 1.0);

    SwapVectors(p, s);

    auto sum_s = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s, 2.0*5, thr, "Error when swapping two Vectors.")) stat_ += 1;
    if (!check(sum_p, 1.0*5, thr, "Error when swapping two Vectors.")) stat_ += 1;
    SwapVectors(p, 1, s, 1);

    auto sum_s_ = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p_ = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s_, 1.0*5, thr, "Error when swapping up two Vectors.")) stat_ += 1;
    if (!check(sum_p_, 2.0*5, thr, "Error when swapping up two Vectors.")) stat_ += 1;

    return stat_;
};

int test_scale_v_cpp(){

    int stat_ = 0;
    
    Vector<double> p(5, 2.0);

    ScaleVector(2.0, p);

    auto sum_p = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_p, 4.0*5, thr2, "Error when scaling a Vector.")) stat_ += 1;
    ScaleVector(0.5, p, 1);

    auto sum_p_ = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_p_, 2.0*5, thr2, "Error when scaling a Vector.")) stat_ += 1;

    return stat_;
};

int test_scale_vf_cpp(){

    int stat_ = 0;
    
    Vector<float> p(5, 2.0);

    ScaleVector(2.0, p);

    auto sum_p = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_p, 4.0*5, thr, "Error when scaling a Vector.")) stat_ += 1;
    ScaleVector(0.5, p, 1);

    auto sum_p_ = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_p_, 2.0*5, thr, "Error when scaling a Vector.")) stat_ += 1;

    return stat_;
};

int test_inner_v_prod_cpp(){

    int stat_ = 0;
    
    Vector<double> p(5, 2.0);
    Vector<double> s(5, 1.0);

    auto prod = InnerVectorProduct(p, s);

    if (!check(prod, 2.0*5, thr2, "Error when swapping two Vectors.")) stat_ += 1;
    auto prod_ = InnerVectorProduct(p, 1, s, 1);

    if (!check(prod_, prod, thr2, "Error when swapping up two Vectors.")) stat_ += 1;

    return stat_;
};

int test_inner_vf_prod_cpp(){

    int stat_ = 0;
    
    Vector<float> p(5, 2.0);
    Vector<float> s(5, 1.0);

    auto prod = InnerVectorProduct(p, s);

    if (!check(prod, 2.0*5, thr2, "Error when swapping two Vectors.")) stat_ += 1;
    auto prod_ = InnerVectorProduct(p, 1, s, 1);

    if (!check(prod_, prod, thr2, "Error when swapping up two Vectors.")) stat_ += 1;

    return stat_;
};


int main(void){
    int stat = 0;
    stat += test_v_addition_c();
    stat += test_v_addition_cpp();
    stat += test_v_addition_and_scale_c();
    stat += test_v_addition_and_scale_cpp();
    stat += test_vf_addition_c();
    stat += test_vf_addition_cpp();
    stat += test_vf_addition_and_scale_c();
    stat += test_vf_addition_and_scale_cpp();
    stat += test_copy_v_c();
    stat += test_copy_vf_c();
    stat += test_copy_v_cpp();
    stat += test_copy_vf_cpp();
    stat += test_swap_v_c();
    stat += test_swap_vf_c(); 
    stat += test_swap_v_cpp();
    stat += test_swap_vf_cpp();
    stat += test_scale_v_c();
    stat += test_scale_vf_c(); 
    stat += test_scale_v_cpp();
    stat += test_scale_vf_cpp();
    stat += test_inner_v_prod_c();
    stat += test_inner_vf_prod_c();
    stat += test_inner_v_prod_cpp();
    stat += test_inner_vf_prod_cpp();

    return stat;
};