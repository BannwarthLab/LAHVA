#include "common.h"
using namespace lahva::gpu;
const double thr2 = 5.0e-15;
const float thr = 5.0e-7;

int test_v_addition_cpp(CudaRuntime& cudart){

    int stat_ = 0;
    
    Vector<double> p(5, 2.0);
    Vector<double> s(5, 1.0);

    AddVectors(cudart, 1.0, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum = std::accumulate(s.begin(), s.end(), 0.0);

    if (!check(sum, 3.0*p.size(), thr2, "Error when adding up two Vectors without scaling.")) stat_ +=1 ;

    AddVectors(cudart, 1.0, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_ = std::accumulate(s.begin(), s.end(), 0.0);

    if (!check(sum_, 5.0*p.size(), thr2, "Error when adding up two Vectors without scaling.")) stat_ +=1 ; 

    return stat_;

};

int test_v_addition_and_scale_cpp(CudaRuntime& cudart){
    int stat_ = 0;
    Vector<double> p(5, 2.0);
    Vector<double> s(5, 1.0);

    AddVectors(cudart, 2.0, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum = std::accumulate(s.begin(), s.end(), 0.0);

    if (!check(sum, 5.0*p.size(), thr2, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    AddVectors(cudart, 2.0, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_ = std::accumulate(s.begin(), s.end(), 0.0);
    if (!check(sum_, 9.0*p.size(), thr2, "Error when adding up two Vectors without scaling.")) stat_ += 1;
    

    return stat_;

};


int test_vf_addition_cpp(CudaRuntime& cudart){

    int stat_ = 0;
    
    Vector<float> p(5, 2.0);
    Vector<float> s(5, 1.0);

    AddVectors(cudart, 1.0, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum = std::accumulate(s.begin(), s.end(), 0.0);

    if (!check(sum, 3.0*p.size(), thr, "Error when adding up two Vectors without scaling.")) stat_ +=1 ;

    AddVectors(cudart, 1.0, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_ = std::accumulate(s.begin(), s.end(), 0.0);

    if (!check(sum_, 5.0*p.size(), thr, "Error when adding up two Vectors without scaling.")) stat_ +=1 ; 

    return stat_;

};

int test_vf_addition_and_scale_cpp(CudaRuntime& cudart){
    int stat_ = 0;
    Vector<float> p(5, 2.0);
    Vector<float> s(5, 1.0);

    AddVectors(cudart, 2.0, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum = std::accumulate(s.begin(), s.end(), 0.0);

    if (!check(sum, 5.0*p.size(), thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    AddVectors(cudart, 2.0, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_ = std::accumulate(s.begin(), s.end(), 0.0);
    if (!check(sum_, 9.0*p.size(), thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;
    

    return stat_;

};

int test_copy_v_cpp(CudaRuntime& cudart){

    int stat_ = 0;
    
    Vector<double> p(5, 2.0);
    Vector<double> s(5);

    CopyVectors(cudart, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_s = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s, sum_p, thr2, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    CopyVectors(cudart, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_s_ = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p_ = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s_, sum_p_, thr2, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    return stat_;
};

int test_copy_vf_cpp(CudaRuntime& cudart){

    int stat_ = 0;
    
    Vector<float> p(5, 2.0);
    Vector<float> s(5);

    CopyVectors(cudart, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_s = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s, sum_p, thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    CopyVectors(cudart, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_s_ = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p_ = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s_, sum_p_, thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    return stat_;
};

int test_swap_v_cpp(CudaRuntime& cudart){

    int stat_ = 0;
    
    Vector<double> p(5, 2.0);
    Vector<double> s(5, 1.0);

    SwapVectors(cudart, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_s = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s, 2.0*5, thr2, "Error when swapping two Vectors.")) stat_ += 1;
    if (!check(sum_p, 1.0*5, thr2, "Error when swapping two Vectors.")) stat_ += 1;
    SwapVectors(cudart, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_s_ = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p_ = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s_, 1.0*5, thr2, "Error when swapping up two Vectors.")) stat_ += 1;
    if (!check(sum_p_, 2.0*5, thr2, "Error when swapping up two Vectors.")) stat_ += 1;

    return stat_;
};

int test_swap_vf_cpp(CudaRuntime& cudart){

    int stat_ = 0;
    
    Vector<float> p(5, 2.0);
    Vector<float> s(5, 1.0);

    SwapVectors(cudart, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_s = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s, 2.0*5, thr, "Error when swapping two Vectors.")) stat_ += 1;
    if (!check(sum_p, 1.0*5, thr, "Error when swapping two Vectors.")) stat_ += 1;
    SwapVectors(cudart, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_s_ = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p_ = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s_, 1.0*5, thr, "Error when swapping up two Vectors.")) stat_ += 1;
    if (!check(sum_p_, 2.0*5, thr, "Error when swapping up two Vectors.")) stat_ += 1;

    return stat_;
};

int test_scale_v_cpp(CudaRuntime& cudart){

    int stat_ = 0;
    
    Vector<double> p(5, 2.0);

    ScaleVector(cudart, 2.0, p);
    p.copy2host(cudart);

    auto sum_p = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_p, 4.0*5, thr2, "Error when scaling a Vector.")) stat_ += 1;
    ScaleVector(cudart, 0.5, p, 1);
    p.copy2host(cudart);
    auto sum_p_ = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_p_, 2.0*5, thr2, "Error when scaling a Vector.")) stat_ += 1;

    return stat_;
};

int test_scale_vf_cpp(CudaRuntime& cudart){

    int stat_ = 0;
    
    Vector<float> p(5, 2.0);

    ScaleVector(cudart, 2.0, p);
    p.copy2host(cudart);

    auto sum_p = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_p, 4.0*5, thr, "Error when scaling a Vector.")) stat_ += 1;
    ScaleVector(cudart, 0.5, p, 1);
    p.copy2host(cudart);
    auto sum_p_ = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_p_, 2.0*5, thr, "Error when scaling a Vector.")) stat_ += 1;

    return stat_;
};

int test_inner_v_prod_cpp(CudaRuntime& cudart){

    int stat_ = 0;
    
    Vector<double> p(5, 2.0);
    Vector<double> s(5, 1.0);

    auto prod = InnerVectorProduct(cudart, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    if (!check(prod, 2.0*5, thr2, "Error when swapping two Vectors.")) stat_ += 1;
    auto prod_ = InnerVectorProduct(cudart, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    if (!check(prod_, prod, thr2, "Error when swapping up two Vectors.")) stat_ += 1;

    return stat_;
};

int test_inner_vf_prod_cpp(CudaRuntime& cudart){

    int stat_ = 0;
    
    Vector<float> p(5, 2.0);
    Vector<float> s(5, 1.0);

    auto prod = InnerVectorProduct(cudart, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    if (!check(prod, 2.0*5, thr2, "Error when swapping two Vectors.")) stat_ += 1;
    auto prod_ = InnerVectorProduct(cudart, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    if (!check(prod_, prod, thr2, "Error when swapping up two Vectors.")) stat_ += 1;

    return stat_;
};


int main(void){
    CudaRuntime cudart;
    cudart.print_cuda_version();
    int stat = 0;
    stat += test_v_addition_cpp(cudart);
    stat += test_v_addition_and_scale_cpp(cudart);
    stat += test_vf_addition_cpp(cudart);
    stat += test_vf_addition_and_scale_cpp(cudart);
    stat += test_copy_v_cpp(cudart);
    stat += test_copy_vf_cpp(cudart);
    stat += test_swap_v_cpp(cudart);
    stat += test_swap_vf_cpp(cudart);
    stat += test_scale_v_cpp(cudart);
    stat += test_scale_vf_cpp(cudart);
    stat += test_inner_v_prod_cpp(cudart);
    stat += test_inner_vf_prod_cpp(cudart);

    return stat;
};