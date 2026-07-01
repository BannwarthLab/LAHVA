#include "test_common.h"
using namespace lahva::gpu;

template <typename T>
int test_v_addition_cpp(CudaRuntime& cudart){

    int stat_ = 0;
    double thr = get_tolerance<T>();

    Vector<T> p(5, (T)2.0);
    Vector<T> s(5, (T)1.0);

    AddVectors(cudart, (T)1.0, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum = std::accumulate(s.begin(), s.end(), (double)0.0);

    if (!check<T>(sum, 3.0*p.size(), thr, "Error when adding up two Vectors without scaling.")) stat_ +=1 ;

    AddVectors(cudart, (T)1.0, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_ = std::accumulate(s.begin(), s.end(), (double)0.0);

    if (!check<T>(sum_, 5.0*p.size(), thr, "Error when adding up two Vectors without scaling.")) stat_ +=1 ;

    return stat_;

};

template <typename T>
int test_v_addition_and_scale_cpp(CudaRuntime& cudart){
    int stat_ = 0;
    double thr = get_tolerance<T>();
    Vector<T> p(5, (T)2.0);
    Vector<T> s(5, (T)1.0);

    AddVectors(cudart, (T)2.0, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum = std::accumulate(s.begin(), s.end(), (double)0.0);

    if (!check<T>(sum, 5.0*p.size(), thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    AddVectors(cudart, (T)2.0, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_ = std::accumulate(s.begin(), s.end(), (double)0.0);
    if (!check<T>(sum_, 9.0*p.size(), thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;


    return stat_;

};

template <typename T>
int test_copy_v_cpp(CudaRuntime& cudart){

    int stat_ = 0;
    double thr = get_tolerance<T>();

    Vector<T> p(5, (T)2.0);
    Vector<T> s(5);

    CopyVectors(cudart, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_s = std::accumulate(s.begin(), s.end(), (double)0.0);
    auto sum_p = std::accumulate(p.begin(), p.end(), (double)0.0);

    if (!check<T>(sum_s, sum_p, thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    CopyVectors(cudart, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_s_ = std::accumulate(s.begin(), s.end(), (double)0.0);
    auto sum_p_ = std::accumulate(p.begin(), p.end(), (double)0.0);

    if (!check<T>(sum_s_, sum_p_, thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    return stat_;
};

template <typename T>
int test_swap_v_cpp(CudaRuntime& cudart){

    int stat_ = 0;
    double thr = get_tolerance<T>();

    Vector<T> p(5, (T)2.0);
    Vector<T> s(5, (T)1.0);

    SwapVectors(cudart, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_s = std::accumulate(s.begin(), s.end(), (double)0.0);
    auto sum_p = std::accumulate(p.begin(), p.end(), (double)0.0);

    if (!check<T>(sum_s, 2.0*5, thr, "Error when swapping two Vectors.")) stat_ += 1;
    if (!check<T>(sum_p, 1.0*5, thr, "Error when swapping two Vectors.")) stat_ += 1;
    SwapVectors(cudart, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_s_ = std::accumulate(s.begin(), s.end(), (double)0.0);
    auto sum_p_ = std::accumulate(p.begin(), p.end(), (double)0.0);

    if (!check<T>(sum_s_, 1.0*5, thr, "Error when swapping up two Vectors.")) stat_ += 1;
    if (!check<T>(sum_p_, 2.0*5, thr, "Error when swapping up two Vectors.")) stat_ += 1;

    return stat_;
};

template <typename T>
int test_scale_v_cpp(CudaRuntime& cudart){

    int stat_ = 0;
    double thr = get_tolerance<T>();

    Vector<T> p(5, (T)2.0);

    ScaleVector(cudart, (T)2.0, p);
    p.copy2host(cudart);

    auto sum_p = std::accumulate(p.begin(), p.end(), (double)0.0);

    if (!check<T>(sum_p, 4.0*5, thr, "Error when scaling a Vector.")) stat_ += 1;
    ScaleVector(cudart, (T)0.5, p, 1);
    p.copy2host(cudart);
    auto sum_p_ = std::accumulate(p.begin(), p.end(), (double)0.0);

    if (!check<T>(sum_p_, 2.0*5, thr, "Error when scaling a Vector.")) stat_ += 1;

    return stat_;
};

template <typename T>
int test_inner_v_prod_cpp(CudaRuntime& cudart){

    int stat_ = 0;
    double thr = get_tolerance<T>();

    Vector<T> p(5, (T)2.0);
    Vector<T> s(5, (T)1.0);

    auto prod = InnerVectorProduct(cudart, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    if (!check<T>(prod, 2.0*5, thr, "Error when swapping two Vectors.")) stat_ += 1;
    auto prod_ = InnerVectorProduct(cudart, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    if (!check<T>(prod_, prod, thr, "Error when swapping up two Vectors.")) stat_ += 1;

    return stat_;
};


int main(void){
    CudaRuntime cudart;
    cudart.print_cuda_version();
    int stat = 0;
    stat += test_v_addition_cpp<double>(cudart);
    stat += test_v_addition_cpp<float>(cudart);
    stat += test_v_addition_and_scale_cpp<double>(cudart);
    stat += test_v_addition_and_scale_cpp<float>(cudart);
    stat += test_copy_v_cpp<double>(cudart);
    stat += test_copy_v_cpp<float>(cudart);
    stat += test_swap_v_cpp<double>(cudart);
    stat += test_swap_v_cpp<float>(cudart);
    stat += test_scale_v_cpp<double>(cudart);
    stat += test_scale_v_cpp<float>(cudart);
    stat += test_inner_v_prod_cpp<double>(cudart);
    stat += test_inner_v_prod_cpp<float>(cudart);

    return stat;
};