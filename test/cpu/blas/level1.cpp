#include "test_common.h"

using namespace lahva::cpu;

template <typename T>
int test_v_addition_c(){

    int stat_ = 0;
    double thr = get_tolerance<T>();

    T v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    T v2[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    T* p = v1;
    T* s = v2;

    AddVectors(5, 1.0, p, s);

    auto sum_ = sum(5, s);

    if (!check(sum_, 3.0*5, thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    AddVectors(5, 1.0, p, 1, s, 1);

    auto sum__ = sum(5, s);

    if (!check(sum__, 5.0*5, thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    return stat_;
};

template <typename T>
int test_v_addition_and_scale_c(){

    int stat_ = 0;
    double thr = get_tolerance<T>();

    T v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    T v2[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    T* p = v1;
    T* s = v2;

    AddVectors(5, 2.0, p, s);

    auto sum_ = sum(5, s);

    if (!check(sum_, 5.0*5, thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    AddVectors(5, 1.0, p, 1, s, 1);

    auto sum__ = sum(5, s);

    if (!check(sum__, 7.0*5, thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    return stat_;
};

template <typename T>
int test_copy_v_c(){

    int stat_ = 0;
    double thr = get_tolerance<T>();

    T v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    T v2[5] ;
    T* p = v1;
    T* s = v2;

    CopyVectors(5, p, s);

    auto sum_ = sum(5, s);
    auto sum_p = sum(5, p);

    if (!check(sum_, sum_p, thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    CopyVectors(5, p, 1, s, 1);

    auto sum__ = sum(5, s);

    if (!check(sum__, sum_p, thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    return stat_;
};

template <typename T>
int test_swap_v_c(){

    int stat_ = 0;
    double thr = get_tolerance<T>();

    T v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    T v2[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    T* p = v1;
    T* s = v2;

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

template <typename T>
int test_scale_v_c(){

    int stat_ = 0;
    double thr = get_tolerance<T>();

    T v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    T* p = v1;

    ScaleVector(5, 2.0, p);

    auto sum_p = sum(5, p);

    if (!check(sum_p, 4.0*5, thr, "Error when scaling a Vector.")) stat_ += 1;

    ScaleVector(5, 0.5, p);

    auto sum_p_ = sum(5, p);

    if (!check(sum_p_, 2.0*5, thr, "Error when scaling a Vector.")) stat_ += 1;

    return stat_;
};

template <typename T>
int test_v_addition_cpp(){

    int stat_ = 0;
    double thr = get_tolerance<T>();

    Vector<T> p(5, 2.0);
    Vector<T> s(5, 1.0);

    AddVectors(1.0, p, s);

    auto sum = std::accumulate(s.begin(), s.end(), 0.0);

    if (!check(sum, 3.0*p.size(), thr, "Error when adding up two Vectors without scaling.")) stat_ +=1 ;

    AddVectors(1.0, p, 1, s, 1);

    auto sum_ = std::accumulate(s.begin(), s.end(), 0.0);

    if (!check(sum_, 5.0*p.size(), thr, "Error when adding up two Vectors without scaling.")) stat_ +=1 ;

    return stat_;

};

template <typename T>
int test_v_addition_and_scale_cpp(){
    int stat_ = 0;
    double thr = get_tolerance<T>();

    Vector<T> p(5, 2.0);
    Vector<T> s(5, 1.0);

    AddVectors(2.0, p, s);

    auto sum = std::accumulate(s.begin(), s.end(), 0.0);

    if (!check(sum, 5.0*p.size(), thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    AddVectors(2.0, p, 1, s, 1);

    auto sum_ = std::accumulate(s.begin(), s.end(), 0.0);
    if (!check(sum_, 9.0*p.size(), thr, "Error when adding up two Vectors without scaling.")) stat_ += 1;

    return stat_;

};

template <typename T>
int test_inner_v_prod_c(){

    int stat_ = 0;
    double thr = get_tolerance<T>();

    T v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    T v2[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    T* p = v1;
    T* s = v2;

    auto prod = InnerVectorProduct(5, p, s);

    if (!check(prod, 2.0*5, thr, "Error when computing inner product of two Vectors.")) stat_ += 1;

    auto prod_ = InnerVectorProduct(5, p, 1, s, 1);

    if (!check(prod_, prod, thr, "Error when computing inner product of two Vectors.")) stat_ += 1;

    return stat_;
};


template <typename T>
int test_copy_v_cpp(){

    int stat_ = 0;
    double thr = get_tolerance<T>();

    Vector<T> p(5, 2.0);
    Vector<T> s(5);

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

template <typename T>
int test_swap_v_cpp(){

    int stat_ = 0;
    double thr = get_tolerance<T>();

    Vector<T> p(5, 2.0);
    Vector<T> s(5, 1.0);

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

template <typename T>
int test_scale_v_cpp(){

    int stat_ = 0;
    double thr = get_tolerance<T>();

    Vector<T> p(5, 2.0);

    ScaleVector(2.0, p);

    auto sum_p = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_p, 4.0*5, thr, "Error when scaling a Vector.")) stat_ += 1;
    ScaleVector(0.5, p, 1);

    auto sum_p_ = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_p_, 2.0*5, thr, "Error when scaling a Vector.")) stat_ += 1;

    return stat_;
};

template <typename T>
int test_inner_v_prod_cpp(){

    int stat_ = 0;
    double thr = get_tolerance<T>();

    Vector<T> p(5, 2.0);
    Vector<T> s(5, 1.0);

    auto prod = InnerVectorProduct(p, s);

    if (!check(prod, 2.0*5, thr, "Error when swapping two Vectors.")) stat_ += 1;
    auto prod_ = InnerVectorProduct(p, 1, s, 1);

    if (!check(prod_, prod, thr, "Error when swapping up two Vectors.")) stat_ += 1;

    return stat_;
};


int main(void){
    int stat = 0;
    stat += test_v_addition_c<double>();
    stat += test_v_addition_c<float>();
    stat += test_v_addition_cpp<double>();
    stat += test_v_addition_cpp<float>();
    stat += test_v_addition_and_scale_c<double>();
    stat += test_v_addition_and_scale_c<float>();
    stat += test_v_addition_and_scale_cpp<double>();
    stat += test_v_addition_and_scale_cpp<float>();
    stat += test_copy_v_c<double>();
    stat += test_copy_v_c<float>();
    stat += test_copy_v_cpp<double>();
    stat += test_copy_v_cpp<float>();
    stat += test_swap_v_c<double>();
    stat += test_swap_v_c<float>();
    stat += test_swap_v_cpp<double>();
    stat += test_swap_v_cpp<float>();
    stat += test_scale_v_c<double>();
    stat += test_scale_v_c<float>();
    stat += test_scale_v_cpp<double>();
    stat += test_scale_v_cpp<float>();
    stat += test_inner_v_prod_c<double>();
    stat += test_inner_v_prod_c<float>();
    stat += test_inner_v_prod_cpp<double>();
    stat += test_inner_v_prod_cpp<float>();

    return stat;
};