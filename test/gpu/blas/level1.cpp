#include "test_common.h"
#include "array_utils.hpp"

using namespace lahva::gpu;
using lahva::CudaRuntime;

template <typename T>
int test_v_addition_cpp(CudaRuntime& cudart){

    Vector<T> p(5, (T)2.0);
    Vector<T> s(5, (T)1.0);

    AddVectors(cudart, (T)1.0, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum = std::accumulate(s.begin(), s.end(), (double)0.0);

    if (!check<T>(sum, 3.0*p.size(), make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    AddVectors(cudart, (T)1.0, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_ = std::accumulate(s.begin(), s.end(), (double)0.0);

    if (!check<T>(sum_, 5.0*p.size(), make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;

};

template <typename T>
int test_v_addition_and_scale_cpp(CudaRuntime& cudart){
        Vector<T> p(5, (T)2.0);
    Vector<T> s(5, (T)1.0);

    AddVectors(cudart, (T)2.0, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum = std::accumulate(s.begin(), s.end(), (double)0.0);

    if (!check<T>(sum, 5.0*p.size(), make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    AddVectors(cudart, (T)2.0, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_ = std::accumulate(s.begin(), s.end(), (double)0.0);
    if (!check<T>(sum_, 9.0*p.size(), make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;


    return TEST_PASS;

};

template <typename T>
int test_copy_v_cpp(CudaRuntime& cudart){

    
    Vector<T> p(5, (T)2.0);
    Vector<T> s(5);

    CopyVectors(cudart, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_s = std::accumulate(s.begin(), s.end(), (double)0.0);
    auto sum_p = std::accumulate(p.begin(), p.end(), (double)0.0);

    if (!check<T>(sum_s, sum_p, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;

    CopyVectors(cudart, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_s_ = std::accumulate(s.begin(), s.end(), (double)0.0);
    auto sum_p_ = std::accumulate(p.begin(), p.end(), (double)0.0);

    if (!check<T>(sum_s_, sum_p_, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
};

template <typename T>
int test_swap_v_cpp(CudaRuntime& cudart){

    
    Vector<T> p(5, (T)2.0);
    Vector<T> s(5, (T)1.0);

    SwapVectors(cudart, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_s = std::accumulate(s.begin(), s.end(), (double)0.0);
    auto sum_p = std::accumulate(p.begin(), p.end(), (double)0.0);

    if (!check<T>(sum_s, 2.0*5, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check<T>(sum_p, 1.0*5, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;
    SwapVectors(cudart, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    auto sum_s_ = std::accumulate(s.begin(), s.end(), (double)0.0);
    auto sum_p_ = std::accumulate(p.begin(), p.end(), (double)0.0);

    if (!check<T>(sum_s_, 1.0*5, make_check_msg(__func__, get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check<T>(sum_p_, 2.0*5, make_check_msg(__func__, get_type_name<T>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
};

template <typename T>
int test_scale_v_cpp(CudaRuntime& cudart){

    
    Vector<T> p(5, (T)2.0);

    ScaleVector(cudart, (T)2.0, p);
    p.copy2host(cudart);

    auto sum_p = std::accumulate(p.begin(), p.end(), (double)0.0);

    if (!check<T>(sum_p, 4.0*5, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;
    ScaleVector(cudart, (T)0.5, p, 1);
    p.copy2host(cudart);
    auto sum_p_ = std::accumulate(p.begin(), p.end(), (double)0.0);

    if (!check<T>(sum_p_, 2.0*5, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
};

template <typename T>
int test_inner_v_prod_cpp(CudaRuntime& cudart){

    
    Vector<T> p(5, (T)2.0);
    Vector<T> s(5, (T)1.0);

    auto prod = InnerVectorProduct(cudart, p, s);
    p.copy2host(cudart);
    s.copy2host(cudart);

    if (!check<T>(prod, 2.0*5, make_check_msg(__func__, get_type_name<T>(), "check 1"))) return TEST_FAIL;
    auto prod_ = InnerVectorProduct(cudart, p, 1, s, 1);
    p.copy2host(cudart);
    s.copy2host(cudart);

    if (!check<T>(prod_, prod, make_check_msg(__func__, get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
};

// ============================================================================
// Main
// ============================================================================

int main(){
    CudaRuntime cudart;

    int total_failures = 0;
    total_failures += test_v_addition_cpp<double>(cudart);
    total_failures += test_v_addition_cpp<float>(cudart);
    total_failures += test_v_addition_and_scale_cpp<double>(cudart);
    total_failures += test_v_addition_and_scale_cpp<float>(cudart);
    total_failures += test_copy_v_cpp<double>(cudart);
    total_failures += test_copy_v_cpp<float>(cudart);
    total_failures += test_swap_v_cpp<double>(cudart);
    total_failures += test_swap_v_cpp<float>(cudart);
    total_failures += test_scale_v_cpp<double>(cudart);
    total_failures += test_scale_v_cpp<float>(cudart);
    total_failures += test_inner_v_prod_cpp<double>(cudart);
    total_failures += test_inner_v_prod_cpp<float>(cudart);

    if (total_failures > 0) {
        std::cerr << "gpu/blas/level1 tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All gpu/blas/level1 tests passed!" << std::endl;
    return TEST_PASS;
};