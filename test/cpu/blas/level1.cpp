#include "test_common.h"
#include "array_utils.hpp"

using namespace lahva::cpu;

// ============================================================================
// C-Level Vector Operation Tests
// ============================================================================

template <typename T>
int test_v_addition_c(){

    T v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    T v2[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    T* p = v1;
    T* s = v2;

    AddVectors(5, 1.0, p, s);

    auto sum_ = sum(5, s);

    if (!check(sum_, 3.0*5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    AddVectors(5, 1.0, p, 1, s, 1);

    auto sum__ = sum(5, s);

    if (!check(sum__, 5.0*5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
};

template <typename T>
int test_v_addition_and_scale_c(){

    T v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    T v2[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    T* p = v1;
    T* s = v2;

    AddVectors(5, 2.0, p, s);

    auto sum_ = sum(5, s);

    if (!check(sum_, 5.0*5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    AddVectors(5, 1.0, p, 1, s, 1);

    auto sum__ = sum(5, s);

    if (!check(sum__, 7.0*5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
};

template <typename T>
int test_copy_v_c(){
    T v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    T v2[5] ;
    T* p = v1;
    T* s = v2;

    CopyVectors(5, p, s);

    auto sum_ = sum(5, s);
    auto sum_p = sum(5, p);

    if (!check(sum_, sum_p, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    CopyVectors(5, p, 1, s, 1);

    auto sum__ = sum(5, s);

    if (!check(sum__, sum_p, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
};

template <typename T>
int test_swap_v_c(){
    T v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    T v2[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    T* p = v1;
    T* s = v2;

    SwapVectors(5, p, s);

    auto sum_s = sum(5, s);
    auto sum_p = sum(5, p);

    if (!check(sum_s, 2.0*5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;
    if (!check(sum_p, 1.0*5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    SwapVectors(5, p, 1, s, 1);

    auto sum_s_ = sum(5, s);
    auto sum_p_ = sum(5, p);

    if (!check(sum_s_, 1.0*5, check_msg(get_type_name<T>(), "check 3"))) return TEST_FAIL;
    if (!check(sum_p_, 2.0*5, check_msg(get_type_name<T>(), "check 4"))) return TEST_FAIL;

    return TEST_PASS;
};

template <typename T>
int test_scale_v_c(){
    T v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    T* p = v1;

    ScaleVector(5, 2.0, p);

    auto sum_p = sum(5, p);

    if (!check(sum_p, 4.0*5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    ScaleVector(5, 0.5, p);

    auto sum_p_ = sum(5, p);

    if (!check(sum_p_, 2.0*5, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
};

// ============================================================================
// C++ Wrapper Vector Operation Tests
// ============================================================================

template <typename T>
int test_v_addition_cpp(){
    Vector<T> p(5, 2.0);
    Vector<T> s(5, 1.0);

    AddVectors(1.0, p, s);

    auto sum = std::accumulate(s.begin(), s.end(), 0.0);

    if (!check(sum, 3.0*p.size(), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    AddVectors(1.0, p, 1, s, 1);

    auto sum_ = std::accumulate(s.begin(), s.end(), 0.0);

    if (!check(sum_, 5.0*p.size(), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
};

template <typename T>
int test_v_addition_and_scale_cpp(){
    Vector<T> p(5, 2.0);
    Vector<T> s(5, 1.0);

    AddVectors(2.0, p, s);

    auto sum = std::accumulate(s.begin(), s.end(), 0.0);

    if (!check(sum, 5.0*p.size(), check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    AddVectors(2.0, p, 1, s, 1);

    auto sum_ = std::accumulate(s.begin(), s.end(), 0.0);
    if (!check(sum_, 9.0*p.size(), check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
};

template <typename T>
int test_inner_v_prod_c(){
    T v1[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    T v2[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    T* p = v1;
    T* s = v2;

    auto prod = InnerVectorProduct(5, p, s);

    if (!check(prod, 2.0*5, check_msg(get_type_name<T>(), "check 1"))) return TEST_FAIL;

    auto prod_ = InnerVectorProduct(5, p, 1, s, 1);

    if (!check(prod_, prod, check_msg(get_type_name<T>(), "check 2"))) return TEST_FAIL;

    return TEST_PASS;
};


template <typename T>
int test_copy_v_cpp(){
    Vector<T> p(5, 2.0);
    Vector<T> s(5);

    CopyVectors(p, s);

    auto sum_s = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s, sum_p, "check 1")) return TEST_FAIL;

    CopyVectors(p, 1, s, 1);

    auto sum_s_ = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p_ = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s_, sum_p_, "check 2")) return TEST_FAIL;

    return TEST_PASS;
};

template <typename T>
int test_swap_v_cpp(){
    Vector<T> p(5, 2.0);
    Vector<T> s(5, 1.0);

    SwapVectors(p, s);

    auto sum_s = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s, 2.0*5, "check 1")) return TEST_FAIL;
    if (!check(sum_p, 1.0*5, "check 2")) return TEST_FAIL;
    SwapVectors(p, 1, s, 1);

    auto sum_s_ = std::accumulate(s.begin(), s.end(), 0.0);
    auto sum_p_ = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_s_, 1.0*5, "check 3")) return TEST_FAIL;
    if (!check(sum_p_, 2.0*5, "check 4")) return TEST_FAIL;

    return TEST_PASS;
};

template <typename T>
int test_scale_v_cpp(){
    Vector<T> p(5, 2.0);

    ScaleVector(2.0, p);

    auto sum_p = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_p, 4.0*5, "check 1")) return TEST_FAIL;
    ScaleVector(0.5, p, 1);

    auto sum_p_ = std::accumulate(p.begin(), p.end(), 0.0);

    if (!check(sum_p_, 2.0*5, "check 2")) return TEST_FAIL;

    return TEST_PASS;
};

template <typename T>
int test_inner_v_prod_cpp(){
    Vector<T> p(5, 2.0);
    Vector<T> s(5, 1.0);

    auto prod = InnerVectorProduct(p, s);

    if (!check(prod, 2.0*5, "check 1")) return TEST_FAIL;
    auto prod_ = InnerVectorProduct(p, 1, s, 1);

    if (!check(prod_, prod, "check 2")) return TEST_FAIL;

    return TEST_PASS;
};

// ============================================================================
// Main
// ============================================================================

int main(){
    int total_failures = 0;

    // C-level vector operation tests
    total_failures += test_v_addition_c<double>();
    total_failures += test_v_addition_c<float>();
    total_failures += test_v_addition_and_scale_c<double>();
    total_failures += test_v_addition_and_scale_c<float>();
    total_failures += test_copy_v_c<double>();
    total_failures += test_copy_v_c<float>();
    total_failures += test_swap_v_c<double>();
    total_failures += test_swap_v_c<float>();
    total_failures += test_scale_v_c<double>();
    total_failures += test_scale_v_c<float>();
    total_failures += test_inner_v_prod_c<double>();
    total_failures += test_inner_v_prod_c<float>();

    // C++ wrapper vector operation tests
    total_failures += test_v_addition_cpp<double>();
    total_failures += test_v_addition_cpp<float>();
    total_failures += test_v_addition_and_scale_cpp<double>();
    total_failures += test_v_addition_and_scale_cpp<float>();
    total_failures += test_copy_v_cpp<double>();
    total_failures += test_copy_v_cpp<float>();
    total_failures += test_swap_v_cpp<double>();
    total_failures += test_swap_v_cpp<float>();
    total_failures += test_scale_v_cpp<double>();
    total_failures += test_scale_v_cpp<float>();
    total_failures += test_inner_v_prod_cpp<double>();
    total_failures += test_inner_v_prod_cpp<float>();

    if (total_failures > 0) {
        std::cerr << "cpu/blas/level1 tests: " << total_failures << " failures" << std::endl;
        return TEST_FAIL;
    }

    std::cout << "All cpu/blas/level1 tests passed!" << std::endl;
    return TEST_PASS;
};