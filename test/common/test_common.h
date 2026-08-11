#pragma once
#include "linalg.hpp"
#include "lahva.hpp"
#include "lahva.h"
#ifdef _CUDA
#include "runtime.hpp"
#endif
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include "comparators.hpp"

// Test result codes
#define TEST_PASS 0
#define TEST_FAIL 1

// Helper to get type name as string
template <typename T>
std::string get_type_name() {
    if constexpr (std::is_same_v<T, double>) return std::string("double");
    else if constexpr (std::is_same_v<T, float>) return std::string("float");
    else if constexpr (std::is_same_v<T, int>) return std::string("int");
    else if constexpr (std::is_same_v<T, complex_float>) return std::string("complex_float");
    else if constexpr (std::is_same_v<T, complex_double>) return std::string("complex_double");
#ifdef _CUDA
    else if constexpr (std::is_same_v<T, __half>) return std::string("__half");
#endif
    else return std::string("unknown");
}

// Helper to create a check message
inline std::string make_check_msg(std::string_view func_name, std::string_view type_name, std::string_view description) {
    std::string msg;
    msg.reserve(func_name.size() + type_name.size() + description.size() + 6);
    msg += '[';
    msg += func_name;
    msg += " (";
    msg += type_name;
    msg += ")] ";
    msg += description;
    return msg;
}

// Macro to automatically capture __func__
#define check_msg(type_name, desc) make_check_msg(__func__, type_name, desc)
