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
constexpr const char* get_type_name() {
    if constexpr (std::is_same_v<T, double>) return "double";
    else if constexpr (std::is_same_v<T, float>) return "float";
    else if constexpr (std::is_same_v<T, int>) return "int";
    else if constexpr (std::is_same_v<T, complex_float>) return "complex_float";
    else if constexpr (std::is_same_v<T, complex_double>) return "complex_double";
#ifdef _CUDA
    else if constexpr (std::is_same_v<T, __half>) return "__half";
#endif
    else return "unknown";
}

// Overload for std::string description
inline const char* make_check_msg(const char* func_name, const char* type_name, const std::string& description) {
    static std::string msg;
    msg = "[";
    msg += func_name;
    msg += " (";
    msg += type_name;
    msg += ")] ";
    msg += description;
    return msg.c_str();
}
