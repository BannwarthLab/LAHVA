#pragma once
#include "linalg.hpp"
#include "lahva.hpp"
#include "lahva.h"
#include "utils.hpp"
#ifdef _CUDA
#include "runtime.hpp"
#endif
#include <iostream>
#include <numeric>

using namespace lahva;

// Test result codes
#define TEST_PASS 0
#define TEST_FAIL 1

// Default tolerance values
const double DEFAULT_DOUBLE_TOLERANCE = 5.0e-7;
const double DEFAULT_STRICT_DOUBLE_TOLERANCE = 5.0e-15;
const float DEFAULT_FLOAT_TOLERANCE = 5.0e-7f;

// Tolerance selector for templated tests
template <typename T>
inline double get_tolerance() {
    if (std::is_same_v<T, float>) {
        return DEFAULT_FLOAT_TOLERANCE;
    }
    return DEFAULT_DOUBLE_TOLERANCE;
}

// Strict tolerance selector for high-precision tests
template <typename T>
inline double get_strict_tolerance() {
    if (std::is_same_v<T, float>) {
        return 1.0e-6;  // Strict tolerance for float
    }
    return DEFAULT_STRICT_DOUBLE_TOLERANCE;
}
