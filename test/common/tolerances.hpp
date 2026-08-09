#pragma once
#include <type_traits>

// Default tolerance values
const double DOUBLE_ABSOLUTE_TOLERANCE = 5.0e-15;
const float FLOAT_ABSOLUTE_TOLERANCE = 5.0e-7f;

const float FLOAT_RELATIVE_TOLERANCE = 50.0;
const double DOUBLE_RELATIVE_TOLERANCE = 50.0;

// Tolerance selector for templated tests
template <typename T>
inline double get_absolute_tolerance() {
    if (std::is_same_v<T, float>) {
        return FLOAT_ABSOLUTE_TOLERANCE;
    }
    return DOUBLE_ABSOLUTE_TOLERANCE;
}

template <typename T>
inline double get_relative_tolerance() {
    if (std::is_same_v<T, float>) {
        return FLOAT_RELATIVE_TOLERANCE;
    }
    return DOUBLE_RELATIVE_TOLERANCE;
}
