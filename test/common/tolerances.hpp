#pragma once
#include <type_traits>

// Default tolerance values
const double DOUBLE_BLAS_ABSOLUTE_TOLERANCE = 5.0e-15;
const float FLOAT_BLAS_ABSOLUTE_TOLERANCE = 5.0e-7f;

const float FLOAT_BLAS_RELATIVE_TOLERANCE = 50;
const double DOUBLE_BLAS_RELATIVE_TOLERANCE = 50;

const double DOUBLE_LAPACK_ABSOLUTE_TOLERANCE = 1.0e-10;
const float FLOAT_LAPACK_ABSOLUTE_TOLERANCE = 4.5e-4f;

const float FLOAT_LAPACK_RELATIVE_TOLERANCE = 5e5;
const double DOUBLE_LAPACK_RELATIVE_TOLERANCE = 5e5;

enum class ToleranceType { BLAS, LAPACK };

// Tolerance selector for templated tests
template <typename T, ToleranceType TType = ToleranceType::BLAS>
inline double get_absolute_tolerance() {
    if (TType == ToleranceType::LAPACK) {
        if (std::is_same_v<T, float>) {
            return FLOAT_LAPACK_ABSOLUTE_TOLERANCE;
        }
        return DOUBLE_LAPACK_ABSOLUTE_TOLERANCE;
    }
    if (std::is_same_v<T, float>) {
        return FLOAT_BLAS_ABSOLUTE_TOLERANCE;
    }
    return DOUBLE_BLAS_ABSOLUTE_TOLERANCE;
}

template <typename T, ToleranceType TType = ToleranceType::BLAS>
inline double get_relative_tolerance() {
    if (TType == ToleranceType::LAPACK) {
        if (std::is_same_v<T, float>) {
            return FLOAT_LAPACK_RELATIVE_TOLERANCE;
        }
        return DOUBLE_LAPACK_RELATIVE_TOLERANCE;
    }
    if (std::is_same_v<T, float>) {
        return FLOAT_BLAS_RELATIVE_TOLERANCE;
    }
    return DOUBLE_BLAS_RELATIVE_TOLERANCE;
}
