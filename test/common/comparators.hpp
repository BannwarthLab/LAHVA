#pragma once
#include <iomanip>
#include <algorithm>
#include <limits>
#include <complex>
#include <cmath>
#include "tolerances.hpp"

template <ToleranceType TType = ToleranceType::BLAS, typename T>
bool compareNumbers(T a, T b)
{
    T rel_tol = get_relative_tolerance<T, TType>();
    T abs_tol = get_absolute_tolerance<T, TType>();

    bool relative = std::abs(a - b) < rel_tol * std::max(
    std::numeric_limits<T>::epsilon() * std::max(std::abs(a), std::abs(b)),
    std::numeric_limits<T>::denorm_min());

    bool absolute = std::abs(a - b) < abs_tol;

    return relative || absolute;
}

template <ToleranceType TType = ToleranceType::BLAS, typename T>
bool compareNumbers(std::complex<T> a, std::complex<T> b)
{
    T rel_tol = get_relative_tolerance<T, TType>();
    T abs_tol = get_absolute_tolerance<T, TType>();

    bool relative = std::abs(a.real() - b.real()) < rel_tol * std::max(
    std::numeric_limits<T>::epsilon() * std::max(std::abs(a.real()), std::abs(b.real())),
    std::numeric_limits<T>::denorm_min()) &&
    std::abs(a.imag() - b.imag()) < rel_tol * std::max(
    std::numeric_limits<T>::epsilon() * std::max(std::abs(a.imag()), std::abs(b.imag())),
    std::numeric_limits<T>::denorm_min());

    bool absolute = std::abs(a.real() - b.real()) < abs_tol &&
                    std::abs(a.imag() - b.imag()) < abs_tol;

    return relative || absolute;
}

template <ToleranceType TType = ToleranceType::BLAS, typename T, typename U>
std::enable_if_t<!std::is_same_v<T, U>, bool> compareNumbers(T a, U b)
{
    float rel_tol = get_relative_tolerance<float, TType>();
    float abs_tol = get_absolute_tolerance<float, TType>();

    float float_a = static_cast<float>(a);
    float float_b = static_cast<float>(b);

    bool relative = std::abs(float_a - float_b) < rel_tol * std::max(
    std::numeric_limits<float>::epsilon() * std::max(std::abs(float_a), std::abs(float_b)),
    std::numeric_limits<float>::denorm_min());

    bool absolute = std::abs(float_a - float_b) < abs_tol;

    return relative || absolute;
}

template <>
inline bool compareNumbers<ToleranceType::BLAS, int>(int a, int b)
{
    return a==b;
}

template <typename T, ToleranceType TType = ToleranceType::BLAS>
bool check(T* a, T* b, int ndim, const char* msg)
{
    for (int i = 0; i != ndim; i++) {
        if (!compareNumbers<TType>(a[i], b[i])) {
            std::cerr << "[Fatal] " << msg << ": Array comparison failed at index " << i
                      << " (expected: " << std::setprecision(15) << b[i]
                      << ", got: " << std::setprecision(15) << a[i] << ")" << std::endl;
            return false;
        }
    }
    return true;
}

template <typename T, typename U, ToleranceType TType = ToleranceType::BLAS>
bool check(T* a, U* b, int ndim, const char* msg)
{
    for (int i = 0; i != ndim; i++) {
        if (!compareNumbers<TType>(a[i], b[i])) {
            std::cerr << "[Fatal] " << msg << ": Array comparison failed at index " << i
                      << " (expected: " << std::setprecision(15) << b[i]
                      << ", got: " << std::setprecision(15) << a[i] << ")" << std::endl;
            return false;
        }
    }
    return true;
}

template <typename T, typename U, ToleranceType TType = ToleranceType::BLAS>
bool check(T* a, U* b, const char* msg)
{
    if (!compareNumbers<TType>(a, b)) {
        std::cerr << "[Fatal] " << msg << " (expected: " << std::setprecision(15) << *b
                  << ", got: " << std::setprecision(15) << *a << ")" << std::endl;
        return false;
    }
    return true;
}

template <typename T, typename U, ToleranceType TType = ToleranceType::BLAS>
bool check(T a, U b, const char* msg)
{
    if (!compareNumbers<TType>(a, b)) {
        std::cerr << "[Fatal] " << msg << " (expected: " << std::setprecision(15) << b
                  << ", got: " << std::setprecision(15) << a << ")" << std::endl;
        return false;
    }
    return true;
}
