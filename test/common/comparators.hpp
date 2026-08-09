#pragma once
#include <iomanip>
#include <algorithm>
#include <limits>
#include <complex>
#include <cmath>
#include "tolerances.hpp"

template <typename T>
bool compareNumbers(T a, T b)
{
    T rel_tol = get_relative_tolerance<T>();
    T abs_tol = get_absolute_tolerance<T>();

    bool relative = std::abs(a - b) < rel_tol * std::max(
    std::numeric_limits<T>::epsilon() * std::max(std::abs(a), std::abs(b)),
    std::numeric_limits<T>::denorm_min());

    bool absolute = std::abs(a - b) < abs_tol;

    return relative || absolute;
}

template <typename T, typename U>
bool compareNumbers(T a, U b)
{
    float rel_tol = get_relative_tolerance<float>();
    float abs_tol = get_absolute_tolerance<float>();

    double double_a = static_cast<float>(a);
    double double_b = static_cast<float>(b);

    bool relative = std::abs(double_a - double_b) < rel_tol * std::max(
    std::numeric_limits<float>::epsilon() * std::max(std::abs(static_cast<float>(double_a)), std::abs(static_cast<float>(double_b))),
    std::numeric_limits<float>::denorm_min());

    bool absolute = std::abs(double_a - double_b) < abs_tol;

    return relative || absolute;
}

template <>
inline bool compareNumbers<int>(int a, int b)
{
    return a==b;
}

template <typename T>
bool compareNumbers(std::complex<T> a, std::complex<T> b)
{
    T rel_tol = get_relative_tolerance<T>();
    T abs_tol = get_absolute_tolerance<T>();

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

template <typename T>
bool check(T* a, T* b, int ndim, const char* msg)
{
    for (int i = 0; i != ndim; i++) {
        if (!compareNumbers(a[i], b[i])) {
            std::cerr << "[Fatal] " << msg << ": Array comparison failed at index " << i
                      << " (expected: " << std::setprecision(15) << a[i]
                      << ", got: " << std::setprecision(15) << b[i] << ")" << std::endl;
            return false;
        }
    }
    return true;
}

template <typename T, typename U>
bool check(T* a, U* b, int ndim, const char* msg)
{
    for (int i = 0; i != ndim; i++) {
        if (!compareNumbers(a[i], b[i])) {
            std::cerr << "[Fatal] " << msg << ": Array comparison failed at index " << i
                      << " (expected: " << std::setprecision(15) << a[i]
                      << ", got: " << std::setprecision(15) << b[i] << ")" << std::endl;
            return false;
        }
    }
    return true;
}

template <typename T, typename U>
bool check(T* a, U* b, const char* msg)
{
    if (!compareNumbers(a, b)) {
        std::cerr << "[Fatal] " << msg << " (expected: " << std::setprecision(15) << *a
                  << ", got: " << std::setprecision(15) << *b << ")" << std::endl;
        return false;
    }
    return true;
}

template <typename T, typename U>
bool check(T a, U b, const char* msg)
{
    if (!compareNumbers(a, b)) {
        std::cerr << "[Fatal] " << msg << " (expected: " << std::setprecision(15) << a
                  << ", got: " << std::setprecision(15) << b << ")" << std::endl;
        return false;
    }
    return true;
}
