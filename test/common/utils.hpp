#pragma once
#include <float.h>

#include <numeric>
#include <algorithm>
#include <cmath>
#include <limits>
#include <complex>
#include <iostream>

template <typename T>
bool compareNumbers(T a, T b, int n)
{
    return std::abs(a - b) < std::pow(10.0, n) * std::max(
    std::numeric_limits<T>::epsilon() * std::max(std::abs(a), std::abs(b)),
    std::numeric_limits<T>::denorm_min()
);
}
template<typename T, typename U>
bool check(T actual, U expected, const char* msg)
{
    if (expected == actual) {
        return true;
    }
    std::cerr << "[Fatal] " << msg << ": expected " << expected << ", got " << actual << std::endl;
    return false;
}

template<typename T, typename U>
bool check(T actual, U expected, double tol, const char* msg)
{
    if (std::abs(expected - actual) < tol) {
        return true;
    }
    std::cerr << "[Fatal] " << msg << ": expected " << expected << ", got " << actual << std::endl;
    return false;
}
template<typename T>
bool check(std::complex<T> actual, std::complex<T> expected, std::complex<T> tol, const char* msg)
{
    if (std::abs(expected.real() - actual.real()) < tol.real() && std::abs(expected.imag() - actual.imag()) < tol.imag()) {
        return true;
    }
    std::cerr << "[Fatal] " << msg << ": expected " << expected << ", got " << actual << std::endl;
    return false;
}

bool check(double *actual, double *expected, double tol, int ndim, const char *msg);

bool check(float *actual, float *expected, float tol, int ndim, const char *msg);

bool check(const int *actual, const int *expected, int ndim, const char *msg);

double norm2(int n, double *vec);

double sum(int n, double *vec);

double sum(int n, float *vec);

template <typename TYPE>
constexpr TYPE getTypeMax()
{
    return 0;
}
template <>
constexpr float getTypeMax<float>()
{
    return FLT_MAX;
}
template <>
constexpr double getTypeMax<double>()
{
    return DBL_MAX;
}

// getTypeMin
template <typename TYPE>
constexpr TYPE getTypeMin()
{
    return 0;
}
template <>
constexpr float getTypeMin<float>()
{
    return FLT_MIN;
}
template <>
constexpr double getTypeMin<double>()
{
    return DBL_MIN;
}
