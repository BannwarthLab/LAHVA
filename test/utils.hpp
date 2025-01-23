#pragma once
#include <float.h>

#include <numeric>
#include <algorithm>
#include <cmath>
#include <limits>

template <typename T>
bool compareNumbers(T a, T b, int n)
{
    return abs(a - b) < std::pow(10.0, n) * std::max(
    std::numeric_limits<T>::epsilon() * std::max(abs(a), abs(b)),
    std::numeric_limits<T>::denorm_min()
);
}

bool check(int actual, int expected, const char *msg);

bool check(double actual, double expected, double tol, const char *msg);

bool check(double *actual, double *expected, double tol, int ndim, const char *msg);

bool check(float *actual, float *expected, float tol, int ndim, const char *msg);

bool check(const int *actual, const int *expected, int ndim, const char *msg);

double norm2(int n, double *vec);

double sum(int n, double *vec);

double sum(int n, float *vec);

template <typename TYPE>
constexpr TYPE getTypeMax()
{
    // fprintf(OUTPUT, "OzBLAS error: TYPE is not specified in getTypeMax.\n");
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

// -------------------------------------
// getTypeMin
// -------------------------------------
template <typename TYPE>
constexpr TYPE getTypeMin()
{
    // fprintf(OUTPUT, "OzBLAS error: TYPE is not specified in getTypeMin.\n");
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