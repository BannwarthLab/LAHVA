#include "utils.hpp"
#include <math.h>
#include <stdio.h>

bool
check(int actual, int expected, const char* msg)
{
    if (expected == actual) {
        return true;
    }
    fprintf(stderr, "[Fatal] %s: expected %d, got %d\n", msg, expected, actual);
    return false;
}

bool
check(double actual, double expected, double tol, const char* msg)
{
    if (fabs(expected - actual) < tol) {
        return true;
    }
    fprintf(stderr, "[Fatal] %s: expected %3.7f, got %3.7f\n", msg, expected, actual);
    return false;
}

bool
check(double* actual, double* expected, double tol, int ndim, const char* msg)
{
    for (int i = 0; i != ndim; i++) {
        if (!check(actual[i], expected[i], tol, msg)) {
            return false;
        }
    }
    return true;
}

bool
check(float* actual, float* expected, float tol, int ndim, const char* msg)
{
    for (int i = 0; i != ndim; i++) {
        if (!check(actual[i], expected[i], tol, msg)) {
            return false;
        }
    }
    return true;
}

bool
check(const int* actual, const int* expected, int ndim, const char* msg)
{
    for (int i = 0; i != ndim; i++) {
        if (!check(actual[i], expected[i], msg)) {
            return false;
        }
    }
    return true;
}

double
norm2(int n, double* vec)
{
    double norm = 0.0;
    int i;
    for (i = 0; i != n; i++)
        norm += vec[i] * vec[i];
    return sqrt(norm);
}

double
sum(int n, double* vec)
{
    double val = 0.0;
    int i;
    for (i = 0; i != n; i++)
        val += vec[i];
    return val;
}

double
sum(int n, float* vec)
{
    double val = 0.0;
    int i;
    for (i = 0; i != n; i++)
        val += vec[i];
    return val;
}