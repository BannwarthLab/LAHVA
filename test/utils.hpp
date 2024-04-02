#pragma once

bool check(int actual, int expected, const char* msg);

bool check(double actual, double expected, double tol, const char* msg);

bool check(double* actual, double* expected, double tol, int ndim, const char* msg);

bool check(const int* actual, const int* expected, int ndim, const char* msg);

double norm2(int n, double* vec);

double sum(int n, double* vec);

double sum(int n, float* vec);