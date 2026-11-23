#pragma once
#include <numeric>
#include <random>
#include <cmath>

double norm2(int n, double* vec)
{
    double norm = 0.0;
    int i;
    for (i = 0; i != n; i++)
        norm += vec[i] * vec[i];
    return sqrt(norm);
}

template <typename T>
inline T sum(int n, T *vec)
{
    T val = 0.0;
    int i;
    for (i = 0; i != n; i++)
        val += vec[i];
    return val;
}

// Fill tensor with random values using normal distribution
template <typename Tensor>
void fill_with_rd_values(Tensor& tensor, double mean = 0.0, double stddev = 1.0e+3)
{
    std::random_device rd;
    std::minstd_rand eng(rd());
    std::normal_distribution<> distr(mean, stddev);

    for (size_t i = 0; i < tensor.size(); i++)
        tensor.data()[i] = distr(eng);
}