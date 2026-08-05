#pragma once
#include "lahva_wrap.hpp"

using namespace BLASwrap;
using namespace cpuBLAS;
#ifdef _CUDA
using namespace gpuBLAS;
#endif

template <typename T, typename blas_impl>
class MullikenExchange : public TensorFactory<blas_impl>
{
protected:
    template <typename U>
    using Vector = typename TensorFactory<blas_impl>::template Vector<U>;
    template <typename U>
    using Matrix = typename TensorFactory<blas_impl>::template Matrix<U>;
    using RunTime = typename TensorFactory<blas_impl>::RunTime;
    using Timer = typename TensorFactory<blas_impl>::Timer;

    std::shared_ptr<RunTime> runtime_;
    std::shared_ptr<Timer> timer_;

private:
    const int nao_;
    Matrix<T> A_ = Matrix<T>(Shape(nao_, nao_));
    Matrix<T> B_ = Matrix<T>(Shape(nao_, nao_));

public:
    MullikenExchange(std::shared_ptr<RunTime> runtime, std::shared_ptr<Timer> timer, int nao);
    ~MullikenExchange();
    
    Matrix<T> SQMBoxKFockSym(Matrix<T> S, Matrix<T> D, Matrix<T> gamma);
};
