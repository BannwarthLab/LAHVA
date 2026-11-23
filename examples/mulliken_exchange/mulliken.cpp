#include "mulliken.hpp"

using namespace BLASwrap;
using namespace cpuBLAS;
#ifdef _CUDA
using namespace gpuBLAS;
#endif

template <typename T, typename blas_impl>
MullikenExchange<T, blas_impl>::MullikenExchange(std::shared_ptr<RunTime> runtime, std::shared_ptr<Timer> timer, int nao)
    : runtime_(runtime), timer_(timer), nao_(nao)
    {}

template <typename T, typename blas_impl>
typename MullikenExchange<T, blas_impl>::template Matrix<T>
MullikenExchange<T, blas_impl>::SQMBoxKFockSym(Matrix<T> S, Matrix<T> D, Matrix<T> gamma) {

    timer_->push("copy_to_device");
#ifdef _CUDA
    if constexpr (std::is_same<blas_impl, gpuBLAS::gpu_BLAS>::value) {

        S.copy2device(*runtime_);
        D.copy2device(*runtime_);
        gamma.copy2device(*runtime_);
        runtime_->synchronize();
    }
#endif
    timer_->pop();

    timer_->push("matrix_products_phase1");
    MatrixMatrixProduct(*runtime_, S, D, A_, (T)1.0, (T)0.0);
    MatrixMatrixProduct(*runtime_, A_, S, B_, (T)0.5, (T)0.0);
    timer_->pop();

    timer_->push("hadamard_phase1");
    HadamardProduct(*runtime_, gamma, B_);
    HadamardProduct(*runtime_, gamma, A_);
    timer_->pop();

    timer_->push("scale_and_hadamard");
    ScaleVector(*runtime_, (T)0.5, gamma);
    HadamardProduct(*runtime_, D, gamma);
    timer_->pop();

    timer_->push("matrix_products_phase2");
    MatrixMatrixProduct(*runtime_, S, gamma, A_, (T)1.0, (T)1.0);
    MatrixMatrixProduct(*runtime_, A_, S, B_, (T)1.0, (T)1.0);
    timer_->pop();

    timer_->push("finalize");
    B_.symmetrize(*runtime_);
    ScaleVector(*runtime_, (T)0.25, B_);
    timer_->pop();

    timer_->push("copy_to_host");
#ifdef _CUDA
    if constexpr (std::is_same<blas_impl, gpuBLAS::gpu_BLAS>::value) {
        B_.copy2host(*runtime_);
    }
#endif
    timer_->pop();

    return B_;
}

template <typename T, typename blas_impl>
MullikenExchange<T, blas_impl>::~MullikenExchange() {}

#ifdef _CUDA
template class MullikenExchange<float, gpuBLAS::gpu_BLAS>;
template class MullikenExchange<double, gpuBLAS::gpu_BLAS>;
#endif
template class MullikenExchange<float, cpuBLAS::cpu_BLAS>;
template class MullikenExchange<double, cpuBLAS::cpu_BLAS>;
