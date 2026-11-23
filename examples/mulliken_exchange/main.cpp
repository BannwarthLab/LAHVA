#include "mulliken.hpp"

template <typename MatrixType>
void create_dummy_matrices(MatrixType& S, MatrixType& D, MatrixType& gamma) {
    const int nao = S.shape().first;

    // Create dummy overlap matrix S (symmetric, positive definite)
    for (int i = 0; i < nao; ++i) {
        for (int j = 0; j < nao; ++j) {
            if (i == j) {
                S.data()[i * nao + j] = 1.0;
            } else {
                S.data()[i * nao + j] = 0.2 * std::exp(-0.1 * std::abs(i - j));
            }
        }
    }

    // Create dummy density matrix D (symmetric)
    for (int i = 0; i < nao; ++i) {
        for (int j = 0; j < nao; ++j) {
            if (i == j) {
                D.data()[i * nao + j] = 0.5;
            } else {
                D.data()[i * nao + j] = 0.1 * std::exp(-0.1 * std::abs(i - j));
            }
        }
    }

    // Create dummy gamma matrix
    for (int i = 0; i < nao; ++i) {
        for (int j = 0; j < nao; ++j) {
            gamma.data()[i * nao + j] = 0.5 + 0.1 * (i + j);
        }
    }

}

template <typename T, typename blas_impl>
int run_exchange(int nao) {
    using TF = TensorFactory<blas_impl>;
    using Matrix = typename TF::template Matrix<T>;

    std::shared_ptr<typename TensorFactory<blas_impl>::RunTime> runtime = std::make_shared<typename TensorFactory<blas_impl>::RunTime>();
    std::shared_ptr<typename TensorFactory<blas_impl>::Timer> timer = std::make_shared<typename TensorFactory<blas_impl>::Timer>();

    MullikenExchange<T, blas_impl> mulliken(runtime, timer, nao);

    Matrix S = Matrix(Shape(nao, nao));
    Matrix D = Matrix(Shape(nao, nao));
    Matrix gamma = Matrix(Shape(nao, nao));

    create_dummy_matrices(S, D, gamma);
    mulliken.SQMBoxKFockSym(S, D, gamma);

    std::cout << "Mulliken test completed successfully" << std::endl;
    std::cout << "Timing Results:" << std::endl;
    std::cout << timer->print_entries() << std::endl;

    return 0;
}

int main(int argc, char* argv[]) {
    int nao = 10000;

    if (argc > 1) {
        nao = std::atoi(argv[1]);
    }

    std::cout << "Tests using " << nao << " basis functions" << std::endl;

    std::cout << "Mulliken: double precision, CPU" << std::endl;
    run_exchange<double, cpuBLAS::cpu_BLAS>(nao);
    std::cout << "Mulliken: single precision, CPU" << std::endl;
    run_exchange<float, cpuBLAS::cpu_BLAS>(nao);

#ifdef _CUDA
    std::cout << "Mulliken: double precision, GPU" << std::endl;
    run_exchange<double, gpuBLAS::gpu_BLAS>(nao);
    std::cout << "Mulliken: single precision, GPU" << std::endl;
    run_exchange<float, gpuBLAS::gpu_BLAS>(nao);
#endif

    return 0;
}

#ifdef _CUDA
template class MullikenExchange<float, gpuBLAS::gpu_BLAS>;
template class MullikenExchange<double, gpuBLAS::gpu_BLAS>;
#endif
template class MullikenExchange<float, cpuBLAS::cpu_BLAS>;
template class MullikenExchange<double, cpuBLAS::cpu_BLAS>;
