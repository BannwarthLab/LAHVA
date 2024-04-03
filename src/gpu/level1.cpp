#include <cstddef>
#include <cuda_runtime.h>
#include <cublas_v2.h>

extern "C" {
    void sHadamard(void *vecinout, void *vecin, size_t ndim2);
    void dHadamard(void *vecinout, void *vecin, size_t ndim2);
}

namespace tcgmtensor {
    static void wrap_sHadamard(cuda_smatrix vecinout, cuda_smatrix vecin, cuda_runtime cudart) {
        size_t vec1 = vecinout.cols * vecinout.rows;
        size_t vec2 = vecin.cols * vecin.rows;
        if (vec1 != vec2) return;
        cudaDeviceSynchronize();
        sHadamard(vecinout.ptr, vecin.ptr, vec1);
        cudaDeviceSynchronize();
    }

    static void wrap_dHadamard(cuda_dmatrix vecinout, cuda_dmatrix vecin, cuda_runtime cudart) {
        size_t vec1 = vecinout.cols * vecinout.rows;
        size_t vec2 = vecin.cols * vecin.rows;
        if (vec1 != vec2) return;
        cudaDeviceSynchronize();
        dHadamard(vecinout.ptr, vecin.ptr, vec1);
        cudaDeviceSynchronize();
    }

    static void wrap_saxpy(float a, cuda_smatrix x, cuda_smatrix y, cuda_runtime cudart, int incx = 1, int incy = 1) {
        int n = (x.cols * x.rows) / abs(incx);
        cudaDeviceSynchronize();
        cublasSaxpy(cudart.handle, n, a, static_cast<float*>(x.ptr), incx, static_cast<float*>(y.ptr), incy);
        cudaDeviceSynchronize();
    }

    static void wrap_daxpy(double a, cuda_dmatrix x, cuda_dmatrix y, cuda_runtime cudart, int incx = 1, int incy = 1) {
        int n = (x.cols * x.rows) / abs(incx);
        cudaDeviceSynchronize();
        cublasDaxpy(cudart.handle, n, a, static_cast<double*>(x.ptr), incx, static_cast<double*>(y.ptr), incy);
        cudaDeviceSynchronize();
    }
};