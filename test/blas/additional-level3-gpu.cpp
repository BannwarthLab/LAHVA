#include "../common.h"
#include "../utils.hpp"

using namespace lahva::gpu;

template<typename T>
T get_threshold_gpu() {
    return T(1.0e-5);
}

template<>
double get_threshold_gpu<double>() {
    return 1.0e-10;
}

template<>
float get_threshold_gpu<float>() {
    return 1.0e-5;
}

// Expand a GPU BlockDiagMatrix to a dense GPU Matrix for reference computations
template<typename T>
Matrix<T> expand_block_diagonal_gpu(const BlockDiagMatrix<T>& bd_matrix) {
    Shape total_shape = bd_matrix.shape();
    int total_rows = total_shape.first;
    int total_cols = total_shape.second;

    Matrix<T> dense(Shape{(size_t)total_rows, (size_t)total_cols}, T(0.0));

    const auto& row_offsets = bd_matrix.get_row_offsets();
    const auto& col_offsets = bd_matrix.get_col_offsets();

    for (size_t i = 0; i < bd_matrix.num_blocks(); ++i) {
        const auto& block = bd_matrix.get_block(i);
        int m = block.shape().first;
        int n = block.shape().second;

        const T* block_data = block.data();
        T* dense_data = dense.data();
        int row_offset = row_offsets[i];
        int col_offset = col_offsets[i];

        for (int ii = 0; ii < m; ++ii) {
            for (int jj = 0; jj < n; ++jj) {
                int global_row = row_offset + ii;
                int global_col = col_offset + jj;
                dense_data[global_col * total_rows + global_row] = block_data[jj * m + ii];
            }
        }
    }

    return dense;
}

template<typename T>
int test_blockdiag_times_dense_square_gpu(CudaRuntime& cudart) {
    int stat = 0;
    T thr = get_threshold_gpu<T>();

    BlockDiagMatrix<T> A;

    {
        Matrix<T> block1(Shape{3, 3}, T(1.0));
        for (int i = 0; i < 9; ++i)
            block1.data()[i] = T(i + 1.0);
        A.add_block(block1);
    }

    {
        Matrix<T> block2(Shape{2, 2}, T(2.0));
        for (int i = 0; i < 4; ++i)
            block2.data()[i] = T(i + 1.0);
        A.add_block(block2);
    }

    Matrix<T> B(Shape{5, 5}, T(0.5));
    Matrix<T> C(Shape{5, 5}, T(0.0));
    MatrixMatrixProduct(cudart, "N", "N", T(1.0), A, B, T(0.0), C);
    C.copy2host(cudart);
    cudart.synchronize();

    Matrix<T> A_dense = expand_block_diagonal_gpu(A);
    Matrix<T> C_expected(Shape{5, 5}, T(0.0));
    MatrixMatrixProduct(cudart, "N", "N", T(1.0), A_dense, B, T(0.0), C_expected);
    C_expected.copy2host(cudart);
    cudart.synchronize();

    if (!check(C.data(), C_expected.data(), thr, 25,
               "GPU BlockDiagMatrix * Matrix result mismatch (square)")) {
        stat += 1;
    }

    return stat;
}

template<typename T>
int test_blockdiag_times_dense_gpu(CudaRuntime& cudart) {
    int stat = 0;
    T thr = get_threshold_gpu<T>();

    BlockDiagMatrix<T> A;

    {
        Matrix<T> block1(Shape{3, 3}, T(1.0));
        for (int i = 0; i < 9; ++i)
            block1.data()[i] = T(i + 1.0);
        A.add_block(block1);
    }

    {
        Matrix<T> block2(Shape{2, 2}, T(2.0));
        for (int i = 0; i < 4; ++i)
            block2.data()[i] = T(i + 1.0);
        A.add_block(block2);
    }

    Matrix<T> B(Shape{5, 4}, T(0.5));
    Matrix<T> C(Shape{5, 4}, T(0.0));
    MatrixMatrixProduct(cudart, "N", "N", T(1.0), A, B, T(0.0), C);
    C.copy2host(cudart);
    cudart.synchronize();

    Matrix<T> A_dense = expand_block_diagonal_gpu(A);
    Matrix<T> C_expected(Shape{5, 4}, T(0.0));
    MatrixMatrixProduct(cudart, "N", "N", T(1.0), A_dense, B, T(0.0), C_expected);
    C_expected.copy2host(cudart);
    cudart.synchronize();

    if (!check(C.data(), C_expected.data(), thr, 20,
               "GPU BlockDiagMatrix * Matrix result mismatch")) {
        stat += 1;
    }

    return stat;
}

template<typename T>
int test_blockdiag_times_dense_alpha_beta_gpu(CudaRuntime& cudart) {
    int stat = 0;
    T thr = get_threshold_gpu<T>();

    BlockDiagMatrix<T> A;

    {
        Matrix<T> block1(Shape{3, 3}, T(1.0));
        for (int i = 0; i < 9; ++i)
            block1.data()[i] = T(i + 1.0);
        A.add_block(block1);
    }

    {
        Matrix<T> block2(Shape{2, 2}, T(2.0));
        for (int i = 0; i < 4; ++i)
            block2.data()[i] = T(i + 1.0);
        A.add_block(block2);
    }

    Matrix<T> B(Shape{5, 4}, T(0.5));
    Matrix<T> A_dense = expand_block_diagonal_gpu(A);

    // alpha = 2.0, beta = 0.0
    {
        Matrix<T> C(Shape{5, 4}, T(0.0));
        MatrixMatrixProduct(cudart, "N", "N", T(2.0), A, B, T(0.0), C);
        C.copy2host(cudart);
        cudart.synchronize();

        Matrix<T> C_expected(Shape{5, 4}, T(0.0));
        MatrixMatrixProduct(cudart, "N", "N", T(2.0), A_dense, B, T(0.0), C_expected);
        C_expected.copy2host(cudart);
        cudart.synchronize();

        if (!check(C.data(), C_expected.data(), thr, 20,
                   "GPU BlockDiagMatrix * Matrix with alpha=2.0, beta=0.0")) {
            stat += 1;
        }
    }

    // alpha = 1.0, beta = 2.0
    {
        Matrix<T> C(Shape{5, 4}, T(1.0));
        MatrixMatrixProduct(cudart, "N", "N", T(1.0), A, B, T(2.0), C);
        C.copy2host(cudart);
        cudart.synchronize();

        Matrix<T> C_expected(Shape{5, 4}, T(1.0));
        MatrixMatrixProduct(cudart, "N", "N", T(1.0), A_dense, B, T(2.0), C_expected);
        C_expected.copy2host(cudart);
        cudart.synchronize();

        if (!check(C.data(), C_expected.data(), thr, 20,
                   "GPU BlockDiagMatrix * Matrix with alpha=1.0, beta=2.0")) {
            stat += 1;
        }
    }

    // alpha = 0.5, beta = 0.5
    {
        Matrix<T> C(Shape{5, 4}, T(2.0));
        MatrixMatrixProduct(cudart, "N", "N", T(0.5), A, B, T(0.5), C);
        C.copy2host(cudart);
        cudart.synchronize();

        Matrix<T> C_expected(Shape{5, 4}, T(2.0));
        MatrixMatrixProduct(cudart, "N", "N", T(0.5), A_dense, B, T(0.5), C_expected);
        C_expected.copy2host(cudart);
        cudart.synchronize();

        if (!check(C.data(), C_expected.data(), thr, 20,
                   "GPU BlockDiagMatrix * Matrix with alpha=0.5, beta=0.5")) {
            stat += 1;
        }
    }

    return stat;
}

template<typename T>
int test_dense_times_blockdiag_square_gpu(CudaRuntime& cudart) {
    int stat = 0;
    T thr = get_threshold_gpu<T>();

    BlockDiagMatrix<T> B;

    {
        Matrix<T> block1(Shape{3, 3}, T(1.0));
        for (int i = 0; i < 9; ++i)
            block1.data()[i] = T(i + 1.0);
        B.add_block(block1);
    }

    {
        Matrix<T> block2(Shape{2, 2}, T(2.0));
        for (int i = 0; i < 4; ++i)
            block2.data()[i] = T(i + 1.0);
        B.add_block(block2);
    }

    Matrix<T> A(Shape{5, 5}, T(0.5));
    Matrix<T> C(Shape{5, 5}, T(0.0));
    MatrixMatrixProduct(cudart, "N", "N", T(1.0), A, B, T(0.0), C);
    C.copy2host(cudart);
    cudart.synchronize();

    Matrix<T> B_dense = expand_block_diagonal_gpu(B);
    Matrix<T> C_expected(Shape{5, 5}, T(0.0));
    MatrixMatrixProduct(cudart, "N", "N", T(1.0), A, B_dense, T(0.0), C_expected);
    C_expected.copy2host(cudart);
    cudart.synchronize();

    if (!check(C.data(), C_expected.data(), thr, 25,
               "GPU Matrix * BlockDiagMatrix result mismatch (square)")) {
        stat += 1;
    }

    return stat;
}

template<typename T>
int test_dense_times_blockdiag_gpu(CudaRuntime& cudart) {
    int stat = 0;
    T thr = get_threshold_gpu<T>();

    BlockDiagMatrix<T> B;

    {
        Matrix<T> block1(Shape{3, 3}, T(1.0));
        for (int i = 0; i < 9; ++i)
            block1.data()[i] = T(i + 1.0);
        B.add_block(block1);
    }

    {
        Matrix<T> block2(Shape{2, 2}, T(2.0));
        for (int i = 0; i < 4; ++i)
            block2.data()[i] = T(i + 1.0);
        B.add_block(block2);
    }

    Matrix<T> A(Shape{4, 5}, T(0.5));
    Matrix<T> C(Shape{4, 5}, T(0.0));
    MatrixMatrixProduct(cudart, "N", "N", T(1.0), A, B, T(0.0), C);
    C.copy2host(cudart);
    cudart.synchronize();

    Matrix<T> B_dense = expand_block_diagonal_gpu(B);
    Matrix<T> C_expected(Shape{4, 5}, T(0.0));
    MatrixMatrixProduct(cudart, "N", "N", T(1.0), A, B_dense, T(0.0), C_expected);
    C_expected.copy2host(cudart);
    cudart.synchronize();

    if (!check(C.data(), C_expected.data(), thr, 20,
               "GPU Matrix * BlockDiagMatrix result mismatch")) {
        stat += 1;
    }

    return stat;
}

template<typename T>
int test_dense_times_blockdiag_alpha_beta_gpu(CudaRuntime& cudart) {
    int stat = 0;
    T thr = get_threshold_gpu<T>();

    BlockDiagMatrix<T> B;

    {
        Matrix<T> block1(Shape{3, 3}, T(1.0));
        for (int i = 0; i < 9; ++i)
            block1.data()[i] = T(i + 1.0);
        B.add_block(block1);
    }

    {
        Matrix<T> block2(Shape{2, 2}, T(2.0));
        for (int i = 0; i < 4; ++i)
            block2.data()[i] = T(i + 1.0);
        B.add_block(block2);
    }

    Matrix<T> A(Shape{4, 5}, T(0.5));
    Matrix<T> B_dense = expand_block_diagonal_gpu(B);

    // alpha = 2.0, beta = 0.0
    {
        Matrix<T> C(Shape{4, 5}, T(0.0));
        MatrixMatrixProduct(cudart, "N", "N", T(2.0), A, B, T(0.0), C);
        C.copy2host(cudart);
        cudart.synchronize();

        Matrix<T> C_expected(Shape{4, 5}, T(0.0));
        MatrixMatrixProduct(cudart, "N", "N", T(2.0), A, B_dense, T(0.0), C_expected);
        C_expected.copy2host(cudart);
        cudart.synchronize();

        if (!check(C.data(), C_expected.data(), thr, 20,
                   "GPU Matrix * BlockDiagMatrix with alpha=2.0, beta=0.0")) {
            stat += 1;
        }
    }

    // alpha = 1.0, beta = 2.0
    {
        Matrix<T> C(Shape{4, 5}, T(1.0));
        MatrixMatrixProduct(cudart, "N", "N", T(1.0), A, B, T(2.0), C);
        C.copy2host(cudart);
        cudart.synchronize();

        Matrix<T> C_expected(Shape{4, 5}, T(1.0));
        MatrixMatrixProduct(cudart, "N", "N", T(1.0), A, B_dense, T(2.0), C_expected);
        C_expected.copy2host(cudart);
        cudart.synchronize();

        if (!check(C.data(), C_expected.data(), thr, 20,
                   "GPU Matrix * BlockDiagMatrix with alpha=1.0, beta=2.0")) {
            stat += 1;
        }
    }

    // alpha = 0.5, beta = 0.5
    {
        Matrix<T> C(Shape{4, 5}, T(2.0));
        MatrixMatrixProduct(cudart, "N", "N", T(0.5), A, B, T(0.5), C);
        C.copy2host(cudart);
        cudart.synchronize();

        Matrix<T> C_expected(Shape{4, 5}, T(2.0));
        MatrixMatrixProduct(cudart, "N", "N", T(0.5), A, B_dense, T(0.5), C_expected);
        C_expected.copy2host(cudart);
        cudart.synchronize();

        if (!check(C.data(), C_expected.data(), thr, 20,
                   "GPU Matrix * BlockDiagMatrix with alpha=0.5, beta=0.5")) {
            stat += 1;
        }
    }

    return stat;
}

int main() {
    int stat = 0;
    CudaRuntime cudart;

    stat += test_blockdiag_times_dense_square_gpu<double>(cudart);
    stat += test_blockdiag_times_dense_gpu<double>(cudart);
    stat += test_blockdiag_times_dense_alpha_beta_gpu<double>(cudart);

    stat += test_blockdiag_times_dense_square_gpu<float>(cudart);
    stat += test_blockdiag_times_dense_gpu<float>(cudart);
    stat += test_blockdiag_times_dense_alpha_beta_gpu<float>(cudart);

    stat += test_dense_times_blockdiag_square_gpu<double>(cudart);
    stat += test_dense_times_blockdiag_gpu<double>(cudart);
    stat += test_dense_times_blockdiag_alpha_beta_gpu<double>(cudart);

    stat += test_dense_times_blockdiag_square_gpu<float>(cudart);
    stat += test_dense_times_blockdiag_gpu<float>(cudart);
    stat += test_dense_times_blockdiag_alpha_beta_gpu<float>(cudart);

    return stat;
}
