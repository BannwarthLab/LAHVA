#include "../common.h"
#include "../utils.hpp"

using namespace lahva::cpu;

template<typename T>
T get_threshold() {
    return T(1.0e-5);
}

template<>
double get_threshold<double>() {
    return 1.0e-13;
}

template<>
float get_threshold<float>() {
    return 1.0e-5;
}

// Helper function to convert BlockDiagMatrix to a dense Matrix
template<typename T>
Matrix<T> expand_block_diagonal(const BlockDiagMatrix<T>& bd_matrix) {
    // Get total dimensions
    Shape total_shape = bd_matrix.shape();
    int total_rows = total_shape.first;
    int total_cols = total_shape.second;
    
    // Create dense matrix initialized to zero
    Matrix<T> dense(Shape{total_rows, total_cols}, 0.0);
    
    
    // Fill in the block diagonal blocks
    const auto& row_offsets = bd_matrix.get_row_offsets();
    const auto& col_offsets = bd_matrix.get_col_offsets();
    
    for (size_t i = 0; i < bd_matrix.num_blocks(); ++i) {
        const Matrix<T>& block = bd_matrix.get_block(i);
        int m = block.shape().first;
        int n = block.shape().second;
        
        // Copy block data into dense matrix
        const T* block_data = block.data();
        T* dense_data = dense.data();
        int row_offset = row_offsets[i];
        int col_offset = col_offsets[i];
        
        for (int ii = 0; ii < m; ++ii) {
            for (int jj = 0; jj < n; ++jj) {
                int global_row = row_offset + ii;
                int global_col = col_offset + jj;
                dense_data[global_col * total_cols + global_row] = block_data[jj * m + ii];
            }
        }
    }
    
    return dense;
}

template<typename T>
int test_blockdiag_times_dense_square() {
    int stat = 0;
    T thr = get_threshold<T>();
    
    BlockDiagMatrix<T> A;
    
    {
        Matrix<T> block1(Shape{3, 3}, T(1.0));
        for (int i = 0; i < 9; ++i) {
            block1.data()[i] = T(i + 1.0);
        }
        A.add_block(block1);
    }
    
    {
        Matrix<T> block2(Shape{2, 2}, T(2.0));
        for (int i = 0; i < 4; ++i) {
            block2.data()[i] = T(i + 1.0);
        }
        A.add_block(block2);
    }
    
    Matrix<T> B(Shape{5, 5}, T(0.5));
    Matrix<T> C(Shape{5, 5}, T(0.0));
    MatrixMatrixProduct("N", "N", T(1.0), A, B, T(0.0), C);
    
    Matrix<T> A_dense = expand_block_diagonal(A);
    Matrix<T> C_expected(Shape{5, 5}, T(0.0));
    MatrixMatrixProduct("N", "N", T(1.0), A_dense, B, T(0.0), C_expected);
    
    if (!check(C.data(), C_expected.data(), thr, 25, 
               "BlockDiagMatrix * Matrix result mismatch (square)")) {
        stat += 1;
    }
    
    return stat;
}

template<typename T>
int test_blockdiag_times_dense() {
    int stat = 0;
    T thr = get_threshold<T>();
    
    BlockDiagMatrix<T> A;
    
    {
        Matrix<T> block1(Shape{3, 3}, T(1.0));
        for (int i = 0; i < 9; ++i) {
            block1.data()[i] = T(i + 1.0);
        }
        A.add_block(block1);
    }
    
    {
        Matrix<T> block2(Shape{2, 2}, T(2.0));
        for (int i = 0; i < 4; ++i) {
            block2.data()[i] = T(i + 1.0);
        }
        A.add_block(block2);
    }
    
    Matrix<T> B(Shape{5, 4}, T(0.5));
    Matrix<T> C(Shape{5, 4}, T(0.0));
    MatrixMatrixProduct("N", "N", T(1.0), A, B, T(0.0), C);
    
    Matrix<T> A_dense = expand_block_diagonal(A);
    Matrix<T> C_expected(Shape{5, 4}, T(0.0));
    MatrixMatrixProduct("N", "N", T(1.0), A_dense, B, T(0.0), C_expected);

    if (!check(C.data(), C_expected.data(), thr, 20, 
               "BlockDiagMatrix * Matrix result mismatch")) {
        stat += 1;
    }
    
    return stat;
}

template<typename T>
int test_blockdiag_times_dense_alpha_beta() {
    int stat = 0;
    T thr = get_threshold<T>();
    
    BlockDiagMatrix<T> A;
    
    {
        Matrix<T> block1(Shape{3, 3}, T(1.0));
        for (int i = 0; i < 9; ++i) {
            block1.data()[i] = T(i + 1.0);
        }
        A.add_block(block1);
    }
    
    {
        Matrix<T> block2(Shape{2, 2}, T(2.0));
        for (int i = 0; i < 4; ++i) {
            block2.data()[i] = T(i + 1.0);
        }
        A.add_block(block2);
    }
    
    Matrix<T> B(Shape{5, 4}, T(0.5));
    
    // alpha = 2.0, beta = 0.0
    {
        Matrix<T> C(Shape{5, 4}, T(0.0));
        MatrixMatrixProduct("N", "N", T(2.0), A, B, T(0.0), C);
        
        Matrix<T> A_dense = expand_block_diagonal(A);
        Matrix<T> C_expected(Shape{5, 4}, T(0.0));
        MatrixMatrixProduct("N", "N", T(2.0), A_dense, B, T(0.0), C_expected);
        
        if (!check(C.data(), C_expected.data(), thr, 20,
                   "BlockDiagMatrix * Matrix with alpha=2.0, beta=0.0")) {
            stat += 1;
        }
    }
    
    // alpha = 1.0, beta = 2.0
    {
        Matrix<T> C(Shape{5, 4}, T(1.0));  // C initialized to 1.0
        MatrixMatrixProduct("N", "N", T(1.0), A, B, T(2.0), C);
        
        Matrix<T> A_dense = expand_block_diagonal(A);
        Matrix<T> C_expected(Shape{5, 4}, T(1.0));
        MatrixMatrixProduct("N", "N", T(1.0), A_dense, B, T(2.0), C_expected);
        
        if (!check(C.data(), C_expected.data(), thr, 20,
                   "BlockDiagMatrix * Matrix with alpha=1.0, beta=2.0")) {
            stat += 1;
        }
    }
    
    // alpha = 0.5, beta = 0.5
    {
        Matrix<T> C(Shape{5, 4}, T(2.0));  // C initialized to 2.0
        MatrixMatrixProduct("N", "N", T(0.5), A, B, T(0.5), C);
        
        Matrix<T> A_dense = expand_block_diagonal(A);
        Matrix<T> C_expected(Shape{5, 4}, T(2.0));
        MatrixMatrixProduct("N", "N", T(0.5), A_dense, B, T(0.5), C_expected);
        
        if (!check(C.data(), C_expected.data(), thr, 20,
                   "BlockDiagMatrix * Matrix with alpha=0.5, beta=0.5")) {
            stat += 1;
        }
    }
    
    return stat;
}

template<typename T>
int test_dense_times_blockdiag_square() {
    int stat = 0;
    T thr = get_threshold<T>();

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
    MatrixMatrixProduct("N", "N", T(1.0), A, B, T(0.0), C);

    Matrix<T> B_dense = expand_block_diagonal(B);
    Matrix<T> C_expected(Shape{5, 5}, T(0.0));
    MatrixMatrixProduct("N", "N", T(1.0), A, B_dense, T(0.0), C_expected);

    if (!check(C.data(), C_expected.data(), thr, 25,
               "Matrix * BlockDiagMatrix result mismatch (square)")) {
        stat += 1;
    }

    return stat;
}

template<typename T>
int test_dense_times_blockdiag() {
    int stat = 0;
    T thr = get_threshold<T>();

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
    MatrixMatrixProduct("N", "N", T(1.0), A, B, T(0.0), C);

    Matrix<T> B_dense = expand_block_diagonal(B);
    Matrix<T> C_expected(Shape{4, 5}, T(0.0));
    MatrixMatrixProduct("N", "N", T(1.0), A, B_dense, T(0.0), C_expected);

    if (!check(C.data(), C_expected.data(), thr, 20,
               "Matrix * BlockDiagMatrix result mismatch")) {
        stat += 1;
    }

    return stat;
}

template<typename T>
int test_dense_times_blockdiag_alpha_beta() {
    int stat = 0;
    T thr = get_threshold<T>();

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
    Matrix<T> B_dense = expand_block_diagonal(B);

    // alpha = 2.0, beta = 0.0
    {
        Matrix<T> C(Shape{4, 5}, T(0.0));
        MatrixMatrixProduct("N", "N", T(2.0), A, B, T(0.0), C);

        Matrix<T> C_expected(Shape{4, 5}, T(0.0));
        MatrixMatrixProduct("N", "N", T(2.0), A, B_dense, T(0.0), C_expected);

        if (!check(C.data(), C_expected.data(), thr, 20,
                   "Matrix * BlockDiagMatrix with alpha=2.0, beta=0.0")) {
            stat += 1;
        }
    }

    // alpha = 1.0, beta = 2.0
    {
        Matrix<T> C(Shape{4, 5}, T(1.0));
        MatrixMatrixProduct("N", "N", T(1.0), A, B, T(2.0), C);

        Matrix<T> C_expected(Shape{4, 5}, T(1.0));
        MatrixMatrixProduct("N", "N", T(1.0), A, B_dense, T(2.0), C_expected);

        if (!check(C.data(), C_expected.data(), thr, 20,
                   "Matrix * BlockDiagMatrix with alpha=1.0, beta=2.0")) {
            stat += 1;
        }
    }

    // alpha = 0.5, beta = 0.5
    {
        Matrix<T> C(Shape{4, 5}, T(2.0));
        MatrixMatrixProduct("N", "N", T(0.5), A, B, T(0.5), C);

        Matrix<T> C_expected(Shape{4, 5}, T(2.0));
        MatrixMatrixProduct("N", "N", T(0.5), A, B_dense, T(0.5), C_expected);

        if (!check(C.data(), C_expected.data(), thr, 20,
                   "Matrix * BlockDiagMatrix with alpha=0.5, beta=0.5")) {
            stat += 1;
        }
    }

    return stat;
}

int main(){
    int stat = 0;

    stat += test_blockdiag_times_dense_square<double>();
    stat += test_blockdiag_times_dense<double>();
    stat += test_blockdiag_times_dense_alpha_beta<double>();

    stat += test_blockdiag_times_dense_square<float>();
    stat += test_blockdiag_times_dense<float>();
    stat += test_blockdiag_times_dense_alpha_beta<float>();

    stat += test_dense_times_blockdiag_square<double>();
    stat += test_dense_times_blockdiag<double>();
    stat += test_dense_times_blockdiag_alpha_beta<double>();

    stat += test_dense_times_blockdiag_square<float>();
    stat += test_dense_times_blockdiag<float>();
    stat += test_dense_times_blockdiag_alpha_beta<float>();

    return stat;
}
