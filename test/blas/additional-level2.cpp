#include "../common.h"
#include "../utils.hpp"
#include <algorithm>

using namespace lahva::cpu;


template<typename T>
std::initializer_list<T>& getam(){
    std::initializer_list<T> am({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    return am;
};

template<typename T>
std::initializer_list<T>& getbm(){
    std::initializer_list<T> bm({7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    return bm;
};

template<typename T>
std::initializer_list<T>& getres()
{
    std::initializer_list<T> rm({8.0, 10.0, 12.0, 14.0, 16.0, 18.0});
    return rm;
};


template<typename T>
int test_add_matrices()
{ 
    Matrix<T> A(Shape(2,3), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Matrix<T> B(Shape(2,3), {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    Matrix<T> C(A.shape(), 0.0);

    AddMatrices(A, B, C, (T) 1.0, (T) 1.0);

    Matrix<T> res(A.shape(),{8.0, 10.0, 12.0, 14.0, 16.0, 18.0});
    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices")))
    {
        std::cout << "Test failed: AddMatrices" << std::endl;
        return 1;
    }

    AddMatrices("N", "N", (T)1.0, A, (T)1.0, B, C);

    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices")))
    {
        std::cout << "Test failed: AddMatrices" << std::endl;
        return 1;
    }

    return 0;
};

template<typename T>
int test_add_matrices_transposed()
{
    Matrix<T> A(Shape(2,3), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Matrix<T> B(Shape(3,2), {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    Matrix<T> C(A.shape(), 0.0);
    A.print();
    B.print();
    AddMatrices(A, B, C, (T)1.0, (T)1.0, "N", "T");

    Matrix<T> res(A.shape(), {8.0, 12.0, 11.0, 15.0, 14.0, 18.0});

    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices: B transposed")))
    {
        std::cout << "Test1 failed: AddMatrices" << std::endl;
        return 1;
    }

    AddMatrices("N", "T", (T) 1.0, A, (T) 1.0, B, C);

    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrcies B transposed")))
    {
        std::cout << "Test2 failed: AddMatrices" << std::endl;
        return 1;
    }    

    B = Matrix<T>(Shape(2,3), {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    A = Matrix<T>(Shape(3,2), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    AddMatrices(A, B, C, (T) 1.0, (T) 1.0, "T", "N");

    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices A transposed")))
    {
        std::cout << "Test3 failed: AddMatrices" << std::endl;
        return 1;
    }

    AddMatrices("T", "N", (T) 1.0, A, (T) 1.0, B, C);

    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices  A transposed")))
    {
        std::cout << "Test4 failed: AddMatrices" << std::endl;
        return 1;
    }

    A = Matrix<T>(Shape(2,3), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    res = Matrix<T>(Shape(3,2), {8.0, 12.0, 16.0, 10.0, 14.0, 18.0});
    C = Matrix<T>(Shape(3,2), 0.0);

    AddMatrices(A, B, C, (T) 1.0, (T) 1.0, "T", "T");

    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices A and B transposed")))
    {
        std::cout << "Test5 failed: AddMatrices" << std::endl;
        return 1;
    }

    AddMatrices("T", "T", (T) 1.0, A, (T) 1.0, B, C);

    if (!(check(C.data(), res.data(), 1e-6, res.size(), "AddMatrices A and B transposed")))
    {
        std::cout << "Test6 failed: AddMatrices" << std::endl;
        return 1;
    }

    return 0;
};


template<typename T>
Matrix<T> expand_block_diagonal(const BlockDiagMatrix<T>& bd_matrix) {
    Shape total_shape = bd_matrix.shape();
    int total_rows = total_shape.first;
    int total_cols = total_shape.second;
    
    Matrix<T> dense(Shape{total_rows, total_cols}, T(0.0));
    
    const auto& row_offsets = bd_matrix.get_row_offsets();
    const auto& col_offsets = bd_matrix.get_col_offsets();
    
    for (size_t i = 0; i < bd_matrix.num_blocks(); ++i) {
        const Matrix<T>& block = bd_matrix.get_block(i);
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
int test_blockdiag_matrix_vector_product() {
    int stat = 0;
    T thr = (std::is_same<T, double>::value) ? T(1.0e-13) : T(1.0e-5);
    
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
    
    Vector<T> x(5);
    for (int i = 0; i < 5; ++i) {
        x.data()[i] = T(i + 1.0);
    }
    
    Vector<T> y(5, T(0.0));
    
    MatrixVectorProduct("N", T(1.0), A, x, 1, T(0.0), y, 1);
    
    // Compute expected result
    Matrix<T> A_dense = expand_block_diagonal(A);
    Vector<T> y_expected(5, T(0.0));
    MatrixVectorProduct(A_dense, x, y_expected);
    
    if (!check(y.data(), y_expected.data(), thr, 5, 
               "BlockDiagMatrix * Vector result mismatch")) {
        stat += 1;
    }
    
    return stat;
}

template<typename T>
int test_blockdiag_matrix_vector_product_transpose() {
    int stat = 0;
    T thr = (std::is_same<T, double>::value) ? T(1.0e-13) : T(1.0e-5);
    
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
    
    Vector<T> x(5);
    for (int i = 0; i < 5; ++i) {
        x.data()[i] = T(i + 1.0);
    }
    
    Vector<T> y(5, T(0.0));
    
    MatrixVectorProduct("T", T(1.0), A, x, 1, T(0.0), y, 1);
    
    // Compute expected result
    Matrix<T> A_dense = expand_block_diagonal(A);
    Vector<T> y_expected(5, T(0.0));
    MatrixVectorProduct(A_dense, x, y_expected, "T");
    
    if (!check(y.data(), y_expected.data(), thr, 5, 
               "BlockDiagMatrix^T * Vector result mismatch")) {
        stat += 1;
    }
    
    return stat;
}

template<typename T>
int test_blockdiag_matrix_vector_product_alpha_beta() {
    int stat = 0;
    T thr = (std::is_same<T, double>::value) ? T(1.0e-13) : T(1.0e-5);
    
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
    
    Vector<T> x(5);
    for (int i = 0; i < 5; ++i) {
        x.data()[i] = T(i + 1.0);
    }
    
    Vector<T> y(5, T(2.0));
    
    MatrixVectorProduct("N", T(2.0), A, x, 1, T(0.5), y, 1);
    
    // Compute expected result
    Matrix<T> A_dense = expand_block_diagonal(A);
    Vector<T> y_expected(5, T(2.0));
    MatrixVectorProduct(A_dense, x, y_expected, "N", T(2.0), T(0.5));
    
    if (!check(y.data(), y_expected.data(), thr, 5, 
               "BlockDiagMatrix * Vector with alpha/beta result mismatch")) {
        stat += 1;
    }
    
    return stat;
}


int main(){
    int exit = 0;
    exit += test_add_matrices<double>();
    exit += test_add_matrices_transposed<double>();
    exit += test_add_matrices<float>();
    exit += test_add_matrices_transposed<float>();
    
    exit += test_blockdiag_matrix_vector_product<double>();
    exit += test_blockdiag_matrix_vector_product<float>();
    exit += test_blockdiag_matrix_vector_product_transpose<double>();
    exit += test_blockdiag_matrix_vector_product_transpose<float>();
    exit += test_blockdiag_matrix_vector_product_alpha_beta<double>();
    exit += test_blockdiag_matrix_vector_product_alpha_beta<float>();

    return exit;
};