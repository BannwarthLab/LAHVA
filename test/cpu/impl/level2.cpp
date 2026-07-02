#include "test_common.h"

#define M 10
#define N 5 

using namespace lahva::cpu;

const float thr = 5.0e-7;
template<typename T>
T get_complex_thr(){
    return T(5.0e-7, 5.0e-7);
}
float vf[9] = {1.0, 4.0, 5.0, 0.0, 2.0, 6.0, 0.0, 0.0, 3.0};
double vd[9] = {1.0, 4.0, 5.0, 0.0, 2.0, 6.0, 0.0, 0.0, 3.0};
float *pf = vf;
double *pd = vd;
float vftri[6] = {1.0, 4.0, 5.0, 2.0, 6.0, 3.0};
double vdtri[6] = {1.0, 4.0, 5.0, 2.0, 6.0, 3.0};
float *pft = vftri;
double *pdt = vdtri;
Vector<float> pvf({1.0, 2.0, 3.0}) ;

template<typename T> T* get_test_data();
template<> double* get_test_data<double>() { return pd; }
template<> float* get_test_data<float>() { return pf; }

template<typename T>
int test_gemv_zero_v_cpp(){
    int stat_ = 0;
    double thr = get_tolerance<T>();
    Shape s(M,N);
    Matrix<T> A(s, 1.0);
    Vector<T> x(N, 0.0);
    Vector<T> y(M, 1.0);

    MatrixVectorProduct("N", 1.0, A, x, 1, 0.0, y, 1);

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    y = Vector<T>(M, 1.0);

    MatrixVectorProduct("N", 1.0, A, x, 1, 1.0, y, 1);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    MatrixVectorProduct(A, x, y, "N", 1.0, 1.0);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    A = Matrix<T>(Shape(N,M), 1.0);

    MatrixVectorProduct(A, x, y, "T"); //Check transpose

    sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    return stat_;
}
template<typename T>
int test_complex_gemv_zero_v_cpp(){
    int stat_ = 0;
    Shape s(M,N);
    Matrix<T> A(s, 1.0);
    Vector<T> x(N, 0.0);
    Vector<T> y(M, 1.0);
    MatrixVectorProduct("N", 1.0, A, x, 1, 0.0, y, 1);
    auto sum_ = y.sum();
    T thrc = get_complex_thr<T>();
    if (!check(sum_, T(0.0, 0.0), thrc, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1; 

    y = Vector<T>(M, T{0.0,1.0});
    std::cout << y.sum() << std::endl;
    
    MatrixVectorProduct("N", 1.0, A, x, 1, T{0.0, -1.0}, y, 1);

    sum_ = y.sum();
    
    if (!check(sum_, T(M*1.0, 0.0), thrc, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;
    y = Vector<T>(M, T{0.0,1.0});
    MatrixVectorProduct(A, x, y, "N", 1.0, T{0.0, -1.0});

    sum_ = y.sum();
    std::cout << y.sum() << std::endl;
    if (!check(sum_, T(M*1.0,0.0), thrc, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    A = Matrix<T>(Shape(N,M), T(0.0, 1.0));
    x = Vector<T>(N, 1.0);
    MatrixVectorProduct(A, x, y, "C", T(0.0,1.0)); //Check complex conjugate

    sum_ = y.sum();

    if (!check(sum_, T(N*M*1.0, 0.0), thrc, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    return stat_;
}


template<typename T>
int test_gemv_v_cpp(){
    int stat_ = 0;
    double thr = get_tolerance<T>();
    Shape s(3,3);
    Matrix<T> A(s, get_test_data<T>(), false);
    Vector<T> x({1.0, 2.0, 3.0});
    Vector<T> y(3, 0.0);

    MatrixVectorProduct("N", 1.0, A, x, 1, 0.0, y, 1);

    Vector<T> vres({1.0, 8.0, 26.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    y = Vector<T>(3, 1.0);

    MatrixVectorProduct("N", 1.0, A, x, 1, 1.0, y, 1);

    vres = Vector<T>({2.0, 9.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    MatrixVectorProduct(A, x, y, "N", 2.0, 0.0);

    vres = Vector<T>({2.0, 16.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

template<typename T>
int test_symv_zero_v_cpp(){
    int stat_ = 0;
    double thr = get_tolerance<T>();
    Shape s(M,M);
    Matrix<T> A(s, 1.0);
    Vector<T> x(M, 0.0);
    Vector<T> y(M, 1.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 0.0, y, 1);

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    y = Vector<T>(M, 1.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 1.0, y, 1);
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(A, x, y, 1.0, 1.0);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    return stat_;
}

template<typename T>
int test_symv_v_cpp(){
    int stat_ = 0;
    double thr = get_tolerance<T>();
    Shape s(3,3);
    Matrix<T> A(s, get_test_data<T>(), false);
    Vector<T> x({1.0, 2.0, 3.0});
    Vector<T> y(3, 0.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 0.0, y, 1);

    Vector<T> vres({24.0, 26.0, 26.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    y = Vector<T>(3, 1.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 1.0, y, 1);

    vres = Vector<T>({25.0, 27.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(A, x, y, 2.0, 0.0);

    vres = Vector<T>({48.0, 52.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

template<typename T>
int test_symv_zero_lowtri_cpp(){
    int stat_ = 0;
    double thr = get_tolerance<T>();
    Shape s(M,M);
    LowTriMatrix<T> A(s, 1.0);
    Vector<T> x(M, 0.0);
    Vector<T> y(M, 1.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 0.0, y, 1);

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, thr, "Error when using Matrix Multiplication with a zero vector. LowTri1")) stat_ += 1;

    y = Vector<T>(M, 1.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 1.0, y, 1);
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector. LowTri2")) stat_ += 1;

    SymMatrixVectorProduct(A, x, y, 1.0, 1.0);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector. LowTri3")) stat_ += 1;

    return stat_;
}

template<typename T>
int test_symv_lowtri_cpp(){
    int stat_ = 0;
    double thr = get_tolerance<T>();
    T* vdtri_ = new T[6] {1.0, 4.0, 5.0, 2.0, 6.0, 3.0};
    LowTriMatrix<T> A(3, vdtri_, std::is_same_v<T, float>);
    Vector<T> x({1.0, 2.0, 3.0});
    Vector<T> y(3, 0.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 0.0, y, 1);

    Vector<T> vres({24.0, 26.0, 26.0});
    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;
    y = Vector<T>(3, 1.0);

    SymMatrixVectorProduct(1.0, A, x, 1, 1.0, y, 1);

    vres = Vector<T>({25.0, 27.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(A, x, y, 2.0, 0.0);

    vres = Vector<T>({48.0, 52.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

template<typename T>
int test_tpmv_cpp(){
    int stat_ = 0;
    double thr = get_tolerance<T>();
    T* vdtri_ = new T[6] {1.0, 4.0, 5.0, 2.0, 6.0, 3.0};
    LowTriMatrix<T> A(3, vdtri_);
    Vector<T> x({1.0, 2.0, 3.0});

    LowTriMatrixVectorProduct("N", CblasNonUnit, A, x, 1);

    Vector<T> vres({1.0, 8.0, 26.0});

    if (!check(x.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    x = Vector<T>({1.0, 2.0, 3.0});
    LowTriMatrixVectorProduct("N", CblasNonUnit, A, x, 1);

    if (!check(x.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    x = Vector<T>({1.0, 2.0, 3.0});
    LowTriMatrixVectorProduct("T", CblasNonUnit, A, x, 1);

    vres = Vector<T>({24.0, 22.0, 9.0});

    if (!check(x.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}



template<typename T>
int test_gemv_zero_v_c(){
    int stat_ = 0;
    double thr = get_tolerance<T>();
    Shape s(M,N);
    Matrix<T> A(s, 1.0);
    Vector<T> x(N, 0.0);
    Vector<T> y(M, 1.0);

    MatrixVectorProduct("N", M, N, 1.0, A.data(), x.data(), 1, 0.0, y.data(), 1);

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    y = Vector<T>(M, 1.0);

    MatrixVectorProduct("N", M, N, 1.0, A.data(), x.data(), 1, 1.0, y.data(), 1);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    MatrixVectorProduct(M, N, A.data(), x.data(), y.data(), "N", 1.0, 1.0);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    A = Matrix<T>(Shape(N,M), 1.0);

    MatrixVectorProduct(N, M, A.data(), x.data(), y.data(), "T"); //Check transpose

    sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    return stat_;
}

template<typename T>
int test_gemv_v_c(){
    int stat_ = 0;
    double thr = get_tolerance<T>();
    Shape s(3,3);
    Matrix<T> A(s, get_test_data<T>(), false);
    Vector<T> x({1.0, 2.0, 3.0});
    Vector<T> y(3, 0.0);

    MatrixVectorProduct("N", 3, 3, 1.0, A.data(), x.data(), 1, 0.0, y.data(), 1);

    Vector<T> vres({1.0, 8.0, 26.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;
    return stat_;

    y = Vector<T>(M, 1.0);

    MatrixVectorProduct("N", 3, 3, 1.0, A.data(), x.data(), 1, 1.0, y.data(), 1);

    vres = Vector<T>({2.0, 9.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    MatrixVectorProduct(3, 3, A.data(), x.data(), y.data(), "N", 2.0, 0.0);

    vres = Vector<T>({2.0, 16.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

template<typename T>
int test_symv_zero_v_c(){
    int stat_ = 0;
    double thr = get_tolerance<T>();
    Shape s(M,M);
    Matrix<T> A(s, 1.0);
    Vector<T> x(M, 0.0);
    Vector<T> y(M, 1.0);

    SymMatrixVectorProduct(M, (T)1.0, A.data(), x.data(), 1, (T)0.0, y.data(), 1);

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    y = Vector<T>(M, 1.0);

    SymMatrixVectorProduct(M, 1.0, A.data(), x.data(), 1, 1.0, y.data(), 1);
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(M, A.data(), x.data(), y.data(), 1.0, 1.0);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    return stat_;
}

template<typename T>
int test_symv_v_c(){
    int stat_ = 0;
    double thr = get_tolerance<T>();
    Shape s(3,3);
    Matrix<T> A(s, get_test_data<T>(), false);
    Vector<T> x({1.0, 2.0, 3.0});
    Vector<T> y(3, 0.0);

    SymMatrixVectorProduct(3, 1.0, A.data(), x.data(), 1, 0.0, y.data(), 1);

    Vector<T> vres({24.0, 26.0, 26.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;
    return stat_;

    y = Vector<T>(M, 1.0);

    SymMatrixVectorProduct(3, 1.0, A.data(), x.data(), 1, 1.0, y.data(), 1);

    vres = Vector<T>({25.0, 27.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(3, A.data(), x.data(), y.data(), 2.0, 0.0);

    vres = Vector<T>({48.0, 52.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

template <typename T>
int test_ger(){
    int stat_ = 0;
    Vector<T> x({1.0, 2.0, 3.0});
    Vector<T> y({2.0, 3.0});
    Matrix<T> A(Shape(3, 2), 0.0);

    OuterVectorProduct(x, y, A);

    Vector<T> vres({2.0, 4.0, 6.0, 3.0, 6.0, 9.0});

    if (!check(A.data(), vres.data(), thr, 6, "Error when computing outer product (double).")) stat_ += 1;

    // Test with alpha = 2.0
    A = Matrix<T>(Shape(3, 2), 0.0);
    OuterVectorProduct(x, y, A, 1, 1, 2.0);

    vres = Vector<T>({4.0, 8.0, 12.0, 6.0, 12.0, 18.0});

    if (!check(A.data(), vres.data(), thr, 6, "Error when computing outer product (double) with alpha=2.0.")) stat_ += 1;


    try {
        Matrix<T> A(Shape(4,2),0.0);
        OuterVectorProduct(x, y, A);
        stat_ += 1; // Should not reach here
        std::cerr << "Error: No exception thrown for dimension mismatch in outer product test." << std::endl;
    }
    catch (std::invalid_argument& e) {
        // Expected exception caught
    }
    catch (const std::exception& e) {
        stat_ += 1; // Unexpected exception type
        std::cerr << "Error: Unexpected exception type caught in outer product dimension mismatch test: " << e.what() << std::endl;
    }

    return stat_;
}

// ============================================================================
// BlockDiagMatrix GEMV Tests (Templated)
// ============================================================================

template<typename T>
int test_blockdiag_gemv_simple() {
    int failures = 0;
    T tol = get_tolerance<T>();

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 3, 2, 4}));
    blocks.push_back(Matrix<T>(Shape{2, 2}, {5, 7, 6, 8}));

    BlockDiagMatrix<T> A(blocks);

    Vector<T> x(4);
    x[0] = static_cast<T>(1);
    x[1] = static_cast<T>(2);
    x[2] = static_cast<T>(3);
    x[3] = static_cast<T>(4);

    Vector<T> y(4, static_cast<T>(0));

    MatrixVectorProduct(static_cast<T>(1), A, x, static_cast<T>(0), y);

    if (!check(y[0], static_cast<T>(5), tol, "y[0] should be 5")) failures += 1;
    if (!check(y[1], static_cast<T>(11), tol, "y[1] should be 11")) failures += 1;
    if (!check(y[2], static_cast<T>(39), tol, "y[2] should be 39")) failures += 1;
    if (!check(y[3], static_cast<T>(53), tol, "y[3] should be 53")) failures += 1;

    return failures;
}

template<typename T>
int test_blockdiag_gemv_with_beta() {
    int failures = 0;
    T tol = get_tolerance<T>();

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 2, 3, 4}));

    BlockDiagMatrix<T> A(blocks);

    Vector<T> x(2, static_cast<T>(1));
    Vector<T> y(2, static_cast<T>(2));

    MatrixVectorProduct(static_cast<T>(1), A, x, static_cast<T>(2), y);

    if (!check(y[0], static_cast<T>(8), tol, "y[0] should be 8")) failures += 1;
    if (!check(y[1], static_cast<T>(10), tol, "y[1] should be 10")) failures += 1;

    return failures;
}

template<typename T>
int test_blockdiag_gemv_varying_blocks() {
    int failures = 0;
    T tol = get_tolerance<T>();

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{1, 1}, {2}));
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 2, 3, 4}));
    blocks.push_back(Matrix<T>(Shape{1, 1}, {3}));

    BlockDiagMatrix<T> A(blocks);

    Vector<T> x(4);
    x[0] = static_cast<T>(1);
    x[1] = static_cast<T>(2);
    x[2] = static_cast<T>(3);
    x[3] = static_cast<T>(4);

    Vector<T> y(4, static_cast<T>(0));

    MatrixVectorProduct(static_cast<T>(1), A, x, static_cast<T>(0), y);

    if (!check(y[0], static_cast<T>(2), tol, "y[0] should be 2")) failures += 1;
    if (!check(y[1], static_cast<T>(11), tol, "y[1] should be 11")) failures += 1;
    if (!check(y[2], static_cast<T>(16), tol, "y[2] should be 16")) failures += 1;
    if (!check(y[3], static_cast<T>(12), tol, "y[3] should be 12")) failures += 1;

    return failures;
}

template<typename T>
int test_blockdiag_gemv_transpose() {
    int failures = 0;
    T tol = get_tolerance<T>();

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 3, 2, 4}));

    BlockDiagMatrix<T> A(blocks);

    Vector<T> x(2);
    x[0] = static_cast<T>(1);
    x[1] = static_cast<T>(2);

    Vector<T> y(2, static_cast<T>(0));

    MatrixVectorProduct(A, x, y, "T");

    if (!check(y[0], static_cast<T>(7), tol, "y[0] should be 7")) failures += 1;
    if (!check(y[1], static_cast<T>(10), tol, "y[1] should be 10")) failures += 1;

    return failures;
}

template<typename T>
int test_blockdiag_gemv_transpose_with_alpha_beta() {
    int failures = 0;
    T tol = get_tolerance<T>();

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {2, 6, 4, 8}));

    BlockDiagMatrix<T> A(blocks);

    Vector<T> x(2);
    x[0] = static_cast<T>(1);
    x[1] = static_cast<T>(2);

    Vector<T> y(2, static_cast<T>(1));

    MatrixVectorProduct(A, x, y, "T", static_cast<T>(0.5), static_cast<T>(2));

    if (!check(y[0], static_cast<T>(9), tol, "y[0] should be 9")) failures += 1;
    if (!check(y[1], static_cast<T>(12), tol, "y[1] should be 12")) failures += 1;

    return failures;
}

template<typename T>
int test_blockdiag_gemv_no_transpose() {
    int failures = 0;
    T tol = get_tolerance<T>();

    std::vector<Matrix<T>> blocks;
    blocks.push_back(Matrix<T>(Shape{2, 2}, {1, 3, 2, 4}));

    BlockDiagMatrix<T> A(blocks);

    Vector<T> x(2);
    x[0] = static_cast<T>(1);
    x[1] = static_cast<T>(2);

    Vector<T> y(2, static_cast<T>(0));

    MatrixVectorProduct(A, x, y, "N");

    if (!check(y[0], static_cast<T>(5), tol, "y[0] should be 5")) failures += 1;
    if (!check(y[1], static_cast<T>(11), tol, "y[1] should be 11")) failures += 1;

    return failures;
}


int main(){
    int stat = 0;
    stat += test_gemv_zero_v_cpp<double>();
    stat += test_gemv_zero_v_cpp<float>();
    stat += test_complex_gemv_zero_v_cpp<complex_double>();
    stat += test_complex_gemv_zero_v_cpp<complex_float>();
    stat += test_gemv_v_cpp<double>();
    stat += test_gemv_v_cpp<float>();
    stat += test_symv_zero_v_cpp<double>();
    stat += test_symv_zero_v_cpp<float>();
    stat += test_symv_v_cpp<double>();
    stat += test_symv_v_cpp<float>();
    stat += test_symv_zero_lowtri_cpp<double>();
    stat += test_symv_zero_lowtri_cpp<float>();
    stat += test_symv_lowtri_cpp<double>();
    stat += test_symv_lowtri_cpp<float>();
    stat += test_tpmv_cpp<double>();
    stat += test_tpmv_cpp<float>();
    stat += test_gemv_zero_v_c<double>();
    stat += test_gemv_zero_v_c<float>();
    stat += test_gemv_v_c<double>();
    stat += test_gemv_v_c<float>();
    stat += test_symv_zero_v_c<double>();
    stat += test_symv_zero_v_c<float>();
    stat += test_symv_v_c<double>();
    stat += test_symv_v_c<float>();
    stat += test_ger<double>();
    stat += test_ger<float>();

    // BlockDiagMatrix GEMV tests
    stat += test_blockdiag_gemv_simple<double>();
    stat += test_blockdiag_gemv_simple<float>();
    stat += test_blockdiag_gemv_with_beta<double>();
    stat += test_blockdiag_gemv_with_beta<float>();
    stat += test_blockdiag_gemv_varying_blocks<double>();
    stat += test_blockdiag_gemv_varying_blocks<float>();
    stat += test_blockdiag_gemv_transpose<double>();
    stat += test_blockdiag_gemv_transpose<float>();
    stat += test_blockdiag_gemv_transpose_with_alpha_beta<double>();
    stat += test_blockdiag_gemv_transpose_with_alpha_beta<float>();
    stat += test_blockdiag_gemv_no_transpose<double>();
    stat += test_blockdiag_gemv_no_transpose<float>();

    return stat;
};
