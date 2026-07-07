#include "test_common.h"
using namespace lahva::gpu;
#define M 10
#define N 5

template<typename T>
using MLow = LowTriMatrix<T, StdAllocator<T>, CudaDeviceAllocator<T>>;
template<typename T>
T get_complex_thr(){
    return T(5.0e-7, 5.0e-7);
}
float vf[9] = {1.0, 4.0, 5.0, 0.0, 2.0, 6.0, 0.0, 0.0, 3.0};
double vd[9] = {1.0, 4.0, 5.0, 0.0, 2.0, 6.0, 0.0, 0.0, 3.0};
float *pf = vf;
double *pd = vd;

template<typename T> T* get_test_data();
template<> double* get_test_data<double>() { return pd; }
template<> float* get_test_data<float>() { return pf; }

template <typename T>
int test_gemv_zero_v_cpp(CudaRuntime& cudart){
    int stat_ = 0;
    double thr = get_tolerance<T>();
    Shape s(M,N);
    Matrix<T> A(s, (T)1.0);
    Vector<T> x(N, (T)0.0);
    Vector<T> y(M, (T)1.0);

    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, 1, (T)0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    y = Vector<T>(M, (T)1.0);

    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, 1, (T)1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    MatrixVectorProduct(cudart, A, x, y, "N", (T)1.0, (T)1.0);
    y.copy2host(cudart);
    A.copy2host(cudart);
    cudart.synchronize();

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    A = Matrix<T>(Shape(N,M), (T)1.0);

    MatrixVectorProduct(cudart, A, x, y, "T"); //Check transpose
    y.copy2host(cudart);
    cudart.synchronize();
    sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    return stat_;
};

template<typename T>
int test_complex_gemv_zero_v_cpp(CudaRuntime& cudart){
    int stat_ = 0;
    Shape s(M,N);
    Matrix<T> A(s, T(1.0,0.0));
    Vector<T> x(N, T(0.0, 0.0));
    Vector<T> y(M, T(1.0, 0.0));
    MatrixVectorProduct("N", 1.0, A, x, 1, 0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    auto sum_ = y.sum();
    T thrc = get_complex_thr<T>();
    if (!check(sum_, T(0.0, 0.0), thrc, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    y = Vector<T>(M, T{0.0,1.0});

    MatrixVectorProduct("N", 1.0, A, x, 1, T{0.0, -1.0}, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    sum_ = y.sum();

    if (!check(sum_, T(M*1.0, 0.0), thrc, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;
    y = Vector<T>(M, T{0.0,1.0});
    MatrixVectorProduct(A, x, y, "N", 1.0, T{0.0, -1.0});
    y.copy2host(cudart);
    cudart.synchronize();
    sum_ = y.sum();

    if (!check(sum_, T(M*1.0,0.0), thrc, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    A = Matrix<T>(Shape(N,M), T(0.0, 1.0));
    x = Vector<T>(N, T(1.0, 0.0));
    A.updateGPUvalues(cudart);
    x.updateGPUvalues(cudart);
    MatrixVectorProduct(A, x, y, "C", T(0.0,1.0)); //Check complex conjugate
    y.copy2host(cudart);
    sum_ = y.sum();

    if (!check(sum_, T(N*M*1.0, 0.0), thrc, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    return stat_;
}

template <typename T>
int test_symv_zero_v_cpp(CudaRuntime& cudart){
    int stat_ = 0;
    double thr = get_tolerance<T>();
    Shape s(M,M);
    Matrix<T> A(s, (T)1.0);
    Vector<T> x(M, (T)0.0);
    Vector<T> y(M, (T)1.0);

    SymMatrixVectorProduct(cudart, (T)1.0, A, x, 1, (T)0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    y = Vector<T>(M, (T)1.0);

    SymMatrixVectorProduct(cudart, (T)1.0, A, x, 1, (T)1.0, y, 1);
    y.copy2host(cudart);
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(cudart, A, x, y, (T)1.0, (T)1.0);
    y.copy2host(cudart);
    A.copy2host(cudart);

    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    return stat_;
}

template <typename T>
int test_symv_v_cpp(CudaRuntime& cudart){
    int stat_ = 0;
    double thr = get_tolerance<T>();
    Shape s(3,3);
    Matrix<T> A(s, get_test_data<T>(), false);
    Vector<T> x({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> y(3, (T)0.0);

    SymMatrixVectorProduct(cudart, (T)1.0, A, x, 1, (T)0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    Vector<T> vres({(T)24.0, (T)26.0, (T)26.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    y = Vector<T>(3, (T)1.0);

    SymMatrixVectorProduct(cudart, (T)1.0, A, x, 1, (T)1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<T>({(T)25.0, (T)27.0, (T)27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(cudart, A, x, y, (T)2.0, (T)0.0);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<T>({(T)48.0, (T)52.0, (T)52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

template <typename T>
int test_gemv_v_cpp(CudaRuntime& cudart){
    int stat_ = 0;
    double thr = get_tolerance<T>();
    Shape s(3,3);
    Matrix<T> A(s, get_test_data<T>(), false);
    Vector<T> x({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> y(3, (T)0.0);

    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, 1, (T)0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    Vector<T> vres({(T)1.0, (T)8.0, (T)26.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    y = Vector<T>(3, (T)1.0);

    MatrixVectorProduct(cudart, "N", (T)1.0, A, x, 1, (T)1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<T>({(T)2.0, (T)9.0, (T)27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    MatrixVectorProduct(cudart, A, x, y, "N", (T)2.0, (T)0.0);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<T>({(T)2.0, (T)16.0, (T)52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

template <typename T>
int test_spmv_v_cpp(CudaRuntime& cudart){
    int stat_ = 0;
    double thr = get_tolerance<T>();
    T* vdtri_ = new T[6] {(T)1.0, (T)4.0, (T)5.0, (T)2.0, (T)6.0, (T)3.0};
    MLow<T> A(3, vdtri_);
    Vector<T> x({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> y(3, (T)0.0);

    SymMatrixVectorProduct(cudart, (T)1.0, A, x, 1, (T)0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    Vector<T> vres({(T)24.0, (T)26.0, (T)26.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    y = Vector<T>(3, (T)1.0);

    SymMatrixVectorProduct(cudart, (T)1.0, A, x, 1, (T)1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<T>({(T)25.0, (T)27.0, (T)27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(cudart, A, x, y, (T)2.0, (T)0.0);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<T>({(T)48.0, (T)52.0, (T)52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

template <typename T>
int test_outer_product(CudaRuntime& cudart){
    int stat_ = 0;
    double thr = get_tolerance<T>();
    Vector<T> x({(T)1.0, (T)2.0, (T)3.0});
    Vector<T> y({(T)2.0, (T)3.0});
    Matrix<T> A(Shape(3, 2), (T)0.0);

    OuterVectorProduct(cudart, x, y, A);
    A.copy2host(cudart);
    cudart.synchronize();

    Vector<T> vres({(T)2.0, (T)4.0, (T)6.0, (T)3.0, (T)6.0, (T)9.0});

    if (!check(A.data(), vres.data(), thr, 6, "Error when computing outer product (double).")) stat_ += 1;

    // Test with alpha = 2.0
    A = Matrix<T>(Shape(3, 2), (T)0.0);
    OuterVectorProduct(cudart, x, y, A, 1, 1, (T)2.0);
    A.copy2host(cudart);
    cudart.synchronize();

    vres = Vector<T>({(T)4.0, (T)8.0, (T)12.0, (T)6.0, (T)12.0, (T)18.0});

    if (!check(A.data(), vres.data(), thr, 6, "Error when computing outer product (double) with alpha=2.0.")) stat_ += 1;

    try {
        Matrix<T> A(Shape(4,2),0.0);
        OuterVectorProduct(cudart, x, y, A);
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

template <typename T>
int test_gpu_blockdiag_gemv_simple() {
    int failures = 0;

    CudaRuntime cudart;

    // Create block diagonal matrix with two 2x2 blocks
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 3, 2, 4}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {5, 7, 6, 8}));

    BlockDiagMatrix<T> A(blocks);

    // Create input vector
    Vector<T, CudaHostAllocator<T>> x(4);
    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;
    x[3] = 4.0;

    // Create output vector
    Vector<T, CudaHostAllocator<T>> y(4, 0.0);

    // Compute y = A * x
    MatrixVectorProduct(cudart, "N", static_cast<T>(1.0), A, x, static_cast<T>(0.0), y);

    // Check results
    if (!check(y[0], static_cast<T>(5.0), get_tolerance<T>(), "y[0] should be 5")) {
        failures += 1;
    }
    if (!check(y[1], static_cast<T>(11.0), get_tolerance<T>(), "y[1] should be 11")) {
        failures += 1;
    }
    if (!check(y[2], static_cast<T>(39.0), get_tolerance<T>(), "y[2] should be 39")) {
        failures += 1;
    }
    if (!check(y[3], static_cast<T>(53.0), get_tolerance<T>(), "y[3] should be 53")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_blockdiag_gemv_with_beta() {
    int failures = 0;

    CudaRuntime cudart;

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));

    BlockDiagMatrix<T> A(blocks);

    Vector<T, CudaHostAllocator<T>> x(2, static_cast<T>(1.0));
    Vector<T, CudaHostAllocator<T>> y(2, static_cast<T>(2.0));

    MatrixVectorProduct(cudart, "N", static_cast<T>(1.0), A, x, static_cast<T>(2.0), y);

    if (!check(y[0], static_cast<T>(8.0), get_tolerance<T>(), "y[0] should be 8")) {
        failures += 1;
    }
    if (!check(y[1], static_cast<T>(10.0), get_tolerance<T>(), "y[1] should be 10")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_blockdiag_gemv_varying_blocks() {
    int failures = 0;

    CudaRuntime cudart;

    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{1, 1}, {2}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{1, 1}, {3}));

    BlockDiagMatrix<T> A(blocks);

    Vector<T, CudaHostAllocator<T>> x(4);
    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;
    x[3] = 4.0;

    Vector<T, CudaHostAllocator<T>> y(4, static_cast<T>(0.0));

    MatrixVectorProduct(cudart, "N", static_cast<T>(1.0), A, x, static_cast<T>(0.0), y);

    if (!check(y[0], static_cast<T>(2.0), get_tolerance<T>(), "y[0] should be 2")) {
        failures += 1;
    }
    if (!check(y[1], static_cast<T>(11.0), get_tolerance<T>(), "y[1] should be 11")) {
        failures += 1;
    }
    if (!check(y[2], static_cast<T>(16.0), get_tolerance<T>(), "y[2] should be 16")) {
        failures += 1;
    }
    if (!check(y[3], static_cast<T>(12.0), get_tolerance<T>(), "y[3] should be 12")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_blockdiag_gemv_transpose() {
    int failures = 0;

    CudaRuntime cudart;

    // Create block diagonal matrix with one 2x2 block
    // Matrix in column-major: {1, 3, 2, 4} represents:
    // [1 2]
    // [3 4]
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 3, 2, 4}));

    BlockDiagMatrix<T> A(blocks);

    // Create input vector
    Vector<T, CudaHostAllocator<T>> x(2);
    x[0] = 1.0;
    x[1] = 2.0;

    // Create output vector
    Vector<T, CudaHostAllocator<T>> y(2, static_cast<T>(0.0));

    // Compute y = A^T * x
    // A^T = [1 3]
    //       [2 4]
    // A^T * x = [1*1 + 3*2, 2*1 + 4*2] = [7, 10]
    MatrixVectorProduct(cudart, "T", static_cast<T>(1.0), A, x, static_cast<T>(0.0), y);

    if (!check(y[0], static_cast<T>(7.0), get_tolerance<T>(), "y[0] should be 7")) {
        failures += 1;
    }
    if (!check(y[1], static_cast<T>(10.0), get_tolerance<T>(), "y[1] should be 10")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_blockdiag_gemv_transpose_with_beta() {
    int failures = 0;

    CudaRuntime cudart;

    // Create block diagonal matrix with one 2x2 block
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 2, 3, 4}));

    BlockDiagMatrix<T> A(blocks);

    Vector<T, CudaHostAllocator<T>> x(2, static_cast<T>(1.0));
    Vector<T, CudaHostAllocator<T>> y(2, static_cast<T>(2.0));

    // Compute y = A^T * x + 2*y
    // A^T = [1 2]
    //       [3 4]
    // A^T * x = [1*1 + 2*1, 3*1 + 4*1] = [3, 7]
    // result = [3, 7] + 2*[2, 2] = [7, 11]
    MatrixVectorProduct(cudart, "T", static_cast<T>(1.0), A, x, static_cast<T>(2.0), y);

    if (!check(y[0], static_cast<T>(7.0), get_tolerance<T>(), "y[0] should be 7")) {
        failures += 1;
    }
    if (!check(y[1], static_cast<T>(11.0), get_tolerance<T>(), "y[1] should be 11")) {
        failures += 1;
    }

    return failures;
}

template <typename T>
int test_gpu_blockdiag_gemv_transpose_multiple_blocks() {
    int failures = 0;

    CudaRuntime cudart;

    // Create block diagonal matrix with two 2x2 blocks
    std::vector<Matrix<T, CudaHostAllocator<T>>> blocks;
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {1, 3, 2, 4}));
    blocks.push_back(Matrix<T, CudaHostAllocator<T>>(Shape{2, 2}, {5, 7, 6, 8}));

    BlockDiagMatrix<T> A(blocks);

    // Create input vector
    Vector<T, CudaHostAllocator<T>> x(4);
    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;
    x[3] = 4.0;

    // Create output vector
    Vector<T, CudaHostAllocator<T>> y(4, static_cast<T>(0.0));

    // Compute y = A^T * x
    // Block 1: A1^T = [1 3], A1^T * [1, 2] = [7, 10]
    //                 [2 4]
    // Block 2: A2^T = [5 7], A2^T * [3, 4] = [43, 50]
    //                 [6 8]
    MatrixVectorProduct(cudart, "T", static_cast<T>(1.0), A, x, static_cast<T>(0.0), y);

    if (!check(y[0], static_cast<T>(7.0), get_tolerance<T>(), "y[0] should be 7")) {
        failures += 1;
    }
    if (!check(y[1], static_cast<T>(10.0), get_tolerance<T>(), "y[1] should be 10")) {
        failures += 1;
    }
    if (!check(y[2], static_cast<T>(43.0), get_tolerance<T>(), "y[2] should be 43")) {
        failures += 1;
    }
    if (!check(y[3], static_cast<T>(50.0), get_tolerance<T>(), "y[3] should be 50")) {
        failures += 1;
    }

    return failures;
}

int main(){
    int stat = 0;
    CudaRuntime cudart;
    cudart.print_cuda_version();
    std::cout << "1st Test" << std::endl;
    stat += test_gemv_zero_v_cpp<double>(cudart);
    stat += test_gemv_zero_v_cpp<float>(cudart);
    std::cout << "2nd Test" << std::endl;
    stat += test_symv_zero_v_cpp<double>(cudart);
    stat += test_symv_zero_v_cpp<float>(cudart);
    std::cout << "3rd Test" << std::endl;
    stat += test_gemv_v_cpp<double>(cudart);
    stat += test_gemv_v_cpp<float>(cudart);
    std::cout << "4th Test" << std::endl;
    stat += test_symv_v_cpp<double>(cudart);
    stat += test_symv_v_cpp<float>(cudart);
    std::cout << "5th Test" << std::endl;
    stat += test_spmv_v_cpp<double>(cudart);
    stat += test_spmv_v_cpp<float>(cudart);
    printf("6th Test");
    stat += test_complex_gemv_zero_v_cpp<complex_double>(cudart);
    stat += test_complex_gemv_zero_v_cpp<complex_float>(cudart);
    printf("7th Test");
    stat += test_outer_product<double>(cudart);
    stat += test_outer_product<float>(cudart);
    std::cout << "8th Test - BlockDiag GEMV (double precision)" << std::endl;
    stat += test_gpu_blockdiag_gemv_simple<double>();
    stat += test_gpu_blockdiag_gemv_with_beta<double>();
    stat += test_gpu_blockdiag_gemv_varying_blocks<double>();
    stat += test_gpu_blockdiag_gemv_transpose<double>();
    stat += test_gpu_blockdiag_gemv_transpose_with_beta<double>();
    stat += test_gpu_blockdiag_gemv_transpose_multiple_blocks<double>();
    std::cout << "9th Test - BlockDiag GEMV (single precision)" << std::endl;
    stat += test_gpu_blockdiag_gemv_simple<float>();
    stat += test_gpu_blockdiag_gemv_with_beta<float>();
    stat += test_gpu_blockdiag_gemv_varying_blocks<float>();
    stat += test_gpu_blockdiag_gemv_transpose<float>();
    stat += test_gpu_blockdiag_gemv_transpose_with_beta<float>();
    stat += test_gpu_blockdiag_gemv_transpose_multiple_blocks<float>();
    return stat;
};