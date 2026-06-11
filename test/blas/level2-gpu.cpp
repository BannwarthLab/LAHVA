#include "../common.h"
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

const double thr2 = 5.0e-15;
const float thr = 5.0e-7;

int test_dgemv_zero_v_cpp(CudaRuntime& cudart){
    int stat_ = 0;
    Shape s(M,N);
    Matrix<double> A(s, 1.0);
    Vector<double> x(N, 0.0);
    Vector<double> y(M, 1.0);

    MatrixVectorProduct(cudart, "N", 1.0, A, x, 1, 0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1; 

    y = Vector<double>(M, 1.0);
    
    MatrixVectorProduct(cudart, "N", 1.0, A, x, 1, 1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    MatrixVectorProduct(cudart, A, x, y, "N", 1.0, 1.0);
    y.copy2host(cudart);
    A.copy2host(cudart);
    cudart.synchronize();  
    
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    A = Matrix<double>(Shape(N,M), 1.0);

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

int test_sgemv_zero_v_cpp(CudaRuntime& cudart){
    int stat_ = 0;
    Shape s(M,N);
    Matrix<float> A(s, 1.0);
    Vector<float> x(N, 0.0);
    Vector<float> y(M, 1.0);

    MatrixVectorProduct(cudart, "N", 1.0, A, x, 1, 0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1; 

    y = Vector<float>(M, 1.0);
    
    MatrixVectorProduct(cudart, "N", 1.0, A, x, 1, 1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    MatrixVectorProduct(cudart, A, x, y, "N", 1.0, 1.0);
    y.copy2host(cudart);
    cudart.synchronize();
    A.copy2host(cudart);
    cudart.synchronize();
    
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    A = Matrix<float>(Shape(N,M), 1.0);

    MatrixVectorProduct(cudart, A, x, y, "T"); //Check transpose
    y.copy2host(cudart);
    cudart.synchronize();
    sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    return stat_;
}

int test_dsymv_zero_v_cpp(CudaRuntime& cudart){
    int stat_ = 0;
    Shape s(M,M);
    Matrix<double> A(s, 1.0);
    Vector<double> x(M, 0.0);
    Vector<double> y(M, 1.0);

    SymMatrixVectorProduct(cudart, 1.0, A, x, 1, 0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1; 

    y = Vector<double>(M, 1.0);
    
    SymMatrixVectorProduct(cudart, 1.0, A, x, 1, 1.0, y, 1);
    y.copy2host(cudart);
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(cudart, A, x, y, 1.0, 1.0);
    y.copy2host(cudart);
    A.copy2host(cudart);  
    
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    return stat_;
}

int test_ssymv_zero_v_cpp(CudaRuntime& cudart){
    int stat_ = 0;
    Shape s(M,M);
    Matrix<float> A(s, 1.0);
    Vector<float> x(M, 0.0);
    Vector<float> y(M, 1.0);

    SymMatrixVectorProduct(cudart, 1.0, A, x, 1, 0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();

    auto sum_ = sum(M, y.data());

    if (!check(sum_, 0.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1; 

    y = Vector<float>(M, 1.0);
    
    SymMatrixVectorProduct(cudart, 1.0, A, x, 1, 1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(cudart, A, x, y, 1.0, 1.0);
    y.copy2host(cudart);
    A.copy2host(cudart);
    cudart.synchronize();  
    
    sum_ = sum(M, y.data());

    if (!check(sum_, M*1.0, thr, "Error when using Matrix Multiplication with a zero vector.")) stat_ += 1;

    return stat_;
}

int test_dsymv_v_cpp(CudaRuntime& cudart){
    int stat_ = 0;
    Shape s(3,3);
    Matrix<double> A(s, pd, false);
    Vector<double> x({1.0, 2.0, 3.0});
    Vector<double> y(3, 0.0);

    SymMatrixVectorProduct(cudart, 1.0, A, x, 1, 0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    Vector<double> vres({24.0, 26.0, 26.0}); 

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1; 
    
    y = Vector<double>(3, 1.0);
    
    SymMatrixVectorProduct(cudart, 1.0, A, x, 1, 1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<double>({25.0, 27.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(cudart, A, x, y, 2.0, 0.0);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<double>({48.0, 52.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

int test_ssymv_v_cpp(CudaRuntime& cudart){
    int stat_ = 0;
    Shape s(3,3);
    Matrix<float> A(s, pf, false);
    Vector<float> x({1.0, 2.0, 3.0});
    Vector<float> y(3, 0.0);

    SymMatrixVectorProduct(cudart, 1.0, A, x, 1, 0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    Vector<float> vres({24.0, 26.0, 26.0}); 

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1; 
    
    y = Vector<float>(3, 1.0);
    
    SymMatrixVectorProduct(cudart, 1.0, A, x, 1, 1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<float>({25.0, 27.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(cudart, A, x, y, 2.0, 0.0);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<float>({48.0, 52.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

int test_dgemv_v_cpp(CudaRuntime& cudart){
    int stat_ = 0;
    Shape s(3,3);
    Matrix<double> A(s, pd, false);
    Vector<double> x({1.0, 2.0, 3.0});
    Vector<double> y(3, 0.0);

    MatrixVectorProduct(cudart, "N", 1.0, A, x, 1, 0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    Vector<double> vres({1.0, 8.0, 26.0}); 

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1; 
    
    y = Vector<double>(3, 1.0);
    
    MatrixVectorProduct(cudart, "N", 1.0, A, x, 1, 1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<double>({2.0, 9.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    MatrixVectorProduct(cudart, A, x, y, "N", 2.0, 0.0);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<double>({2.0, 16.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

int test_sgemv_v_cpp(CudaRuntime& cudart){
    int stat_ = 0;
    Shape s(3,3);
    Matrix<float> A(s, pf, false);
    Vector<float> x({1.0, 2.0, 3.0});
    Vector<float> y(3, 0.0);

    MatrixVectorProduct(cudart, "N", 1.0, A, x, 1, 0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    Vector<float> vres({1.0, 8.0, 26.0}); 

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1; 
    
    y = Vector<float>(3, 1.0);
    
    MatrixVectorProduct(cudart, "N", 1.0, A, x, 1, 1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<float>({2.0, 9.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    MatrixVectorProduct(cudart, A, x, y, "N", 2.0, 0.0);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<float>({2.0, 16.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;
}

int test_dspmv_v_cpp(CudaRuntime& cudart){
    int stat_ = 0;
    double* vdtri_ = new double[6] {1.0, 4.0, 5.0, 2.0, 6.0, 3.0};
    MLow<double> A(3, vdtri_);
    Vector<double> x({1.0, 2.0, 3.0});
    Vector<double> y(3, 0.0);

    SymMatrixVectorProduct(cudart, 1.0, A, x, 1, 0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    Vector<double> vres({24.0, 26.0, 26.0}); 

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1; 
    
    y = Vector<double>(3, 1.0);
    
    SymMatrixVectorProduct(cudart, 1.0, A, x, 1, 1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<double>({25.0, 27.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(cudart, A, x, y, 2.0, 0.0);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<double>({48.0, 52.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;
    
    return stat_;
}

int test_sspmv_v_cpp(CudaRuntime& cudart){
    int stat_ = 0;
    float* vdtri_ = new float[6] {1.0, 4.0, 5.0, 2.0, 6.0, 3.0};
    MLow<float> A(3, vdtri_);
    Vector<float> x({1.0, 2.0, 3.0});
    Vector<float> y(3, 0.0);

    SymMatrixVectorProduct(cudart, 1.0, A, x, 1, 0.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    Vector<float> vres({24.0, 26.0, 26.0}); 

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1; 
    
    y = Vector<float>(3, 1.0);
    
    SymMatrixVectorProduct(cudart, 1.0, A, x, 1, 1.0, y, 1);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<float>({25.0, 27.0, 27.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    SymMatrixVectorProduct(cudart, A, x, y, 2.0, 0.0);
    y.copy2host(cudart);
    cudart.synchronize();
    vres = Vector<float>({48.0, 52.0, 52.0});

    if (!check(y.data(), vres.data(), thr, 3, "Error when using Matrix Multiplication with a non-zero vector.")) stat_ += 1;

    return stat_;

}

template <typename T>
int test_outer_product(CudaRuntime& cudart){
    int stat_ = 0;
    Vector<T> x({1.0, 2.0, 3.0});
    Vector<T> y({2.0, 3.0});
    Matrix<T> A(Shape(3, 2), 0.0);

    OuterVectorProduct(cudart, x, y, A);
    A.copy2host(cudart);
    cudart.synchronize();

    Vector<T> vres({2.0, 4.0, 6.0, 3.0, 6.0, 9.0});

    if (!check(A.data(), vres.data(), thr, 6, "Error when computing outer product (double).")) stat_ += 1;

    // Test with alpha = 2.0
    A = Matrix<T>(Shape(3, 2), 0.0);
    OuterVectorProduct(cudart, x, y, A, 1, 1, 2.0);
    A.copy2host(cudart);
    cudart.synchronize();

    vres = Vector<T>({4.0, 8.0, 12.0, 6.0, 12.0, 18.0});

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

int main(){
    int stat = 0;
    CudaRuntime cudart;
    cudart.print_cuda_version();
    std::cout << "1st Test" << std::endl;
    stat += test_dgemv_zero_v_cpp(cudart);
    std::cout << "2nd Test" << std::endl;
    stat += test_sgemv_zero_v_cpp(cudart);
    std::cout << "3rd Test" << std::endl;
    stat += test_dsymv_zero_v_cpp(cudart);
    std::cout << "4th Test" << std::endl;
    stat += test_ssymv_zero_v_cpp(cudart);
    std::cout << "5th Test" << std::endl;
    stat += test_sgemv_v_cpp(cudart);
    std::cout << "6th Test" << std::endl;
    stat += test_dgemv_v_cpp(cudart);
    printf("7th Test");
    stat += test_ssymv_v_cpp(cudart);
    printf("8th Test");
    stat += test_dsymv_v_cpp(cudart);
    stat += test_sspmv_v_cpp(cudart);
    stat += test_dspmv_v_cpp(cudart);
    printf("9th Test");
    stat += test_complex_gemv_zero_v_cpp<complex_double>(cudart);
    stat += test_complex_gemv_zero_v_cpp<complex_float>(cudart);
    printf("10th Test");
    stat += test_outer_product<double>(cudart);
    stat += test_outer_product<float>(cudart);
    return stat;
};