#include "../../utils/utils.hpp"
#include "impl/blas/gpu/level1.hpp"
#include "../../gpu-utils/utils.hpp"

namespace lahva{
    namespace gpu{
    /*! @brief Take inner product of two Vectors of doubles
        \param[in] nelemXY number of elements in Vectors X and Y
        \param[in] X Vector
        \param[in] Y Vector
    */ 
    double InnerVectorProduct(const CudaRuntime& cudart, const GPUTensor_<double>& X, const GPUTensor_<double>& Y)
    {   
        check_equal_size(X,Y);
        check_device_alloc( cudart, X);
        check_device_alloc( cudart, Y);
        
        double result;
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDdot(cudart.handle, X.size(), X.gpu_data(), 1, Y.gpu_data(), 1, &result);
        get_cublas_error(istat);
        return result;}

    /*! @brief Take inner product of two Vectors of float
        \param[in] X Vector
        \param[in] Y Vector
    */ 
    float InnerVectorProduct(const CudaRuntime& cudart, const GPUTensor_<float>& X, const GPUTensor_<float>& Y)
    { 
        check_equal_size(X,Y);
        check_device_alloc( cudart, X);
        check_device_alloc( cudart, Y);
        
        float result;
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasSdot(cudart.handle, X.size(), X.gpu_data(), 1, Y.gpu_data(), 1, &result);
        get_cublas_error(istat);
        return result;
    }

    // strided Vector product
    double InnerVectorProduct(const CudaRuntime& cudart, const GPUTensor_<double>& X, const size_t strideX, const GPUTensor_<double>& Y, const size_t strideY)
    {
        check_equal_size(X,Y);
        check_device_alloc( cudart, X);
        check_device_alloc( cudart, Y);
        
        double result;
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDdot(cudart.handle, X.size(), X.gpu_data(), strideX, Y.gpu_data(), strideY, &result);
        get_cublas_error(istat);
        return result;
    }

    float InnerVectorProduct(const CudaRuntime& cudart, const GPUTensor_<float>& X, const size_t strideX, const GPUTensor_<float>& Y, const size_t strideY)
    {
        check_equal_size(X,Y);
        check_device_alloc( cudart, X);
        check_device_alloc( cudart, Y);
        
        float result;
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasSdot(cudart.handle, X.size(), X.gpu_data(), strideX, Y.gpu_data(), strideY, &result);
        get_cublas_error(istat);
        return result;
    }


    /*! Simple interface to DAXPY \f$\vec{y}=\alpha\vec{x}+\vec{y}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] a \f$\alpha\f$
    */
    void AddVectors(const CudaRuntime& cudart, const double a, const GPUTensor_<double>& x, GPUTensor_<double>& y) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDaxpy(cudart.handle, x.size(), &a, x.gpu_data(), 1, y.gpu_data(), 1);
        get_cublas_error(istat);
    }

    /*! Simple interface to SAXPY \f$\vec{y}=\alpha\vec{x}+\vec{y}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] a \f$\alpha\f$
    */
    void AddVectors(const CudaRuntime& cudart, const float a, const GPUTensor_<float>& x, GPUTensor_<float>& y) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasSaxpy(cudart.handle, x.size(), &a, x.gpu_data(), 1, y.gpu_data(), 1);
        get_cublas_error(istat); 
    }


    /*! Simple interface to DAXPY \f$\vec{y}=a\vec{x}+\vec{y}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] a \f$\alpha\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void AddVectors(const CudaRuntime& cudart, const double a, const GPUTensor_<double>& x, size_t ix, GPUTensor_<double>& y, size_t iy) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDaxpy(cudart.handle, x.size(), &a, x.gpu_data(), ix, y.gpu_data(), iy);
        get_cublas_error(istat);
    }

    /*! Simple interface to sAXPY \f$\vec{y}=a\vec{x}+\vec{y}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] a \f$\alpha\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void AddVectors(const CudaRuntime& cudart, const float a, const GPUTensor_<float>& x, size_t ix, GPUTensor_<float>& y, size_t iy) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasSaxpy(cudart.handle, x.size(), &a, x.gpu_data(), ix, y.gpu_data(), iy);
        get_cublas_error(istat);
    }

    //Copy routines////////////////////////////////////////////////////////////////////

    /*! Simple interface to DCOPY \f$\vec{y}=\vec{x}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void CopyVectors(const CudaRuntime& cudart, const GPUTensor_<double>& x, GPUTensor_<double>& y) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDcopy(cudart.handle, x.size(), x.gpu_data(), 1, y.gpu_data(), 1);
        get_cublas_error(istat); 
    }

    /*! Simple interface to SCOPY \f$\vec{y}=\vec{x}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void CopyVectors(const CudaRuntime& cudart, const GPUTensor_<float>& x, GPUTensor_<float>& y) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasScopy(cudart.handle, x.size(), x.gpu_data(), 1, y.gpu_data(), 1);
        get_cublas_error(istat); 
    }


    /*! Simple interface to DCOPY \f$\vec{y}=\vec{x}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void CopyVectors(const CudaRuntime& cudart, const GPUTensor_<double>& x, size_t ix, GPUTensor_<double>& y, size_t iy) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDcopy(cudart.handle, x.size(), x.gpu_data(), ix, y.gpu_data(), iy);
        get_cublas_error(istat);
    }

    /*! Simple interface to SCOPY \f$\vec{y}=\vec{x}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void CopyVectors(const CudaRuntime& cudart, const GPUTensor_<float>& x, size_t ix, GPUTensor_<float>& y, size_t iy) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasScopy(cudart.handle, x.size(), x.gpu_data(), ix, y.gpu_data(), iy);
        get_cublas_error(istat);
    }

    ////Swap routines////////////////////////////////////////////////////////////////////

    /*! Simple interface to DSWAP \f$\vec{y}<=>\vec{x}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void SwapVectors(const CudaRuntime& cudart, GPUTensor_<double>& x, GPUTensor_<double>& y) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDswap(cudart.handle, x.size(), x.gpu_data(), 1, y.gpu_data(), 1);
        get_cublas_error(istat);
    }

    /*! Simple interface to SSWAP \f$\vec{y}<=>\vec{x}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void SwapVectors(const CudaRuntime& cudart, GPUTensor_<float>& x, GPUTensor_<float>& y) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasSswap(cudart.handle, x.size(), x.gpu_data(), 1, y.gpu_data(), 1);
        get_cublas_error(istat);
    }

    /*! Simple interface to DSWAP \f$\vec{y}<=>\vec{x}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void SwapVectors(const CudaRuntime& cudart, GPUTensor_<double>& x, size_t ix, GPUTensor_<double>& y, size_t iy) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDswap(cudart.handle, x.size(), x.gpu_data(), ix, y.gpu_data(), iy);
        get_cublas_error(istat);
    }

    /*! Simple interface to SSWAP \f$\vec{y}<=>\vec{x}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void SwapVectors(const CudaRuntime& cudart, GPUTensor_<float>& x, size_t ix, GPUTensor_<float>& y, size_t iy) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasSswap(cudart.handle, x.size(), x.gpu_data(), ix, y.gpu_data(), iy);
        get_cublas_error(istat);
    } 

    ////Scale routines////////////////////////////////////////////////////////////////////

    /*! Simple interface to DSCAL \f$\vec{x}=\alpha\vec{x}\f$ assuming unit stride
        \param[in, out] x \f$\vec{x}\f$
        \param[in,out] a \f$\alpha\f$
    */
    void ScaleVector(const CudaRuntime& cudart, const double a, GPUTensor_<double>& x) {
        check_device_alloc( cudart, x); 
        
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDscal(cudart.handle, x.size(), &a, x.gpu_data(), 1);
        get_cublas_error(istat);
    }
     /*! Simple interface to SSCAL \f$\vec{x}=\alpha\vec{x}\f$ assuming unit stride
        \param[in, out] x \f$\vec{x}\f$
        \param[in,out] a \f$\alpha\f$
    */
    void ScaleVector(const CudaRuntime& cudart, const float a, GPUTensor_<float>& x){
        check_device_alloc( cudart, x); 
        
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasSscal(cudart.handle, x.size(), &a, x.gpu_data(), 1);
        get_cublas_error(istat);
    }

    void ScaleVector(const CudaRuntime& cudart, const double a, GPUTensor_<double>& x, size_t ix){
        check_device_alloc( cudart, x); 
        
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDscal(cudart.handle, x.size(), &a, x.gpu_data(), ix);
        get_cublas_error(istat);
    }
     /*! Simple interface to SSCAL \f$\vec{x}=\alpha\vec{x}\f$ assuming unit stride
        \param[in] n size of Vectors \f$x\f$ and \f$y\f$
        \param[in, out] x \f$\vec{x}\f$
        \param[in,out] a \f$\alpha\f$
        stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
    */
    void ScaleVector(const CudaRuntime& cudart, const float a, GPUTensor_<float>& x, size_t ix){
        check_device_alloc( cudart, x); 
        
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasSscal(cudart.handle, x.size(), &a, x.gpu_data(), ix);
        get_cublas_error(istat);
    }
    }
}