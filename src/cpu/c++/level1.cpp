#include "../../utils/utils.hpp"
#include "impl/cpu/level1.hpp"
#include "impl/cpu/level1.h"

namespace tcgmtensor{
    namespace cpu{
    /*! @brief Take inner product of two Vectors of doubles
        \param[in] nelemXY number of elements in Vectors X and Y
        \param[in] X Vector
        \param[in] Y Vector
    */ 
    double InnerVectorProduct(const Tensor<double>& X, const Tensor<double>& Y)
    {   check_equal_size(X,Y);
        return InnerVectorProduct(X.size(), X.data(), Y.data());}

    /*! @brief Take inner product of two Vectors of float
        \param[in] X Vector
        \param[in] Y Vector
    */ 
    float InnerVectorProduct( const Tensor<float>& X, const Tensor<float>& Y)
    { 
        check_equal_size(X,Y);
    return  InnerVectorProduct(X.size(), X.data(), Y.data());
    }

    // strided Vector product
    double InnerVectorProduct(const Tensor<double>& X, const size_t strideX, const Tensor<double>& Y, const size_t strideY)
    {
        check_equal_size(X,Y);
    return InnerVectorProduct(X.size(), X.data(), strideX, Y.data(), strideY);
    }

    float InnerVectorProduct(const Tensor<float>& X, const size_t strideX, const Tensor<float>& Y, const size_t strideY)
    {
        check_equal_size(X,Y);
    return InnerVectorProduct(X.size(), X.data(), strideX, Y.data(), strideY);
    }

    double InnerVectorProduct(const CPURuntime& rt_, const Tensor<double>& X, const Tensor<double>& Y)
    {
        return InnerVectorProduct(X, Y);
    };
    float InnerVectorProduct(const CPURuntime& rt_, const Tensor<float>& X, const Tensor<float>& Y)
    {
        return InnerVectorProduct(X, Y);
    };
    double InnerVectorProduct(const CPURuntime& rt_, const Tensor<double>& X, const size_t strideX, const Tensor<double>& Y, const size_t strideY)
    {
        return InnerVectorProduct(X, strideX, Y, strideY);
    };
    float InnerVectorProduct(const CPURuntime& rt_, const Tensor<float>& X, const size_t strideX, const Tensor<float>& Y, const size_t strideY)
    {
        return InnerVectorProduct(X, strideX, Y, strideY);
    };


    /*! Simple interface to DAXPY \f$\vec{y}=\alpha\vec{x}+\vec{y}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] a \f$\alpha\f$
    */
    void AddVectors(const double a, const Tensor<double>& x, Tensor<double>& y) {
        check_equal_size(x,y);
        AddVectors(x.size(), a, x.data(), y.data());
    }

    /*! Simple interface to SAXPY \f$\vec{y}=\alpha\vec{x}+\vec{y}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] a \f$\alpha\f$
    */
    void AddVectors(const float a, const Tensor<float>& x, Tensor<float>& y) {
        check_equal_size(x,y);
        AddVectors(x.size(), a, x.data(), y.data());
    }


    /*! Simple interface to DAXPY \f$\vec{y}=a\vec{x}+\vec{y}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] a \f$\alpha\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void AddVectors(const double a, const Tensor<double>& x, size_t ix, Tensor<double>& y, size_t iy) {
        check_equal_size(x,y);
        AddVectors(x.size(), a, x.data(), ix, y.data(), iy);
    }

    /*! Simple interface to sAXPY \f$\vec{y}=a\vec{x}+\vec{y}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] a \f$\alpha\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void AddVectors(const float a, const Tensor<float>& x, size_t ix, Tensor<float>& y, size_t iy) {
        check_equal_size(x,y);
        AddVectors(x.size(), a, x.data(), ix, y.data(), iy);
    }

    //Copy routines////////////////////////////////////////////////////////////////////

    /*! Simple interface to DCOPY \f$\vec{y}=\vec{x}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void CopyVectors(const Tensor<double>& x, Tensor<double>& y) {
        check_equal_size(x,y);
        CopyVectors(x.size(), x.data(), y.data());
    }

    /*! Simple interface to SCOPY \f$\vec{y}=\vec{x}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void CopyVectors(const Tensor<float>& x, Tensor<float>& y) {
        check_equal_size(x,y);
        CopyVectors(x.size(), x.data(), y.data());
    }
    
    void CopyVectors(const Tensor<float>& x, Tensor<double>& y) 
    {
        
        #pragma omp for 
        for (size_t i = 0; i < x.size() ; i++ )
        {
            y.data()[i] = static_cast<double>(x.data()[i]);
        }
    }

    void CopyVectors(const Tensor<double>& x, Tensor<float>& y) 
    {
        
        #pragma omp for 
        for (size_t i = 0; i < x.size() ; i++ )
        {
            y.data()[i] = static_cast<float>(x.data()[i]);
        }
    }

    /*! Simple interface to DCOPY \f$\vec{y}=\vec{x}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void CopyVectors(const Tensor<double>& x, size_t ix, Tensor<double>& y, size_t iy) {
        check_equal_size(x,y);
        CopyVectors(x.size(), x.data(), ix, y.data(), iy);
    }

    /*! Simple interface to SCOPY \f$\vec{y}=\vec{x}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void CopyVectors(const Tensor<float>& x, size_t ix, Tensor<float>& y, size_t iy) {
        CopyVectors(x.size(), x.data(), ix, y.data(), iy);
    }

    

    //Swap routines////////////////////////////////////////////////////////////////////

    /*! Simple interface to DSWAP \f$\vec{y}<=>\vec{x}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void SwapVectors(Tensor<double>& x, Tensor<double>& y) {
        check_equal_size(x,y);
        SwapVectors(x.size(), x.data(), y.data());
    }

    /*! Simple interface to SSWAP \f$\vec{y}<=>\vec{x}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void SwapVectors(Tensor<float>& x, Tensor<float>& y) {
        check_equal_size(x,y);
        SwapVectors(x.size(), x.data(), y.data());
    }


    /*! Simple interface to DSWAP \f$\vec{y}<=>\vec{x}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void SwapVectors(Tensor<double>& x, size_t ix, Tensor<double>& y, size_t iy) {
        check_equal_size(x,y);
        SwapVectors(x.size(), x.data(), ix, y.data(), iy);
    }

    /*! Simple interface to SSWAP \f$\vec{y}<=>\vec{x}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void SwapVectors(Tensor<float>& x, size_t ix, Tensor<float>& y, size_t iy) {
        check_equal_size(x,y);
        SwapVectors(x.size(), x.data(), ix, y.data(), iy);
    } 

    //Scale routines////////////////////////////////////////////////////////////////////

    /*! Simple interface to DSCAL \f$\vec{x}=\alpha\vec{x}\f$ assuming unit stride
        \param[in, out] x \f$\vec{x}\f$
        \param[in,out] a \f$\alpha\f$
    */
    void ScaleVector(const double a, Tensor<double>& x) {
        ScaleVector(x.size(), a, x.data());
    }
     /*! Simple interface to SSCAL \f$\vec{x}=\alpha\vec{x}\f$ assuming unit stride
        \param[in, out] x \f$\vec{x}\f$
        \param[in,out] a \f$\alpha\f$
    */
    void ScaleVector(const float a, Tensor<float>& x){
        ScaleVector(x.size(), a, x.data());
    }

    void ScaleVector(const double a, Tensor<double>& x, size_t ix){
        ScaleVector(x.size(), a, x.data(), ix);
    }
     /*! Simple interface to SSCAL \f$\vec{x}=\alpha\vec{x}\f$ assuming unit stride
        \param[in] n size of Vectors \f$x\f$ and \f$y\f$
        \param[in, out] x \f$\vec{x}\f$
        \param[in,out] a \f$\alpha\f$
        stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
    */
    void ScaleVector(const float a, Tensor<float>& x, size_t ix){
        ScaleVector(x.size(), a, x.data());
    }
    }
}