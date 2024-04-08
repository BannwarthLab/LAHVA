#include "linalg.hpp"
#include "../../utils/utils.hpp"
#include "level1.hpp"
#include "level1.h"

namespace tcgmtensor{
    namespace cpu{
    /*! @brief Take inner product of two Vectors of doubles
        \param[in] nelemXY number of elements in Vectors X and Y
        \param[in] X Vector
        \param[in] Y Vector
    */ 
    double InnerVectorProduct(const Vector<double>& X, const Vector<double>& Y)
    {   check_equal_size(X,Y);
        return InnerVectorProduct(X.size(), X.data(), Y.data());}

    /*! @brief Take inner product of two Vectors of float
        \param[in] X Vector
        \param[in] Y Vector
    */ 
    float InnerVectorProduct( const Vector<float>& X, const Vector<float>& Y)
    { 
        check_equal_size(X,Y);
    return  InnerVectorProduct(X.size(), X.data(), Y.data());
    }

    // strided Vector product
    double InnerVectorProduct(const Vector<double>& X, const size_t strideX, const Vector<double>& Y, const size_t strideY)
    {
        check_equal_size(X,Y);
    return InnerVectorProduct(X.size(), X.data(), strideX, Y.data(), strideY);
    }

    float InnerVectorProduct(const Vector<float>& X, const size_t strideX, const Vector<float>& Y, const size_t strideY)
    {
        check_equal_size(X,Y);
    return InnerVectorProduct(X.size(), X.data(), strideX, Y.data(), strideY);
    }


    /*! Simple interface to DAXPY \f$\vec{y}=\alpha\vec{x}+\vec{y}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] a \f$\alpha\f$
    */
    void AddVectors(const double a, const Vector<double>& x, Vector<double>& y) {
        check_equal_size(x,y);
        AddVectors(x.size(), a, x.data(), y.data());
    }

    /*! Simple interface to SAXPY \f$\vec{y}=\alpha\vec{x}+\vec{y}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] a \f$\alpha\f$
    */
    void AddVectors(const float a, const Vector<float>& x, Vector<float>& y) {
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
    void AddVectors(const double a, const Vector<double>& x, size_t ix, Vector<double>& y, size_t iy) {
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
    void AddVectors(const float a, const Vector<float>& x, size_t ix, Vector<float>& y, size_t iy) {
        check_equal_size(x,y);
        AddVectors(x.size(), a, x.data(), ix, y.data(), iy);
    }

    //Copy routines////////////////////////////////////////////////////////////////////

    /*! Simple interface to DCOPY \f$\vec{y}=\vec{x}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void CopyVectors(const Vector<double>& x, Vector<double>& y) {
        check_equal_size(x,y);
        CopyVectors(x.size(), x.data(), y.data());
    }

    /*! Simple interface to SCOPY \f$\vec{y}=\vec{x}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void CopyVectors(const Vector<float>& x, Vector<float>& y) {
        check_equal_size(x,y);
        CopyVectors(x.size(), x.data(), y.data());
    }


    /*! Simple interface to DCOPY \f$\vec{y}=\vec{x}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void CopyVectors(const Vector<double>& x, size_t ix, Vector<double>& y, size_t iy) {
        check_equal_size(x,y);
        CopyVectors(x.size(), x.data(), ix, y.data(), iy);
    }

    /*! Simple interface to SCOPY \f$\vec{y}=\vec{x}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void CopyVectors(const Vector<float>& x, size_t ix, Vector<float>& y, size_t iy) {
        CopyVectors(x.size(), x.data(), ix, y.data(), iy);
    }

    //Swap routines////////////////////////////////////////////////////////////////////

    /*! Simple interface to DSWAP \f$\vec{y}<=>\vec{x}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void SwapVectors(Vector<double>& x, Vector<double>& y) {
        check_equal_size(x,y);
        SwapVectors(x.size(), x.data(), y.data());
    }

    /*! Simple interface to SSWAP \f$\vec{y}<=>\vec{x}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void SwapVectors(Vector<float>& x, Vector<float>& y) {
        check_equal_size(x,y);
        SwapVectors(x.size(), x.data(), y.data());
    }


    /*! Simple interface to DSWAP \f$\vec{y}<=>\vec{x}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void SwapVectors(Vector<double>& x, size_t ix, Vector<double>& y, size_t iy) {
        check_equal_size(x,y);
        SwapVectors(x.size(), x.data(), ix, y.data(), iy);
    }

    /*! Simple interface to SSWAP \f$\vec{y}<=>\vec{x}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void SwapVectors(Vector<float>& x, size_t ix, Vector<float>& y, size_t iy) {
        check_equal_size(x,y);
        SwapVectors(x.size(), x.data(), ix, y.data(), iy);
    } 

    //Scale routines////////////////////////////////////////////////////////////////////

    /*! Simple interface to DSCAL \f$\vec{x}=\alpha\vec{x}\f$ assuming unit stride
        \param[in, out] x \f$\vec{x}\f$
        \param[in,out] a \f$\alpha\f$
    */
    void ScaleVector(const double a, Vector<double>& x) {
        ScaleVector(x.size(), a, x.data());
    }
     /*! Simple interface to SSCAL \f$\vec{x}=\alpha\vec{x}\f$ assuming unit stride
        \param[in, out] x \f$\vec{x}\f$
        \param[in,out] a \f$\alpha\f$
    */
    void ScaleVector(const float a, Vector<float>& x){
        ScaleVector(x.size(), a, x.data());
    }

    void ScaleVector(const double a, Vector<double>& x, size_t ix){
        ScaleVector(x.size(), a, x.data(), ix);
    }
     /*! Simple interface to SSCAL \f$\vec{x}=\alpha\vec{x}\f$ assuming unit stride
        \param[in] n size of Vectors \f$x\f$ and \f$y\f$
        \param[in, out] x \f$\vec{x}\f$
        \param[in,out] a \f$\alpha\f$
        stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
    */
    void ScaleVector(const float a, Vector<float>& x, size_t ix){
        ScaleVector(x.size(), a, x.data());
    }
    }
}