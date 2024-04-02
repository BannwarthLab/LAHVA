#include "linalg.hpp"
#include "../../utils/utils.hpp"
#include "level1.hpp"
#include "level1.h"

namespace tcgmtensor{

    /*! @brief Take inner product of two vectors of doubles
        \param[in] nelemXY number of elements in vectors X and Y
        \param[in] X vector
        \param[in] Y vector
    */ 
    double InnerVectorProduct(const vector<double>& X, const vector<double>& Y)
    {   check_equal_size(X,Y);
        return InnerVectorProduct(X.size(), X.data(), Y.data());}

    /*! @brief Take inner product of two vectors of float
        \param[in] X vector
        \param[in] Y vector
    */ 
    float InnerVectorProduct( const vector<float>& X, const vector<float>& Y)
    { 
        check_equal_size(X,Y);
    return  InnerVectorProduct(X.size(), X.data(), Y.data());
    }

    // strided vector product
    double InnerVectorProduct(const vector<double>& X, const size_t strideX, const vector<double>& Y, const size_t strideY)
    {
        check_equal_size(X,Y);
    return InnerVectorProduct(X.size(), X.data(), strideX, Y.data(), strideY);
    }

    float InnerVectorProduct(const vector<float>& X, const size_t strideX, const vector<float>& Y, const size_t strideY)
    {
        check_equal_size(X,Y);
    return InnerVectorProduct(X.size(), X.data(), strideX, Y.data(), strideY);
    }


    /*! Simple interface to DAXPY \f$\vec{y}=\alpha\vec{x}+\vec{y}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] a \f$\alpha\f$
    */
    void AddVectors(const double a, const vector<double>& x, vector<double>& y) {
        check_equal_size(x,y);
        AddVectors(x.size(), a, x.data(), y.data());
    }

    /*! Simple interface to SAXPY \f$\vec{y}=\alpha\vec{x}+\vec{y}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] a \f$\alpha\f$
    */
    void AddVectors(const float a, const vector<float>& x, vector<float>& y) {
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
    void AddVectors(const double a, const vector<double>& x, size_t ix, vector<double>& y, size_t iy) {
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
    void AddVectors(const float a, const vector<float>& x, size_t ix, vector<float>& y, size_t iy) {
        check_equal_size(x,y);
        AddVectors(x.size(), a, x.data(), ix, y.data(), iy);
    }

    //Copy routines////////////////////////////////////////////////////////////////////

    /*! Simple interface to DCOPY \f$\vec{y}=\vec{x}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void CopyVectors(const vector<double>& x, vector<double>& y) {
        check_equal_size(x,y);
        CopyVectors(x.size(), x.data(), y.data());
    }

    /*! Simple interface to SCOPY \f$\vec{y}=\vec{x}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void CopyVectors(const vector<float>& x, vector<float>& y) {
        check_equal_size(x,y);
        CopyVectors(x.size(), x.data(), y.data());
    }


    /*! Simple interface to DCOPY \f$\vec{y}=\vec{x}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void CopyVectors(const vector<double>& x, size_t ix, vector<double>& y, size_t iy) {
        check_equal_size(x,y);
        CopyVectors(x.size(), x.data(), ix, y.data(), iy);
    }

    /*! Simple interface to SCOPY \f$\vec{y}=\vec{x}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void CopyVectors(const vector<float>& x, size_t ix, vector<float>& y, size_t iy) {
        CopyVectors(x.size(), x.data(), ix, y.data(), iy);
    }

    //Swap routines////////////////////////////////////////////////////////////////////

    /*! Simple interface to DSWAP \f$\vec{y}<=>\vec{x}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void SwapVectors(vector<double>& x, vector<double>& y) {
        check_equal_size(x,y);
        SwapVectors(x.size(), x.data(), y.data());
    }

    /*! Simple interface to SSWAP \f$\vec{y}<=>\vec{x}\f$ assuming unit stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void SwapVectors(vector<float>& x, vector<float>& y) {
        check_equal_size(x,y);
        SwapVectors(x.size(), x.data(), y.data());
    }


    /*! Simple interface to DSWAP \f$\vec{y}<=>\vec{x}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void SwapVectors(vector<double>& x, size_t ix, vector<double>& y, size_t iy) {
        check_equal_size(x,y);
        SwapVectors(x.size(), x.data(), ix, y.data(), iy);
    }

    /*! Simple interface to SSWAP \f$\vec{y}<=>\vec{x}\f$ for specified stride
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void SwapVectors(vector<float>& x, size_t ix, vector<float>& y, size_t iy) {
        check_equal_size(x,y);
        SwapVectors(x.size(), x.data(), ix, y.data(), iy);
    } 

    //Scale routines////////////////////////////////////////////////////////////////////

    /*! Simple interface to DSCAL \f$\vec{x}=\alpha\vec{x}\f$ assuming unit stride
        \param[in, out] x \f$\vec{x}\f$
        \param[in,out] a \f$\alpha\f$
    */
    void ScaleVector(const double a, vector<double>& x) {
        ScaleVector(x.size(), a, x.data());
    }
     /*! Simple interface to SSCAL \f$\vec{x}=\alpha\vec{x}\f$ assuming unit stride
        \param[in, out] x \f$\vec{x}\f$
        \param[in,out] a \f$\alpha\f$
    */
    void ScaleVector(const float a, vector<float>& x){
        ScaleVector(x.size(), a, x.data());
    }

    void ScaleVector(const double a, vector<double>& x, size_t ix){
        ScaleVector(x.size(), a, x.data(), ix);
    }
     /*! Simple interface to SSCAL \f$\vec{x}=\alpha\vec{x}\f$ assuming unit stride
        \param[in] n size of vectors \f$x\f$ and \f$y\f$
        \param[in, out] x \f$\vec{x}\f$
        \param[in,out] a \f$\alpha\f$
        stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
    */
    void ScaleVector(const float a, vector<float>& x, size_t ix){
        ScaleVector(x.size(), a, x.data());
    }
}