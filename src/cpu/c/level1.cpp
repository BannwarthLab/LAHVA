
#include  "impl/cpu/level1.h"


namespace tcgmtensor{
    namespace cpu
    {
    
    /*! @brief Take inner product of two Vectors of doubles
        \param[in] nelemXY number of elements in Vectors X and Y
        \param[in] X Vector
        \param[in] Y Vector
    */ 
    double InnerVectorProduct(const size_t nelemXY, const double* X, const double* Y)
    {
    if(nelemXY == 0) return 0.0;
    BLAS_INT nXY=(BLAS_INT) nelemXY;
    BLAS_INT incx=1;
    BLAS_INT incy=1;

    return  cblas_ddot(nXY, X, incx, Y, incy);
    }

    /*! @brief Take inner product of two Vectors of float
        \param[in] nelemXY number of elements in Vectors X and Y
        \param[in] X Vector
        \param[in] Y Vector
    */ 
    float InnerVectorProduct(const size_t nelemXY, const float* X, const float* Y)
    {
    if(nelemXY == 0) return 0.0;
    BLAS_INT nXY=(BLAS_INT) nelemXY;
    BLAS_INT incx=1;
    BLAS_INT incy=1;


    return cblas_sdot(nXY, X, incx, Y, incy);
    }

    /// @brief take inner vector product 
    /// @param nelemXY number of elements in the vectors
    /// @param X pointer to X values
    /// @param strideX stride in X
    /// @param Y pointer to Y values
    /// @param strideY stride in Y
    /// @return inner product of X and Y
    double InnerVectorProduct(const size_t nelemXY, const double* X, const size_t strideX, const double* Y, const size_t strideY)
    {
    if(nelemXY == 0) return 0.0;
    BLAS_INT nXY=(BLAS_INT) nelemXY;
    BLAS_INT incx=(BLAS_INT) strideX;
    BLAS_INT incy=(BLAS_INT) strideY;
    return cblas_ddot(nXY, X, incx, Y, incy);
    }

    /// @brief take inner vector product 
    /// @param nelemXY number of elements in the vectors
    /// @param X pointer to X values
    /// @param strideX stride in X
    /// @param Y pointer to Y values
    /// @param strideY stride in Y
    /// @return inner product of X and Y
    float InnerVectorProduct(const size_t nelemXY, const float* X, const size_t strideX, const float* Y, const size_t strideY)
    {
    if(nelemXY == 0) return 0.0;
    BLAS_INT nXY=(BLAS_INT) nelemXY;
    BLAS_INT incx=(BLAS_INT) strideX;
    BLAS_INT incy=(BLAS_INT) strideY;

    return cblas_sdot(nXY, X, incx, Y, incy);
    }


    /*! Simple interface to DAXPY \f$\vec{y}=\alpha\vec{x}+\vec{y}\f$ assuming unit stride
        \param[in] n size of Vectors \f$x\f$ and \f$y\f$
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] a \f$\alpha\f$
    */
    void AddVectors(const size_t ndim, const double a, const double* x, double* y) {
    const BLAS_INT n= (BLAS_INT) ndim;
    const BLAS_INT one = 1;
    cblas_daxpy(n, a, x, one, y, one);
    }

    /*! Simple interface to SAXPY \f$\vec{y}=\alpha\vec{x}+\vec{y}\f$ assuming unit stride
        \param[in] n size of Vectors \f$x\f$ and \f$y\f$
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] a \f$\alpha\f$
    */
    void AddVectors(const size_t ndim, const float a, const float* x, float* y) {
    const BLAS_INT n= (BLAS_INT) ndim;
    const BLAS_INT one = 1;
    cblas_saxpy(n, a, x, one, y, one);
    }


    /*! Simple interface to DAXPY \f$\vec{y}=a\vec{x}+\vec{y}\f$ for specified stride
        \param[in] n size of Vectors \f$x\f$ and \f$y\f$
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] a \f$\alpha\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void AddVectors(const size_t ndim, const double a, const double* x, size_t ix, double* y, size_t iy) {
    const BLAS_INT n = (BLAS_INT) ndim;
    const BLAS_INT incx = (BLAS_INT) ix;
    const BLAS_INT incy = (BLAS_INT) iy;
    cblas_daxpy(n, a, x, incx, y, incy);
    }

    /*! Simple interface to sAXPY \f$\vec{y}=a\vec{x}+\vec{y}\f$ for specified stride
        \param[in] n size of Vectors \f$x\f$ and \f$y\f$
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] a \f$\alpha\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void AddVectors(const size_t ndim, const float a, const float* x, size_t ix, float* y, size_t iy) {
    const BLAS_INT n = (BLAS_INT) ndim;
    const BLAS_INT incx = (BLAS_INT) ix;
    const BLAS_INT incy = (BLAS_INT) iy;
    cblas_saxpy(n, a, x, incx, y, incy);
    }

    //Copy routines////////////////////////////////////////////////////////////////////

    /*! Simple interface to DCOPY \f$\vec{y}=\vec{x}\f$ assuming unit stride
        \param[in] n size of Vectors \f$x\f$ and \f$y\f$
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void CopyVectors(const size_t ndim, const double* x, double* y) {
    const BLAS_INT n= (BLAS_INT) ndim;
    const BLAS_INT one = 1;
    cblas_dcopy(n, x, one, y, one);
    }

    /*! Simple interface to SCOPY \f$\vec{y}=\vec{x}\f$ assuming unit stride
        \param[in] n size of Vectors \f$x\f$ and \f$y\f$
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void CopyVectors(const size_t ndim, const float* x, float* y) {
    const BLAS_INT n= (BLAS_INT) ndim;
    const BLAS_INT one = 1;
    cblas_scopy(n, x, one, y, one);
    }


    /*! Simple interface to DCOPY \f$\vec{y}=\vec{x}\f$ for specified stride
        \param[in] n size of Vectors \f$x\f$ and \f$y\f$
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void CopyVectors(const size_t ndim, const double* x, size_t ix, double* y, size_t iy) {
    const BLAS_INT n = (BLAS_INT) ndim;
    const BLAS_INT incx = (BLAS_INT) ix;
    const BLAS_INT incy = (BLAS_INT) iy;
    cblas_dcopy(n, x, incx, y, incy);
    }

    /*! Simple interface to SCOPY \f$\vec{y}=\vec{x}\f$ for specified stride
        \param[in] n size of Vectors \f$x\f$ and \f$y\f$
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void CopyVectors(const size_t ndim, const float* x, size_t ix, float* y, size_t iy) {
    const BLAS_INT n = (BLAS_INT) ndim;
    const BLAS_INT incx = (BLAS_INT) ix;
    const BLAS_INT incy = (BLAS_INT) iy;
    cblas_scopy(n, x, incx, y, incy);
    }

    /*! Copy for \f$\vec{y}=\vec{x}\f$ where y is in floating point precision and x in double precision
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void CopyVectors(const size_t nelemXY, const double* X, float* Y)
    {
        
        //#pragma omp for 
        for (size_t i = 0; i < nelemXY; i++ )
        {
            Y[i] = static_cast<float>(X[i]);
        }
    }

    //Swap routines////////////////////////////////////////////////////////////////////

    /*! Simple interface to DSWAP \f$\vec{y}<=>\vec{x}\f$ assuming unit stride
        \param[in] n size of Vectors \f$x\f$ and \f$y\f$
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void SwapVectors(const size_t ndim, double* x, double* y) {
    const BLAS_INT n= (BLAS_INT) ndim;
    const BLAS_INT one = 1;
    cblas_dswap(n, x, one, y, one);
    }

    /*! Simple interface to SSWAP \f$\vec{y}<=>\vec{x}\f$ assuming unit stride
        \param[in] n size of Vectors \f$x\f$ and \f$y\f$
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
    */
    void SwapVectors(const size_t ndim, float* x, float* y) {
    const BLAS_INT n = (BLAS_INT) ndim;
    const BLAS_INT one = 1;
    cblas_sswap(n, x, one, y, one);
    }


    /*! Simple interface to DSWAP \f$\vec{y}=a\vec{x}+\vec{y}\f$ for specified stride
        \param[in] n size of Vectors \f$x\f$ and \f$y\f$
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void SwapVectors(const size_t ndim, double* x, size_t ix, double* y, size_t iy) {
    const BLAS_INT n = (BLAS_INT) ndim;
    const BLAS_INT incx = (BLAS_INT) ix;
    const BLAS_INT incy = (BLAS_INT) iy;
    cblas_dswap(n, x, incx, y, incy);
    }

    /*! Simple interface to SSWAP \f$\vec{y}=a\vec{x}+\vec{y}\f$ for specified stride
        \param[in] n size of Vectors \f$x\f$ and \f$y\f$
        \param[in] x \f$\vec{x}\f$
        \param[in,out] y \f$\vec{y}\f$
        \param[in] a \f$\alpha\f$
        \param[in] incx stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \param[in] incy stride of \f$\vec{y}\f$ : \f$\vec{y}_i=y[i*incy]\f$
    */
    void SwapVectors(const size_t ndim, float* x, size_t ix, float* y, size_t iy) {
    const BLAS_INT n = (BLAS_INT) ndim;
    const BLAS_INT incx = (BLAS_INT) ix;
    const BLAS_INT incy = (BLAS_INT) iy;
    cblas_sswap(n, x, incx, y, incy);
    } 

    //Scale routines////////////////////////////////////////////////////////////////////

    /*! Simple interface to DSCAL \f$\vec{x}=\alpha\vec{x}\f$ assuming unit stride
        \param[in] n size of Vectors \f$x\f$ and \f$y\f$
        \param[in, out] x \f$\vec{x}\f$
        \param[in,out] a \f$\alpha\f$
    */
    void ScaleVector(const size_t ndim, const double a, double* x) {
        const BLAS_INT n = (BLAS_INT) ndim;
        const BLAS_INT one = 1;
        cblas_dscal(n, a, x, one);
    }
     /*! Simple interface to SSCAL \f$\vec{x}=\alpha\vec{x}\f$ assuming unit stride
        \param[in] n size of Vectors \f$x\f$ and \f$y\f$
        \param[in, out] x \f$\vec{x}\f$
        \param[in,out] a \f$\alpha\f$
    */
    void ScaleVector(const size_t ndim, const float a, float* x){
        const BLAS_INT n = (BLAS_INT) ndim;
        const BLAS_INT one = 1;
        cblas_sscal(n, a, x, one);
    }

    void ScaleVector(const size_t ndim, const double a, double* x, size_t ix){
        const BLAS_INT n = (BLAS_INT) ndim;
        const BLAS_INT incx = (BLAS_INT) ix;
        cblas_dscal(n, a, x, incx);
    }
     /*! Simple interface to SSCAL \f$\vec{x}=\alpha\vec{x}\f$ assuming unit stride
        \param[in] n size of Vectors \f$x\f$ and \f$y\f$
        \param[in, out] x \f$\vec{x}\f$
        \param[in,out] a \f$\alpha\f$
        stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
    */
    void ScaleVector(const size_t ndim, const float a, float* x, size_t ix){
        const BLAS_INT n = (BLAS_INT) ndim;
        const BLAS_INT incx = (BLAS_INT) ix;
        cblas_sscal(n, a, x, incx);
    }

    //Extrema in Vector routines////////////////////////////////////////////////////////
    /*! Computes the index \f$\i\f$ of the element with the maximum absolute value in vector \f$x\f$.
        \param[in] n size of the vector \f$x\f$
        \param[in] x pointer to the input vector \f$\vec{x}\f$
        \return Index (1-based) of the element with the maximum absolute value in \f$x\f$.
    */
    int IndexMaxFromVector(const size_t ndim, const double* x) {
        const BLAS_INT n = (BLAS_INT) ndim;
        const BLAS_INT one = 1;

        return cblas_idamax(n, x, one);
    }

    int IndexMaxFromVector(const size_t ndim, const float* x) {
        const BLAS_INT n = (BLAS_INT) ndim;
        const BLAS_INT one = 1;

        return cblas_isamax(n, x, one);
    }
    /*! Computes the index \f$\i\f$ of the element with the maximum absolute value in vector \f$x\f$.
        \param[in] n size of the vector \f$x\f$
        \param[in] x pointer to the input vector \f$\vec{x}\f$
        \param[in] ix stride of \f$\vec{x}\f$ : \f$\vec{x}_i=x[i*incx]\f$
        \return Index (1-based) of the element with the maximum absolute value in \f$x\f$.
    */
    int IndexMaxFromVector(const size_t ndim, const double* x, size_t ix) {
        const BLAS_INT n = (BLAS_INT) ndim;
        const BLAS_INT incx = (BLAS_INT) ix;

        return cblas_idamax(n, x, incx);
    }

    int IndexMaxFromVector(const size_t ndim, const float* x, size_t ix) {
        const BLAS_INT n = (BLAS_INT) ndim;
        const BLAS_INT incx = (BLAS_INT) ix;

        return cblas_isamax(n, x, incx);
    }

    }
}
