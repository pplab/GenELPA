#pragma once
// simple wrappers for blas, pblas and scalapack
// NOTE: some parameters of these functions are not supported
extern "C"
{
    #include "blas.h"
    #include "Cblacs.h"
    #include "pblas.h"
    #include "scalapack.h"
}
#include <complex>

static inline void Cdcopy(const int n, double* a, double* b)
{
    int inc=1;
    dcopy_(&n, a, &inc, b, &inc);
}

static inline void Czcopy(const int n, double _Complex* a, double _Complex* b)
{
    int inc=1;
    zcopy_(&n, a, &inc, b, &inc);
}

static inline void Cpddot(int n, double& dot,
                          double* x, int ix, int jx, int incx,
                          double* y, int iy, int jy, int incy, int* desc)
{
    pddot_(&n, &dot, x, &ix, &jx, desc, &incx,
                     y, &iy, &jy, desc, &incy);
}

static inline void Cpzdotc(int n, std::complex<double>& dotc,
                          std::complex<double>* x, int ix, int jx, int incx,
                          std::complex<double>* y, int iy, int jy, int incy, int* desc)
{
	double _Complex* xx=reinterpret_cast<double _Complex*> (x);
	double _Complex* yy=reinterpret_cast<double _Complex*> (y);
	double _Complex* dotc_c=reinterpret_cast<double _Complex*> (&dotc);
    pzdotc_(&n, dotc_c, xx, &ix, &jx, desc, &incx,
                     yy, &iy, &jy, desc, &incy);
}

static inline int Cpdpotrf(const char uplo, const int na, double* U, int* desc)
{
    int isrc=1;
    int info;
    pdpotrf_(&uplo, &na, U, &isrc, &isrc, desc, &info);
    return info;
}

static inline int Cpzpotrf(const char uplo, const int na, std::complex<double>* U, int* desc)
{
    int isrc=1;
    int info;
    double _Complex* uu=reinterpret_cast<double _Complex*> (U);
    pzpotrf_(&uplo, &na, uu, &isrc, &isrc, desc, &info);
    return info;
}

static inline void Cpdtrmm(char side, char uplo, char trans, char diag,
                          int na, double alpha, double* a, double* b, int* desc)
{
    int isrc=1;
    pdtrmm_(&side, &uplo, &trans, &diag, &na, &na,
            &alpha, a, &isrc, &isrc, desc,
                    b, &isrc, &isrc, desc);
}

static inline void Cpztrmm(char side, char uplo, char trans, char diag,
                          int na, double alpha, std::complex<double>* a, 
                          std::complex<double>* b, int* desc)
{
    int isrc=1;
    double _Complex* aa=reinterpret_cast<double _Complex*> (a);
    double _Complex* bb=reinterpret_cast<double _Complex*> (b);
    pztrmm_(&side, &uplo, &trans, &diag, &na, &na,
            &alpha, aa, &isrc, &isrc, desc,
                    bb, &isrc, &isrc, desc);
}

static inline void Cpdgemm(char transa, char transb, int na,
                           double alpha, double* a, double* b,
                           double beta, double* c, int* desc)
{
    int isrc=1;
    pdgemm_(&transa, &transb, &na, &na, &na,
            &alpha, a, &isrc, &isrc, desc,
                    b, &isrc, &isrc, desc,
            &beta,  c, &isrc, &isrc, desc);
}

static inline void Cpzgemm(char transa, char transb, int na,
                           double alpha, std::complex<double>* a, std::complex<double>* b,
                           double beta, std::complex<double>* c, int* desc)
{
	double _Complex* aa=reinterpret_cast<double _Complex*> (a);
	double _Complex* bb=reinterpret_cast<double _Complex*> (b);
	double _Complex* cc=reinterpret_cast<double _Complex*> (c);
    int isrc=1;
    pzgemm_(&transa, &transb, &na, &na, &na,
            &alpha, aa, &isrc, &isrc, desc,
                    bb, &isrc, &isrc, desc,
            &beta,  cc, &isrc, &isrc, desc);
}

static inline void Cpdsymm(char side, char uplo, int na,
                           double alpha, double* a, double* b,
                           double beta, double* c, int* desc)
{
    int isrc=1;
    pdsymm_(&side, &uplo, &na, &na,
            &alpha, a, &isrc, &isrc, desc,
                    b, &isrc, &isrc, desc,
            &beta,  c, &isrc, &isrc, desc);
}

static inline void Cpzsymm(char side, char uplo, int na,
                           std::complex<double> alpha, std::complex<double>* a, std::complex<double>* b,
                           std::complex<double> beta, std::complex<double>* c, int* desc)
{
	double _Complex* alpha_c=reinterpret_cast<double _Complex*> (&alpha);
	double _Complex* beta_c=reinterpret_cast<double _Complex*> (&beta);
	double _Complex* aa=reinterpret_cast<double _Complex*> (a);
	double _Complex* bb=reinterpret_cast<double _Complex*> (b);
	double _Complex* cc=reinterpret_cast<double _Complex*> (c);
    int isrc=1;
    pzsymm_(&side, &uplo, &na, &na,
            alpha_c, aa, &isrc, &isrc, desc,
                    bb, &isrc, &isrc, desc,
            beta_c,  cc, &isrc, &isrc, desc);
}

static inline void Cpdgemr2d(int M, int N,
                            double* a, int ia, int ja, int* desca,
                            double* b, int ib, int jb, int* descb, int blacs_ctxt)
{
    pdgemr2d_(&M, &N, a, &ia, &ja, desca, b, &ib, &jb, descb, &blacs_ctxt);
}

static inline void Cpzgemr2d(int M, int N,
                            std::complex<double>* a, int ia, int ja, int* desca,
                            std::complex<double>* b, int ib, int jb, int* descb, int blacs_ctxt)
{
	double _Complex* aa=reinterpret_cast<double _Complex*> (a);
	double _Complex* bb=reinterpret_cast<double _Complex*> (b);
    pzgemr2d_(&M, &N, aa, &ia, &ja, desca, bb, &ib, &jb, descb, &blacs_ctxt);
}
