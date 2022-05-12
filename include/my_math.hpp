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

static inline int Cpdpotrf(const char uplo, const int na, double* U, int* desc)
{
    int isrc=1;
    int info;
    pdpotrf_(&uplo, &na, U, &isrc, &isrc, desc, &info);
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

static inline void Cpdgemm(char transa, char transb, int na,
                           double alpha, double* A, double* B,
                           double beta, double* C, int* desc)
{
    int isrc=1;
    pdgemm_(&transa, &transb, &na, &na, &na,
            &alpha, A, &isrc, &isrc, desc,
                    B, &isrc, &isrc, desc,
            &beta,  C, &isrc, &isrc, desc);
}

static inline void Cpdsymm(char side, char uplo, int na,
                           double alpha, double* A, double* B,
                           double beta, double* C, int* desc)
{
    int isrc=1;
    pdsymm_(&side, &uplo, &na, &na,
            &alpha, A, &isrc, &isrc, desc,
                    B, &isrc, &isrc, desc,
            &beta,  C, &isrc, &isrc, desc);
}

static inline void Cpdgemr2d(int M, int N,
                            double* a, int ia, int ja, int* desca,
                            double* b, int ib, int jb, int* descb, int blacs_ctxt)
{
    pdgemr2d_(&M, &N, a, &ia, &ja, desca, b, &ib, &jb, descb, &blacs_ctxt);
}

static inline void Cpzgemr2d(int M, int N,
                            complex<double>* a, int ia, int ja, int* desca,
                            complex<double>* b, int ib, int jb, int* descb, int blacs_ctxt)
{
	double _Complex* aa=reinterpret_cast<double _Complex*> (a);
	double _Complex* bb=reinterpret_cast<double _Complex*> (b);
    pzgemr2d_(&M, &N, aa, &ia, &ja, desca, bb, &ib, &jb, descb, &blacs_ctxt);
}
