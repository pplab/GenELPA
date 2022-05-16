#include <complex>
#include <regex>
#include <fstream>
#include <cfloat>
#include <cmath>
#include <cstring>

#include <iostream>
#include <sstream>

#include <mpi.h>

#include "elpa_legacy.h"
#include "elpa_solver.h"

#include "my_math.hpp"
#include "utils.h"

ELPA_Solver::ELPA_Solver(bool isReal, MPI_Comm comm, int nev,
                         int narows, int nacols, int* desc)
{
    this->comm=comm;
    this->nev=nev;
    this->narows=narows;
    this->nacols=nacols;
    this->desc=desc;

    method=0;
    kernel_id=0;
    useQR=0;
    wantDebug=0;
    cblacs_ctxt=desc[1];
    nFull=desc[2];
    nblk=desc[4];
    lda=desc[8];
    Cblacs_gridinfo(cblacs_ctxt, &nprows, &npcols, &myprow, &mypcol);
    comm_f=MPI_Comm_c2f(comm);
    elpa_get_communicators(comm_f, myprow, mypcol, &mpi_comm_rows, &mpi_comm_cols);
    allocate_work(isReal);
    if(isReal)
    {
        kernel_id=read_real_kernel();
    } else
    {
        kernel_id=read_complex_kernel();
    }
    MPI_Comm_rank(comm, &myid);
}

ELPA_Solver::ELPA_Solver(bool isReal, MPI_Comm comm, int nev,
                         int narows, int nacols, int* desc, int* otherParameter)
{
    this->comm=comm;
    this->nev=nev;
    this->narows=narows;
    this->nacols=nacols;
    this->desc=desc;

    kernel_id=otherParameter[0];
    useQR=otherParameter[1];
    wantDebug=otherParameter[2];
    cblacs_ctxt=desc[1];
    nFull=desc[2];
    nblk=desc[4];
    lda=desc[8];
    Cblacs_gridinfo(cblacs_ctxt, &nprows, &npcols, &myprow, &mypcol);
    comm_f=MPI_Comm_c2f(comm);
    elpa_get_communicators(comm_f, myprow, mypcol, &mpi_comm_rows, &mpi_comm_cols);
    allocate_work(isReal);
    if(isReal)
    {
        kernel_id=read_real_kernel();
    } else
    {
        kernel_id=read_complex_kernel();
    }
}

void ELPA_Solver::setLoglevel(int loglevel)
{
    this->loglevel=loglevel;
    if(loglevel>=2)
        wantDebug=1;
}


void ELPA_Solver::setKernel(int kernel)
{
    this->kernel_id=kernel;
}

void ELPA_Solver::setKernel(int kernel, int useQR)
{
    this->kernel_id=kernel;
    this->useQR=useQR;
}

void ELPA_Solver::exit()
{
    delete[] dwork;
    delete[] zwork;
}

int ELPA_Solver::eigenvector(double* A, double* EigenValue, double* EigenVector)
{

    int info=0;
    int success, allsuccess;
    double t;

    if(loglevel>0)
    {
        t=-1;
        timer(myid, "elpa_solve_evp_real_2stage", "1", t);
    }
    info=elpa_solve_evp_real_2stage(nFull, nev, A, lda, EigenValue, EigenVector, lda, nblk, nacols,
                                    mpi_comm_rows, mpi_comm_cols, comm_f,
                                    kernel_id, useQR);
    if(loglevel>0)
    {
        timer(myid, "elpa_solve_evp_real_2stage", "1", t);
    }
    MPI_Allreduce(&success, &allsuccess, 1, MPI_INT, MPI_MIN, comm);
    if(allsuccess == 1)
        info=0;
    else
        info=1;
    return info;
}

int ELPA_Solver::generalized_eigenvector(double* A, double* B, int& DecomposedState,
                                         double* EigenValue, double* EigenVector)
{
    int allinfo;
    int success, allsuccess;
    double t;

    if(loglevel>0)
    {
        t=-1;
        timer(myid, "decomposeRightMatrix", "1", t);
    }
    if(DecomposedState==0)
        allinfo=decomposeRightMatrix(B, EigenValue, EigenVector, DecomposedState);
    else
        allinfo=0;
    if(loglevel>0)
    {
        timer(myid, "decomposeRightMatrix", "1", t);
    }

    if(allinfo != 0)
        return allinfo;

    if(DecomposedState == 1 || DecomposedState == 2)
    {
        // calculate A*U^-1, put to work
        if(loglevel>0)
        {
            t=-1;
            timer(myid, "pdgemm_", "2", t);
        }
        Cpdgemm('T', 'N', nFull, 1.0, A, B, 0.0, dwork, desc);
        if(loglevel>0)
        {
            timer(myid, "pdgemm_", "2", t);
        }

        // calculate U^-T*(A*U^-1), put to a
        if(loglevel>0)
        {
            t=-1;
            timer(myid, "pdgemm_", "3", t);
        }
        Cpdgemm('T', 'N', nFull, 1.0, B, dwork, 0.0, A, desc);
        if(loglevel>0)
        {
            timer(myid, "pdgemm_", "3", t);
        }
    }
    else
    {
        // calculate b*a^T and put to work
        if(loglevel>0)
        {
            t=-1;
            timer(myid, "pdgemm_", "2", t);
        }
        Cpdgemm('N', 'T', nFull, 1.0, B, A, 0.0, dwork, desc);
        if(loglevel>0)
        {
            timer(myid, "pdgemm_", "2", t);
        }
        // calculate b*work^T and put to a -- origian A*x=v*B*x was transform to a*x'=v*x'
        if(loglevel>0)
        {
            t=-1;
            timer(myid, "pdgemm_", "3", t);
        }
        Cpdgemm('N', 'T', nFull, 1.0, B, dwork, 0.0, A, desc);
        if(loglevel>0)
        {
            timer(myid, "pdgemm_", "3", t);
        }
    }
    // calculate the eigenvalues and eigenvectors, put to ev and q
    if(loglevel>0)
    {
        t=-1;
        timer(myid, "elpa_solve_evp_real_2stage", "4", t);
    }
    success=elpa_solve_evp_real_2stage(nFull, nev, A, lda, EigenValue, EigenVector, lda, nblk, nacols,
                                        mpi_comm_rows, mpi_comm_cols, comm_f,
                                        kernel_id, useQR);
    if(loglevel>0)
    {
        timer(myid, "elpa_solve_evp_real_2stage", "4", t);
    }
    MPI_Allreduce(&success, &allsuccess, 1, MPI_INT, MPI_MIN, comm);
    if(allsuccess != 1)
        return allinfo=1;
    if(loglevel>0)
    {
        t=-1;
        timer(myid, "composeEigenVector", "5", t);
    }
    allinfo=composeEigenVector(DecomposedState, B, EigenVector);
    if(loglevel>0)
    {
        timer(myid, "composeEigenVector", "5", t);
    }
    return allinfo;
}

int ELPA_Solver::eigenvector(complex<double>* A, double* EigenValue, complex<double>* EigenVector)
{

    int info=0;
    int success, allsuccess;
    double t;

    if(loglevel>0)
    {
        t=-1;
        timer(myid, "elpa_solve_evp_complex_2stage", "1", t);
    }
    info=elpa_solve_evp_complex_2stage(nFull, nev,  reinterpret_cast<double _Complex*>(A), lda,
                                       EigenValue,  reinterpret_cast<double _Complex*>(EigenVector),
                                       lda, nblk, nacols, comm_f,
                                       mpi_comm_rows, mpi_comm_cols, kernel_id);
    if(loglevel>0)
    {
        timer(myid, "elpa_solve_evp_complex_2stage", "1", t);
    }
    MPI_Allreduce(&success, &allsuccess, 1, MPI_INT, MPI_MIN, comm);
    if(allsuccess == 1)
        info=0;
    else
        info=1;
    return info;
}

int ELPA_Solver::generalized_eigenvector(complex<double>* A, complex<double>* B, int& DecomposedState,
                                         double* EigenValue, complex<double>* EigenVector)
{
    int allinfo;
    int success, allsuccess;
    double t;

    if(loglevel>0)
    {
        t=-1;
        timer(myid, "decomposeRightMatrix", "1", t);
    }
    if(DecomposedState==0)
        allinfo=decomposeRightMatrix(B, EigenValue, EigenVector, DecomposedState);
    else
        allinfo=0;
    if(loglevel>0)
    {
        timer(myid, "decomposeRightMatrix", "1", t);
    }

    if(allinfo != 0)
        return allinfo;

    if(DecomposedState == 1 || DecomposedState == 2)
    {
        // calculate A*U^-1, put to work
        if(loglevel>0)
        {
            t=-1;
            timer(myid, "A*U^-1", "2", t);
        }
        Cpzgemm('C', 'N', nFull, 1.0, A, B, 0.0, zwork, desc);
        if(loglevel>0)
        {
            timer(myid, "A*U^-1", "2", t);
        }

        // calculate U^-C^(A*U^-1), put to a
        if(loglevel>0)
        {
            t=-1;
            timer(myid, "U^-C^(A*U^-1)", "3", t);
        }
        Cpzgemm('C', 'N', nFull, 1.0, B, zwork, 0.0, A, desc);
        if(loglevel>0)
        {
            timer(myid, "U^-C^(A*U^-1)", "3", t);
        }
    }
    else
    {
        // calculate b*a^C and put to work
        if(loglevel>0)
        {
            t=-1;
            timer(myid, "b*a^C", "2", t);
        }
        Cpzgemm('N', 'C', nFull, 1.0, B, A, 0.0, zwork, desc);
        if(loglevel>0)
        {
            timer(myid, "b*a^C", "2", t);
        }
        // calculate b*work^C and put to a -- original A*x=v*B*x was transform to a*x'=v*x'
        if(loglevel>0)
        {
            t=-1;
            timer(myid, "b*work^C", "3", t);
        }
        Cpzgemm('N', 'C', nFull, 1.0, B, zwork, 0.0, A, desc);
        if(loglevel>0)
        {
            timer(myid, "b*work^C", "3", t);
        }
    }

    if(loglevel>0)
    {
        t=-1;
        timer(myid, "elpa_solve_evp_complex_2stage", "2", t);
    }
    success=elpa_solve_evp_complex_2stage(nFull, nev,  reinterpret_cast<double _Complex*>(A), lda,
                                          EigenValue, reinterpret_cast<double _Complex*>(EigenVector),
                                          lda, nblk, nacols, comm_f,
                                          mpi_comm_rows, mpi_comm_cols, kernel_id);
    if(loglevel>0)
    {
        timer(myid, "elpa_solve_evp_complex_2stage", "2", t);
    }
    MPI_Allreduce(&success, &allsuccess, 1, MPI_INT, MPI_MIN, comm);
    if(allsuccess != 1)
        return allinfo=1;
    if(loglevel>0)
    {
        t=-1;
        timer(myid, "composeEigenVector", "3", t);
    }
    allinfo=composeEigenVector(DecomposedState, B, EigenVector);
    if(loglevel>0)
    {
        timer(myid, "composeEigenVector", "3", t);
    }
    return allinfo;
}

int ELPA_Solver::read_cpuflag()
{
    int cpuflag=0;

    ifstream f_cpuinfo("/proc/cpuinfo");
    string cpuinfo_line;
    regex cpuflag_ex("flags.*");
    regex cpuflag_avx512(".*avx512.*");
    regex cpuflag_avx2(".*avx2.*");
    regex cpuflag_avx(".*avx.*");
    regex cpuflag_sse(".*sse.*");
    while( getline(f_cpuinfo, cpuinfo_line) )
    {
        if(regex_match(cpuinfo_line, cpuflag_ex) )
        {
            //cout<<cpuinfo_line<<endl;
            if(regex_match(cpuinfo_line, cpuflag_avx512))
            {
                cpuflag=4;
            }
            else if(regex_match(cpuinfo_line, cpuflag_avx2))
            {
                cpuflag=3;
            }
            else if(regex_match(cpuinfo_line, cpuflag_avx))
            {
                cpuflag=2;
            }
            else if(regex_match(cpuinfo_line, cpuflag_sse))
            {
                cpuflag=1;
            }
            break;
        }
    }
    f_cpuinfo.close();
    return cpuflag;
}

int ELPA_Solver::read_real_kernel()
{
    int kernel_id;
    if (const char* env = getenv("ELPA_REAL_KERNEL") )
    {
        if(strcmp(env, "ELPA2_REAL_KERNEL_GENERIC_SIMPLE") == 0)
            kernel_id=ELPA2_REAL_KERNEL_GENERIC_SIMPLE;
        else if(strcmp(env, "ELPA2_REAL_KERNEL_BGP") == 0)
            kernel_id=ELPA2_REAL_KERNEL_BGP;
        else if(strcmp(env, "ELPA2_REAL_KERNEL_BGQ") == 0)
            kernel_id=ELPA2_REAL_KERNEL_BGQ;
        else if(strcmp(env, "ELPA2_REAL_KERNEL_SSE") == 0)
            kernel_id=ELPA2_REAL_KERNEL_SSE;
        else if(strcmp(env, "ELPA2_REAL_KERNEL_SSE_BLOCK2") == 0)
            kernel_id=ELPA2_REAL_KERNEL_SSE_BLOCK2;
        else if(strcmp(env, "ELPA2_REAL_KERNEL_SSE_BLOCK4") == 0)
            kernel_id=ELPA2_REAL_KERNEL_SSE_BLOCK4;
        else if(strcmp(env, "ELPA2_REAL_KERNEL_SSE_BLOCK6") == 0)
            kernel_id=ELPA2_REAL_KERNEL_SSE_BLOCK6;
        else if(strcmp(env, "ELPA2_REAL_KERNEL_AVX_BLOCK2") == 0)
            kernel_id=ELPA2_REAL_KERNEL_AVX_BLOCK2;
        else if(strcmp(env, "ELPA2_REAL_KERNEL_AVX_BLOCK4") == 0)
            kernel_id=ELPA2_REAL_KERNEL_AVX_BLOCK4;
        else if(strcmp(env, "ELPA2_REAL_KERNEL_AVX_BLOCK6") == 0)
            kernel_id=ELPA2_REAL_KERNEL_AVX_BLOCK6;
        else if(strcmp(env, "ELPA2_REAL_KERNEL_AVX2_BLOCK2") == 0)
            kernel_id=ELPA2_REAL_KERNEL_AVX2_BLOCK2;
        else if(strcmp(env, "ELPA2_REAL_KERNEL_AVX2_BLOCK4") == 0)
            kernel_id=ELPA2_REAL_KERNEL_AVX2_BLOCK4;
        else if(strcmp(env, "ELPA2_REAL_KERNEL_AVX2_BLOCK6") == 0)
            kernel_id=ELPA2_REAL_KERNEL_AVX2_BLOCK6;
        else
            kernel_id=ELPA2_REAL_KERNEL_GENERIC;
    }
    else
    {
        int cpuflag=read_cpuflag();
        switch (cpuflag)
        {
            case 4:
            case 3:
                kernel_id=ELPA2_REAL_KERNEL_AVX2_BLOCK6;
                break;
            case 2:
                kernel_id=ELPA2_REAL_KERNEL_AVX_BLOCK6;
                break;
            case 1:
                kernel_id=ELPA2_REAL_KERNEL_SSE_BLOCK6;
                break;
            default:
                kernel_id=ELPA2_REAL_KERNEL_GENERIC;
                break;
        }
    }
    return kernel_id;
}

int ELPA_Solver::read_complex_kernel()
{
    int kernel_id;
    if ( const char* env = getenv("ELPA_COMPLEX_KERNEL") )
    {
        if(strcmp(env, "ELPA2_COMPLEX_KERNEL_GENERIC_SIMPLE") == 0)
            kernel_id=ELPA2_COMPLEX_KERNEL_GENERIC_SIMPLE;
        else if(strcmp(env, "ELPA2_COMPLEX_KERNEL_BGP") == 0)
            kernel_id=ELPA2_COMPLEX_KERNEL_BGP;
        else if(strcmp(env, "ELPA2_COMPLEX_KERNEL_BGQ") == 0)
            kernel_id=ELPA2_COMPLEX_KERNEL_BGQ;
        else if(strcmp(env, "ELPA2_COMPLEX_KERNEL_SSE") == 0)
            kernel_id=ELPA2_COMPLEX_KERNEL_SSE;
        else if(strcmp(env, "ELPA2_COMPLEX_KERNEL_SSE_BLOCK1") == 0)
            kernel_id=ELPA2_COMPLEX_KERNEL_SSE_BLOCK1;
        else if(strcmp(env, "ELPA2_COMPLEX_KERNEL_SSE_BLOCK2") == 0)
            kernel_id=ELPA2_COMPLEX_KERNEL_SSE_BLOCK2;
        else if(strcmp(env, "ELPA2_COMPLEX_KERNEL_AVX_BLOCK1") == 0)
            kernel_id=ELPA2_COMPLEX_KERNEL_AVX_BLOCK1;
        else if(strcmp(env, "ELPA2_COMPLEX_KERNEL_AVX_BLOCK2") == 0)
            kernel_id=ELPA2_COMPLEX_KERNEL_AVX_BLOCK2;
        else if(strcmp(env, "ELPA2_COMPLEX_KERNEL_AVX2_BLOCK1") == 0)
            kernel_id=ELPA2_COMPLEX_KERNEL_AVX2_BLOCK1;
        else if(strcmp(env, "ELPA2_COMPLEX_KERNEL_AVX2_BLOCK2") == 0)
            kernel_id=ELPA2_COMPLEX_KERNEL_AVX2_BLOCK2;
        else
            kernel_id=ELPA2_COMPLEX_KERNEL_GENERIC;
    }
    else
    {
        int cpuflag=read_cpuflag();
        switch (cpuflag)
        {
            case 4:
            case 3:
                kernel_id=ELPA2_COMPLEX_KERNEL_AVX2_BLOCK2;
                break;
            case 2:
                kernel_id=ELPA2_COMPLEX_KERNEL_AVX_BLOCK2;
                break;
            case 1:
                kernel_id=ELPA2_COMPLEX_KERNEL_SSE_BLOCK2;
                break;
            default:
                kernel_id=ELPA2_COMPLEX_KERNEL_GENERIC;
                break;
        }
    }
    return kernel_id;
}

int ELPA_Solver::allocate_work(bool isReal)
{
    unsigned long nloc=narows*nacols; // local size
    unsigned long maxloc; // maximum local size
    MPI_Allreduce(&nloc, &maxloc, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);
    if(isReal)
        dwork=new double[maxloc];
    else
        zwork=new complex<double>[maxloc];
    if(dwork||zwork)
        return 0;
    else
        return 1;
}

int ELPA_Solver::decomposeRightMatrix(double* B, double* EigenValue, double* EigenVector, int& DecomposedState)
{
    int info=0;
    int allinfo=0;
    int success;
    int allsuccess;
    double t;

    // first try cholesky decomposing
    if(nFull<CHOLESKY_CRITICAL_SIZE)
    {
        DecomposedState=1;
        if(loglevel>0)
        {
            t=-1;
            timer(myid, "pdpotrf_", "1", t);
        }
        info=Cpdpotrf('U', nFull, B, desc);
        if(loglevel>0)
        {
            timer(myid, "pdpotrf_", "1", t);
        }
        MPI_Allreduce(&info, &allinfo, 1, MPI_INT, MPI_MAX, comm);
        if(allinfo != 0) //pdpotrf fail, try elpa_cholesky_real
        {
            DecomposedState=2;
            if(loglevel>0)
            {
                t=-1;
                timer(myid, "elpa_cholesky_real", "2", t);
            }
            success=elpa_cholesky_real(nFull, B, narows, nblk, nacols, mpi_comm_rows, mpi_comm_cols, wantDebug);
            if(loglevel>0)
            {
                timer(myid, "elpa_cholesky_real", "2", t);
            }
            MPI_Allreduce(&success, &allsuccess, 1, MPI_INT, MPI_MIN, comm);
            if(allsuccess != 1)
                allinfo=1;
        }
    } else
    {
        DecomposedState=2;
        if(loglevel>0)
        {
            t=-1;
            timer(myid, "elpa_cholesky_real", "1", t);
        }
        success=elpa_cholesky_real(nFull, B, narows, nblk, nacols, mpi_comm_rows, mpi_comm_cols, wantDebug);
        if(loglevel>0)
        {
            timer(myid, "elpa_cholesky_real", "1", t);
        }
        MPI_Allreduce(&success, &allsuccess, 1, MPI_INT, MPI_MIN, comm);
        if(allsuccess == 1)
            allinfo=0;
        else
            allinfo=1;
        if(allinfo != 0)
        {
            DecomposedState=1;
            if(loglevel>0)
            {
                t=-1;
                timer(myid, "pdpotrf_", "2", t);
            }
            info=Cpdpotrf('U', nFull, B, desc);
            if(loglevel>0)
            {
                timer(myid, "pdpotrf_", "2", t);
            }
            MPI_Allreduce(&info, &allinfo, 1, MPI_INT, MPI_MAX, comm);
        }
    }

    // if cholesky decomposing failed, try diagonalize
    if(allinfo!=0)
    {
        DecomposedState=3;
        if(loglevel>0)
        {
            t=-1;
            timer(myid, "elpa_solve_evp_real_2stage", "1", t);
        }
        success=elpa_solve_evp_real_2stage(nFull, nFull, B, lda, EigenValue, EigenVector, lda, nblk, nacols,
                                           mpi_comm_rows, mpi_comm_cols, comm_f,
                                           kernel_id, useQR);
        if(loglevel>0)
        {
            timer(myid, "elpa_solve_evp_real_2stage", "1", t);
        }
        MPI_Allreduce(&success, &allsuccess, 1, MPI_INT, MPI_MIN, comm);
        if(allsuccess != 1)
            allinfo=0;
        else
            allinfo=1;
        // calculate B^{-1/2}_{i,j}=\sum_k q_{i,k}*ev_k^{-1/2}*q_{j,k} and put to b, which will be b^-1/2
        // calculate q*ev and put to work
        for(int i=0; i<nacols; ++i)
        {
            int eidx=globalIndex(i, nblk, npcols, mypcol);
            //double ev_sqrt=1.0/sqrt(ev[eidx]);
            double ev_sqrt=EigenValue[eidx]>DBL_MIN?1.0/sqrt(EigenValue[eidx]):0;
            for(int j=0; j<narows; ++j)
                dwork[i*lda+j]=EigenVector[i*lda+j]*ev_sqrt;
        }

        // calculate qevq=qev*q^T, put to b, which is B^{-1/2}
        if(loglevel>0)
        {
            t=-1;
            timer(myid, "pdgemm_", "2", t);
        }
        Cpdgemm('N',  'T', nFull, 1.0, dwork, EigenVector, 0.0, B, desc);
        if(loglevel>0)
        {
            timer(myid, "pdgemm_", "2", t);
        }
    }
    else// calculate U^{-1}
    {
        // clear low triangle
        for(int j=0; j<nacols; ++j)
        {
            int jGlobal=globalIndex(j, nblk, npcols, mypcol);
            for(int i=0; i<narows; ++i)
            {
                int iGlobal=globalIndex(i, nblk, nprows, myprow);
                if(iGlobal>jGlobal) B[i+j*narows]=0;
            }
        }
        if(loglevel>2) saveMatrix("U.dat", nFull, B, desc, cblacs_ctxt);
        // calculate the inverse U^{-1}
        int ldb=narows;
        info=elpa_invert_trm_real(nFull, B, ldb, nblk, nacols, mpi_comm_rows, mpi_comm_cols, wantDebug);
        if(loglevel>2) saveMatrix("U_inv.dat", nFull, B, desc, cblacs_ctxt);
    }
    return allinfo;
}

int ELPA_Solver::composeEigenVector(int DecomposedState, double* B, double* EigenVector)
{
    if(DecomposedState==1 || DecomposedState==2)
    {
        // transform the eigenvectors to original general equation, let U^-1*q, and put to q

        Cpdtrmm('L', 'U', 'N', 'N', nFull, 1.0, B, EigenVector, desc);
    } else
    {
        // transform the eigenvectors to original general equation, let b^T*q, and put to q
        Cpdgemm('T', 'N', nFull, 1.0, B, dwork, 0.0, EigenVector, desc);

    }
    return 0;
}

int ELPA_Solver::decomposeRightMatrix(complex<double>* B, double* EigenValue, complex<double>* EigenVector, int& DecomposedState)
{
    double _Complex* b = reinterpret_cast<double _Complex*>(B);
    double _Complex* q = reinterpret_cast<double _Complex*>(EigenVector);

    int info=0;
    int allinfo=0;
    int success;
    int allsuccess;
    double t;

    // first try cholesky decomposing
    if(nFull<CHOLESKY_CRITICAL_SIZE)
    {
        DecomposedState=1;
        if(loglevel>0)
        {
            t=-1;
            timer(myid, "pzpotrf_", "1", t);
        }
        info=Cpzpotrf('U', nFull, B, desc);
        if(loglevel>0)
        {
            timer(myid, "pzpotrf_", "1", t);
        }
        MPI_Allreduce(&info, &allinfo, 1, MPI_INT, MPI_MAX, comm);
        if(allinfo != 0) //pdpotrf fail, try elpa_cholesky_real
        {
            DecomposedState=2;
            if(loglevel>0)
            {
                t=-1;
                timer(myid, "elpa_cholesky_complex", "2", t);
            }
            success=elpa_cholesky_complex(nFull, b, narows, nblk, nacols, mpi_comm_rows, mpi_comm_cols, wantDebug);
            if(loglevel>0)
            {
                timer(myid, "elpa_cholesky_complex", "2", t);
            }
            MPI_Allreduce(&success, &allsuccess, 1, MPI_INT, MPI_MIN, comm);
            if(allsuccess != 1)
                allinfo=1;
        }
    } else
    {
        DecomposedState=2;
        if(loglevel>0)
        {
            t=-1;
            timer(myid, "elpa_cholesky_complex", "1", t);
        }
        success=elpa_cholesky_complex(nFull, b, narows, nblk, nacols, mpi_comm_rows, mpi_comm_cols, wantDebug);
        if(loglevel>0)
        {
            timer(myid, "elpa_cholesky_complex", "1", t);
        }
        MPI_Allreduce(&success, &allsuccess, 1, MPI_INT, MPI_MIN, comm);
        if(allsuccess == 1)
            allinfo=0;
        else
            allinfo=1;
        if(allinfo != 0)
        {
            DecomposedState=1;
            if(loglevel>0)
            {
                t=-1;
                timer(myid, "pzpotrf_", "2", t);
            }
            info=Cpzpotrf('U', nFull, B, desc);
            if(loglevel>0)
            {
                timer(myid, "pzpotrf_", "2", t);
            }
            MPI_Allreduce(&info, &allinfo, 1, MPI_INT, MPI_MAX, comm);
        }
    }

    // if cholesky decomposing failed, try diagonalize
    if(allinfo!=0)
    {
        DecomposedState=3;
        if(loglevel>0)
        {
            t=-1;
            timer(myid, "elpa_solve_evp_complex_2stage", "1", t);
        }
        success=elpa_solve_evp_complex_2stage(nFull, nFull, b, lda, EigenValue, q, lda, nblk, nacols,
                                              mpi_comm_rows, mpi_comm_cols, comm_f, kernel_id);
        if(loglevel>0)
        {
            timer(myid, "elpa_solve_evp_complex_2stage", "1", t);
        }
        MPI_Allreduce(&success, &allsuccess, 1, MPI_INT, MPI_MIN, comm);
        if(allsuccess != 1)
            allinfo=0;
        else
            allinfo=1;
        // calculate B^{-1/2}_{i,j}=\sum_k q_{i,k}*ev_k^{-1/2}*q_{j,k} and put to b, which will be b^-1/2
        // calculate q*ev and put to work
        for(int i=0; i<nacols; ++i)
        {
            int eidx=globalIndex(i, nblk, npcols, mypcol);
            //double ev_sqrt=1.0/sqrt(ev[eidx]);
            double ev_sqrt=EigenValue[eidx]>DBL_MIN?1.0/sqrt(EigenValue[eidx]):0;
            for(int j=0; j<narows; ++j)
                zwork[i*lda+j]=q[i*lda+j]*ev_sqrt;
        }

        // calculate qevq=qev*q^T, put to b, which is B^{-1/2}
        if(loglevel>0)
        {
            t=-1;
            timer(myid, "pzgemm_", "2", t);
        }
        Cpzgemm('N', 'C', nFull, 1.0, zwork, EigenVector, 0.0, B, desc);
        if(loglevel>0)
        {
            timer(myid, "pzgemm_", "2", t);
        }
    }
    return allinfo;
}

int ELPA_Solver::composeEigenVector(int DecomposedState, complex<double>* B, complex<double>* EigenVector)
{
    if(DecomposedState==1 || DecomposedState==2)
    {
        // transform the eigenvectors to original general equation, let U^-1*q, and put to q
        Cpztrmm('L', 'U', 'N', 'N', nFull, 1.0, B, EigenVector, desc);
    } else
    {
        // transform the eigenvectors to original general equation, let b^C*q, and put to q
        Cpzgemm('C', 'N', nFull, 1.0, B, zwork, 0.0, EigenVector, desc);

    }
    return 0;
}

void ELPA_Solver::timer(int myid, const char function[], const char step[], double &t0)
{
    //MPI_Barrier(comm);
    double t1;
    stringstream outlog;
    if(t0<0)  // t0 < 0 means this is the init call before the function
    {
        t0=MPI_Wtime();
        outlog.str("");
        outlog<<"DEBUG: Process "<<myid<<" Call "<<function<<endl;
        cout<<outlog.str();
    }
    else {
        t1=MPI_Wtime();
        outlog.str("");
        outlog<<"DEBUG: Process "<<myid<<" Step "
              <<step<<" "<<function<<" time: "<<t1-t0<<" s"<<endl;
        cout<<outlog.str();
    }
}

// calculate remains of A*V - V*D
// V: eigenvector matrix
// D: Diaganal matrix of eigenvalue
// maxRemain: maximum absolute value of remains
// meanRemain: mean absolute value of remains
void ELPA_Solver::verify(double* A, double* EigenValue, double* EigenVector,
                         double &maxError, double &meanError)
{
    double* V=EigenVector;
    const int naloc=narows*nacols;
    double* D=new double[naloc];
    double* R=dwork;

    for(int i=0; i<naloc; ++i)
        D[i]=0;

    for(int i=0; i<nFull; ++i)
    {
        int localRow, localCol;
        int localProcRow, localProcCol;

        localRow=localIndex(i, nblk, nprows, localProcRow);
        if(myprow==localProcRow)
        {
            localCol=localIndex(i, nblk, npcols, localProcCol);
            if(mypcol==localProcCol)
            {
                int idx = localRow + localCol*narows;
                D[idx]=EigenValue[i];
            }
        }
    }

    // R=V*D
    Cpdsymm('R', 'U', nFull, 1.0, D, V, 0.0, R, desc);
    // R=A*V-V*D=A*V-R
    Cpdsymm('L', 'U', nFull, 1.0, A, V, -1.0, R, desc);
    // calculate the maximum and mean value of sum_i{R(:,i)*R(:,i)}
    double sumError=0;
    maxError=0;
    for(int i=1; i<=nFull; ++i)
    {
        double E;
        Cpddot(nFull, E, R, 1, i, 1,
                         R, 1, i, 1, desc);
        //printf("myid: %d, i: %d, E: %lf\n", myid, i, E);
        sumError+=E;
        maxError=std::max(E, maxError);
    }
    meanError=sumError/nFull;
    // global mean and max Error
    delete[] D;
}


// calculate the computational error
// $ \ket{ \delta \psi_i } = (H - \epsilon_i S)\ket{\psi_i} $
// $ \delta_i = \braket{ \delta \psi_i | \delta \psi_i } $
//
// V: eigenvector matrix
// D: Diaganal matrix of eigenvalue
// maxError: maximum absolute value of error
// meanError: mean absolute value of error
void ELPA_Solver::verify(double* A, double* B, double* EigenValue, double* EigenVector,
                        double &maxError, double &meanError)
{
    double* V=EigenVector;
    const int naloc=narows*nacols;
    double* D=new double[naloc];
    double* R=new double[naloc];

    for(int i=0; i<naloc; ++i)
        D[i]=0;

    for(int i=0; i<nFull; ++i)
    {
        int localRow, localCol;
        int localProcRow, localProcCol;

        localRow=localIndex(i, nblk, nprows, localProcRow);
        if(myprow==localProcRow)
        {
            localCol=localIndex(i, nblk, npcols, localProcCol);
            if(mypcol==localProcCol)
            {
                int idx = localRow + localCol*narows;
                D[idx]=EigenValue[i];
            }
        }
    }

    // dwork=B*V
    Cpdsymm('L', 'U', nFull, 1.0, B, V, 0.0, dwork, desc);
    // R=B*V*D=dwork*D
    Cpdsymm('R', 'U', nFull, 1.0, D, dwork, 0.0, R, desc);
    // R=A*V-B*V*D=A*V-R
    Cpdsymm('L', 'U', nFull, 1.0, A, V, -1.0, R, desc);
    // calculate the maximum and mean value of sum_i{R(:,i)*R(:,i)}
    double sumError=0;
    maxError=0;
    for(int i=1; i<=nFull; ++i)
    {
        double E;
        Cpddot(nFull, E, R, 1, i, 1,
                         R, 1, i, 1, desc);
        //printf("myid: %d, i: %d, E: %lf\n", myid, i, E);
        sumError+=E;
        maxError=maxError>E?maxError:E;
    }
    meanError=sumError/nFull;

    delete[] D;
    delete[] R;
}
