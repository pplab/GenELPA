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

int ELPA_Solver::eigenvector(complex<double>* A, double* EigenValue, complex<double>* EigenVector)
{   int info, success, allsuccess;
    double t;

    if(loglevel>0)
    {
        t=-1;
        timer(myid, "elpa_solve_evp_complex_2stage", "1", t);
    }
    info=elpa_solve_evp_complex_2stage(nFull, nev,  reinterpret_cast<double _Complex*>(A), lda,
                                       EigenValue,  reinterpret_cast<double _Complex*>(EigenVector),
                                       lda, nblk, nacols,
                                       mpi_comm_rows, mpi_comm_cols, comm_f, kernel_id);
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

    // transform A to A~
    if(loglevel>0)
    {
        t=-1;
        timer(myid, "transform A to A~", "2", t);
    }
    if(DecomposedState == 1 || DecomposedState == 2)
    {
        // calculate A*U^-1, put to work
        Cpzgemm('C', 'N', nFull, 1.0, A, B, 0.0, zwork, desc);

        // calculate U^-C^(A*U^-1), put to a
        Cpzgemm('C', 'N', nFull, 1.0, B, zwork, 0.0, A, desc);
    }
    else {
        // calculate b*a^C and put to work
        Cpzgemm('N', 'C', nFull, 1.0, B, A, 0.0, zwork, desc);
        // calculate b*work^C and put to a -- original A*x=v*B*x was transform to a*x'=v*x'
        Cpzgemm('N', 'C', nFull, 1.0, B, zwork, 0.0, A, desc);
    }
    if(loglevel>0)
    {
        timer(myid, "transform A to A~", "2", t);
    }

    if(loglevel>0)
    {
        t=-1;
        timer(myid, "elpa_solve_evp_complex_2stage", "3", t);
    }
    success=elpa_solve_evp_complex_2stage(nFull, nev,  reinterpret_cast<double _Complex*>(A), lda,
                                          EigenValue, reinterpret_cast<double _Complex*>(EigenVector),
                                          lda, nblk, nacols,
                                          mpi_comm_rows, mpi_comm_cols, comm_f, kernel_id);
    if(loglevel>0)
    {
        timer(myid, "elpa_solve_evp_complex_2stage", "3", t);
    }
    MPI_Allreduce(&success, &allsuccess, 1, MPI_INT, MPI_MIN, comm);
    if(allsuccess != 1)
        return allinfo=1;
    if(loglevel>0)
    {
        t=-1;
        timer(myid, "composeEigenVector", "4", t);
    }
    allinfo=composeEigenVector(DecomposedState, B, EigenVector);
    if(loglevel>0)
    {
        timer(myid, "composeEigenVector", "4", t);
    }
    return allinfo;
}

int ELPA_Solver::decomposeRightMatrix(complex<double>* B, double* EigenValue, complex<double>* EigenVector, int& DecomposedState)
{
    double _Complex* b = reinterpret_cast<double _Complex*>(B);
    double _Complex* q = reinterpret_cast<double _Complex*>(EigenVector);

    int info=0;
    int allinfo=0;
    int success;
    int allsuccess;

    // first try cholesky decomposing
    if(nFull<CHOLESKY_CRITICAL_SIZE)
    {
        DecomposedState=1;
        info=Cpzpotrf('U', nFull, B, desc);
        MPI_Allreduce(&info, &allinfo, 1, MPI_INT, MPI_MAX, comm);
        if(allinfo != 0) //if pdpotrf fail, try elpa_cholesky_real
        {
            DecomposedState=2;
            success=elpa_cholesky_complex(nFull, b, narows, nblk, nacols, mpi_comm_rows, mpi_comm_cols, wantDebug);
            MPI_Allreduce(&success, &allsuccess, 1, MPI_INT, MPI_MIN, comm);
            if(allsuccess != 1)
                allinfo=1;
        }
    } else
    {
        DecomposedState=2;
        success=elpa_cholesky_complex(nFull, b, narows, nblk, nacols, mpi_comm_rows, mpi_comm_cols, wantDebug);
        MPI_Allreduce(&success, &allsuccess, 1, MPI_INT, MPI_MIN, comm);
        if(allsuccess == 1)
            allinfo=0;
        else
            allinfo=1;
        if(allinfo != 0)
        {
            DecomposedState=1;
            info=Cpzpotrf('U', nFull, B, desc);
            MPI_Allreduce(&info, &allinfo, 1, MPI_INT, MPI_MAX, comm);
        }
    }

    if(allinfo==0) // calculate the inverse U^{-1}
    {
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
        info=elpa_invert_trm_complex(nFull, b, lda, nblk, nacols, mpi_comm_rows, mpi_comm_cols, wantDebug);
        if(loglevel>2) saveMatrix("U_inv.dat", nFull, B, desc, cblacs_ctxt);
    }
    else {
        // if cholesky decomposing failed, try diagonalize// calculate U^{-1}
        // calculate B^{-1/2}_{i,j}=\sum_k q_{i,k}*ev_k^{-1/2}*q_{j,k} and put to b, which will be b^-1/2
        DecomposedState=3;
        success=elpa_solve_evp_complex_2stage(nFull, nFull, b, lda, EigenValue, q, lda, nblk, nacols,
                                              mpi_comm_rows, mpi_comm_cols, comm_f, kernel_id);
        MPI_Allreduce(&success, &allsuccess, 1, MPI_INT, MPI_MIN, comm);
        if(allsuccess != 1)
            allinfo=0;
        else
            allinfo=1;
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
        Cpzgemm('N', 'C', nFull, 1.0, zwork, EigenVector, 0.0, B, desc);
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

// calculate the error
// $ \ket{ \delta \psi_i } = H\ket{\psi_i} $
// $ \delta_i = \braket{ \delta \psi_i | \delta \psi_i } $
//
// V: eigenvector matrix
// D: Diagonal matrix of eigenvalue
// maxError: maximum absolute value of error
// meanError: mean absolute value of error
void ELPA_Solver::verify(complex<double>* A, double* EigenValue, complex<double>* EigenVector,
                         double &maxError, double &meanError)
{
    complex<double>* V=EigenVector;
    const int naloc=narows*nacols;
    complex<double>* D=new complex<double>[naloc];
    complex<double>* R=zwork;

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
    Cpzhemm('R', 'U', nFull, 1.0, D, V, 0.0, R, desc);
    // R=A*V-V*D=A*V-R
    Cpzhemm('L', 'U', nFull, 1.0, A, V, -1.0, R, desc);
    // calculate the maximum and mean value of sum_i{R(:,i)*R(:,i)}
    double sumError=0;
    maxError=0;
    for(int i=1; i<=nFull; ++i)
    {
        complex<double> E;
        Cpzdotc(nFull, E, R, 1, i, 1,
                         R, 1, i, 1, desc);
        double abs_E=std::abs(E);
        sumError+=abs_E;
        maxError=std::max(maxError, abs_E);
    }
    meanError=sumError/nFull;
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
void ELPA_Solver::verify(complex<double>* A, complex<double>* B,
                        double* EigenValue, complex<double>* EigenVector,
                        double &maxError, double &meanError)
{
    complex<double>* V=EigenVector;
    const int naloc=narows*nacols;
    complex<double>* D=new complex<double>[naloc];
    complex<double>* R=new complex<double>[naloc];

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
    Cpzhemm('L', 'U', nFull, 1.0, B, V, 0.0, zwork, desc);
    // R=B*V*D=dwork*D
    Cpzhemm('R', 'U', nFull, 1.0, D, zwork, 0.0, R, desc);
    // R=A*V-B*V*D=A*V-R
    Cpzhemm('L', 'U', nFull, 1.0, A, V, -1.0, R, desc);
    // calculate the maximum and mean value of sum_i{R(:,i)*R(:,i)}
    double sumError=0;
    maxError=0;
    for(int i=1; i<=nFull; ++i)
    {
        complex<double> E;
        Cpzdotc(nFull, E, R, 1, i, 1,
                         R, 1, i, 1, desc);
        double abs_E=std::abs(E);
        sumError+=abs_E;
        maxError=std::max(maxError, abs_E);
    }
    meanError=sumError/nFull;

    delete[] D;
    delete[] R;
}
