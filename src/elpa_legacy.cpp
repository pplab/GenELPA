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
