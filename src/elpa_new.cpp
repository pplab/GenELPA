#include <complex>
#include <map>
#include <regex>
#include <fstream>
#include <cfloat>
#include <cstring>
#include <mpi.h>

#include <iostream>
#include <sstream>

#include "elpa_new.h"
#include "elpa_solver.h"

#include "my_math.hpp"
#include "utils.hpp"

using namespace std;

static map<int, elpa_t> handle_pool;

ELPA_Solver::ELPA_Solver(bool isReal, MPI_Comm comm, int nev, int narows, int nacols, int* desc)
{
    this->comm=comm;
    this->nev=nev;
    this->narows=narows;
    this->nacols=nacols;
    this->desc=desc;

    kernel_id=0;
    cblacs_ctxt=desc[1];
    nFull=desc[2];
    nblk=desc[4];
    lda=desc[8];
    MPI_Comm_rank(comm, &myid);
    Cblacs_gridinfo(cblacs_ctxt, &nprows, &npcols, &myprow, &mypcol);
    allocate_work(isReal);
    if(isReal)
    {
        kernel_id=read_real_kernel();
    } else
    {
        kernel_id=read_complex_kernel();
    }

    int error;

    static int total_handle=0;

    elpa_init(20210430);

    handle_id = ++total_handle;
    elpa_t handle;

    handle=elpa_allocate(&error);
    handle_pool[handle_id]=handle;

    elpa_set_integer(handle_pool[handle_id], "na", nFull, &error);
    elpa_set_integer(handle_pool[handle_id], "nev", nev, &error);
    elpa_set_integer(handle_pool[handle_id], "local_nrows", narows, &error);
    elpa_set_integer(handle_pool[handle_id], "local_ncols", nacols, &error);
    elpa_set_integer(handle_pool[handle_id], "nblk", nblk, &error);
    elpa_set_integer(handle_pool[handle_id], "mpi_comm_parent", MPI_Comm_c2f(comm), &error);
    elpa_set_integer(handle_pool[handle_id], "process_row", myprow, &error);
    elpa_set_integer(handle_pool[handle_id], "process_col", mypcol, &error);

    error = elpa_setup(handle_pool[handle_id]);
    elpa_set_integer(handle_pool[handle_id], "solver", ELPA_SOLVER_2STAGE, &error);
    if(isReal)
        elpa_set_integer(handle_pool[handle_id], "real_kernel", kernel_id, &error);
    else
        elpa_set_integer(handle_pool[handle_id], "real_kernel", kernel_id, &error);
}

ELPA_Solver::ELPA_Solver(bool isReal, MPI_Comm comm, int nev, int narows, int nacols, int* desc, int* otherParameter)
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
    MPI_Comm_rank(comm, &myid);
    Cblacs_gridinfo(cblacs_ctxt, &nprows, &npcols, &myprow, &mypcol);
    allocate_work(isReal);

    int error;
    static map<int, elpa_t> handle_pool;
    static int total_handle;

    elpa_init(20210430);

    handle_id = ++total_handle;
    elpa_t handle;
    handle=elpa_allocate(&error);
    handle_pool[handle_id]=handle;

    elpa_set_integer(handle_pool[handle_id], "na", nFull, &error);
    elpa_set_integer(handle_pool[handle_id], "nev", nev, &error);
    elpa_set_integer(handle_pool[handle_id], "local_nrows", narows, &error);
    elpa_set_integer(handle_pool[handle_id], "local_ncols", nacols, &error);
    elpa_set_integer(handle_pool[handle_id], "nblk", nblk, &error);
    elpa_set_integer(handle_pool[handle_id], "mpi_comm_parent", MPI_Comm_c2f(comm), &error);
    elpa_set_integer(handle_pool[handle_id], "process_row", myprow, &error);
    elpa_set_integer(handle_pool[handle_id], "process_col", mypcol, &error);
    elpa_set_integer(handle_pool[handle_id], "blacs_context", cblacs_ctxt, &error);
    elpa_set_integer(handle_pool[handle_id], "solver", ELPA_SOLVER_2STAGE, &error);
    elpa_set_integer(handle_pool[handle_id], "debug", wantDebug, &error);
    if(isReal)
    {
        elpa_set_integer(handle_pool[handle_id], "real_kernel", kernel_id, &error);
        elpa_set_integer(handle_pool[handle_id], "qr", useQR, &error);
    }
    else
    {
        elpa_set_integer(handle_pool[handle_id], "complex_kernel", kernel_id, &error);
    }
}

void ELPA_Solver::setLoglevel(int loglevel)
{
    int error;
    this->loglevel=loglevel;
    if(loglevel>=2)
    {
        wantDebug=1;
        elpa_set_integer(handle_pool[handle_id], "verbose", 1, &error);
        elpa_set_integer(handle_pool[handle_id], "debug", wantDebug, &error);

        outputParameters();
    }
}

void ELPA_Solver::setKernel(int kernel)
{
    int error;
    elpa_set_integer(handle_pool[handle_id], "real_kernel", kernel, &error);
}

void ELPA_Solver::setKernel(int kernel, int useQR)
{
    int error;
    elpa_set_integer(handle_pool[handle_id], "real_kernel", kernel, &error);
    elpa_set_integer(handle_pool[handle_id], "qr", useQR, &error);
}

ELPA_Solver::~ELPA_Solver(void)
{
    delete[] dwork;
    delete[] zwork;
    int error;
    for(auto const &handle:handle_pool)
    {
        elpa_deallocate(handle.second, &error);
    }
}

int ELPA_Solver::eigenvector(double* A, double* EigenValue, double* EigenVector)
{
    int info;
    elpa_eigenvectors_all_host_arrays_d(handle_pool[handle_id], A, EigenValue, EigenVector, &info);
    return info;
}

int ELPA_Solver::generalized_eigenvector(double* A, double* B, int& DecomposedState,
                                         double* EigenValue, double* EigenVector)
{
    int info, allinfo;
    double t;

    if(loglevel>0)
    {
        std::stringstream outlog;
        outlog.str("");
        outlog<<"DEBUG: Process "<<myid<<" Enter ELPA_Solver::generalized_eigenvector"<<std::endl;
        std::cout<<outlog.str();
    }

    if(loglevel>0)
        t=-1, timer(myid, "decomposeRightMatrix", "1", t);
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

        // calculate U^-T^(A*U^-1), put to a
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
        // calculate b*work^T and put to a -- original A*x=v*B*x was transform to a*x'=v*x'
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

    if(loglevel>0)
    {
        /*const int naloc=narows*nacols;
        Cdcopy(naloc, A, dwork);
        for(int i=0; i<naloc; ++i)
            EigenVector[i]=0;
        for(int i=0; i<nFull; ++i)
            EigenValue[i]=0;*/
        t=-1;
        timer(myid, "elpa_eigenvectors", "2", t);
    }
    if(loglevel>2) saveMatrix("A_tilde.dat", nFull, A, desc, cblacs_ctxt);
    //elpa_eigenvectors_all_host_arrays_d(handle_pool[handle_id], A, EigenValue, EigenVector, &info);
    info=eigenvector(A, EigenValue, EigenVector);

    if(loglevel>0)
    {
        timer(myid, "elpa_eigenvectors", "2", t);
    }

    MPI_Allreduce(&info, &allinfo, 1, MPI_INT, MPI_MAX, comm);
    if(loglevel>0)
    {
        t=-1;
        timer(myid, "composeEigenVector", "3", t);
    }

    if(loglevel>2) saveMatrix("EigenVector_tilde.dat", nFull, EigenVector, desc, cblacs_ctxt);
    allinfo=composeEigenVector(DecomposedState, B, EigenVector);
    if(loglevel>0)
    {
        timer(myid, "composeEigenVector", "3", t);
    }
    return allinfo;
}

int ELPA_Solver::eigenvector(complex<double>* A, double* EigenValue, complex<double>* EigenVector)
{
    int info;
    elpa_eigenvectors_all_host_arrays_dc(handle_pool[handle_id], reinterpret_cast<double _Complex*>(A),
                        EigenValue, reinterpret_cast<double _Complex*>(EigenVector), &info);
    return info;
}

int ELPA_Solver::generalized_eigenvector(complex<double>* A, complex<double>* B, int& DecomposedState,
                                         double* EigenValue, complex<double>* EigenVector)
{
    int info, allinfo;
    double t;

    if(loglevel>0)
    {
        std::stringstream outlog;
        outlog.str("");
        outlog<<"DEBUG: Process "<<myid<<" Enter ELPA_Solver::generalized_eigenvector(complex)"<<std::endl;
        std::cout<<outlog.str();
    }

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

    if(loglevel>0)
    {
        t=-1;
        timer(myid, "elpa_eigenvectors", "2", t);
    }
    elpa_eigenvectors_all_host_arrays_dc(handle_pool[handle_id], reinterpret_cast<double _Complex*>(A),
                         EigenValue, reinterpret_cast<double _Complex*>(EigenVector), &info);
    if(loglevel>0)
    {
        timer(myid, "elpa_eigenvectors", "2", t);
    }
    MPI_Allreduce(&info, &allinfo, 1, MPI_INT, MPI_MAX, comm);
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

    if (const char* env = getenv("ELPA_DEFAULT_real_kernel") )
    {
        if(strcmp(env, "ELPA_2STAGE_REAL_GENERIC_SIMPLE") == 0)
            kernel_id=ELPA_2STAGE_REAL_GENERIC_SIMPLE;
        else if(strcmp(env, "ELPA_2STAGE_REAL_BGP") == 0)
            kernel_id=ELPA_2STAGE_REAL_BGP;
        else if(strcmp(env, "ELPA_2STAGE_REAL_BGQ") == 0)
            kernel_id=ELPA_2STAGE_REAL_BGQ;
        else if(strcmp(env, "ELPA_2STAGE_REAL_SSE_ASSEMBLY") == 0)
            kernel_id=ELPA_2STAGE_REAL_SSE_ASSEMBLY;
        else if(strcmp(env, "ELPA_2STAGE_REAL_SSE_BLOCK2") == 0)
            kernel_id=ELPA_2STAGE_REAL_SSE_BLOCK2;
        else if(strcmp(env, "ELPA_2STAGE_REAL_SSE_BLOCK4") == 0)
            kernel_id=ELPA_2STAGE_REAL_SSE_BLOCK4;
        else if(strcmp(env, "ELPA_2STAGE_REAL_SSE_BLOCK6") == 0)
            kernel_id=ELPA_2STAGE_REAL_SSE_BLOCK6;
        else if(strcmp(env, "ELPA_2STAGE_REAL_AVX_BLOCK2") == 0)
            kernel_id=ELPA_2STAGE_REAL_AVX_BLOCK2;
        else if(strcmp(env, "ELPA_2STAGE_REAL_AVX_BLOCK4") == 0)
            kernel_id=ELPA_2STAGE_REAL_AVX_BLOCK4;
        else if(strcmp(env, "ELPA_2STAGE_REAL_AVX_BLOCK6") == 0)
            kernel_id=ELPA_2STAGE_REAL_AVX_BLOCK6;
        else if(strcmp(env, "ELPA_2STAGE_REAL_AVX2_BLOCK2") == 0)
            kernel_id=ELPA_2STAGE_REAL_AVX2_BLOCK2;
        else if(strcmp(env, "ELPA_2STAGE_REAL_AVX2_BLOCK4") == 0)
            kernel_id=ELPA_2STAGE_REAL_AVX2_BLOCK4;
        else if(strcmp(env, "ELPA_2STAGE_REAL_AVX2_BLOCK6") == 0)
            kernel_id=ELPA_2STAGE_REAL_AVX2_BLOCK6;
        else if(strcmp(env, "ELPA_2STAGE_REAL_AVX512_BLOCK2") == 0)
            kernel_id=ELPA_2STAGE_REAL_AVX512_BLOCK2;
        else if(strcmp(env, "ELPA_2STAGE_REAL_AVX512_BLOCK4") == 0)
            kernel_id=ELPA_2STAGE_REAL_AVX512_BLOCK4;
        else if(strcmp(env, "ELPA_2STAGE_REAL_AVX512_BLOCK6") == 0)
            kernel_id=ELPA_2STAGE_REAL_AVX512_BLOCK6;
        else if(strcmp(env, "ELPA_2STAGE_REAL_SPARC64_BLOCK2") == 0)
            kernel_id=ELPA_2STAGE_REAL_SPARC64_BLOCK2;
        else if(strcmp(env, "ELPA_2STAGE_REAL_SPARC64_BLOCK4") == 0)
            kernel_id=ELPA_2STAGE_REAL_SPARC64_BLOCK4;
        else if(strcmp(env, "ELPA_2STAGE_REAL_SPARC64_BLOCK6") == 0)
            kernel_id=ELPA_2STAGE_REAL_SPARC64_BLOCK6;
        else if(strcmp(env, "ELPA_2STAGE_REAL_NEON_ARCH64_BLOCK2") == 0)
            kernel_id=ELPA_2STAGE_REAL_NEON_ARCH64_BLOCK2;
        else if(strcmp(env, "ELPA_2STAGE_REAL_NEON_ARCH64_BLOCK4") == 0)
            kernel_id=ELPA_2STAGE_REAL_NEON_ARCH64_BLOCK4;
        else if(strcmp(env, "ELPA_2STAGE_REAL_NEON_ARCH64_BLOCK6") == 0)
            kernel_id=ELPA_2STAGE_REAL_NEON_ARCH64_BLOCK6;
        else if(strcmp(env, "ELPA_2STAGE_REAL_VSX_BLOCK2") == 0)
            kernel_id=ELPA_2STAGE_REAL_VSX_BLOCK2;
        else if(strcmp(env, "ELPA_2STAGE_REAL_VSX_BLOCK4") == 0)
            kernel_id=ELPA_2STAGE_REAL_VSX_BLOCK4;
        else if(strcmp(env, "ELPA_2STAGE_REAL_VSX_BLOCK6") == 0)
            kernel_id=ELPA_2STAGE_REAL_VSX_BLOCK6;
        else if(strcmp(env, "ELPA_2STAGE_REAL_SVE128_BLOCK2") == 0)
            kernel_id=ELPA_2STAGE_REAL_SVE128_BLOCK2;
        else if(strcmp(env, "ELPA_2STAGE_REAL_SVE128_BLOCK4") == 0)
            kernel_id=ELPA_2STAGE_REAL_SVE128_BLOCK4;
        else if(strcmp(env, "ELPA_2STAGE_REAL_SVE128_BLOCK6") == 0)
            kernel_id=ELPA_2STAGE_REAL_SVE128_BLOCK6;
        else if(strcmp(env, "ELPA_2STAGE_REAL_SVE256_BLOCK2") == 0)
            kernel_id=ELPA_2STAGE_REAL_SVE256_BLOCK2;
        else if(strcmp(env, "ELPA_2STAGE_REAL_SVE256_BLOCK4") == 0)
            kernel_id=ELPA_2STAGE_REAL_SVE256_BLOCK4;
        else if(strcmp(env, "ELPA_2STAGE_REAL_SVE256_BLOCK6") == 0)
            kernel_id=ELPA_2STAGE_REAL_SVE256_BLOCK6;
        else if(strcmp(env, "ELPA_2STAGE_REAL_SVE512_BLOCK2") == 0)
            kernel_id=ELPA_2STAGE_REAL_SVE512_BLOCK2;
        else if(strcmp(env, "ELPA_2STAGE_REAL_SVE512_BLOCK4") == 0)
            kernel_id=ELPA_2STAGE_REAL_SVE512_BLOCK4;
        else if(strcmp(env, "ELPA_2STAGE_REAL_SVE512_BLOCK6") == 0)
            kernel_id=ELPA_2STAGE_REAL_SVE512_BLOCK6;
        else if(strcmp(env, "ELPA_2STAGE_REAL_GENERIC_SIMPLE_BLOCK4") == 0)
            kernel_id=ELPA_2STAGE_REAL_GENERIC_SIMPLE_BLOCK4;
        else if(strcmp(env, "ELPA_2STAGE_REAL_GENERIC_SIMPLE_BLOCK6") == 0)
            kernel_id=ELPA_2STAGE_REAL_GENERIC_SIMPLE_BLOCK6;
        else
            kernel_id=ELPA_2STAGE_REAL_GENERIC;
    }
    else
    {
        int cpuflag=read_cpuflag();
        switch (cpuflag)
        {
            case 4:
                kernel_id=ELPA_2STAGE_REAL_AVX512_BLOCK6;
                break;
            case 3:
                kernel_id=ELPA_2STAGE_REAL_AVX2_BLOCK6;
                break;
            case 2:
                kernel_id=ELPA_2STAGE_REAL_AVX_BLOCK6;
                break;
            case 1:
                kernel_id=ELPA_2STAGE_REAL_SSE_BLOCK6;
                break;
            default:
                kernel_id=ELPA_2STAGE_REAL_GENERIC_SIMPLE_BLOCK6;
                break;
        }
    }
    return kernel_id;
}

int ELPA_Solver::read_complex_kernel()
{
    int kernel_id;
    if (const char* env = getenv("ELPA_DEFAULT_complex_kernel"))
    {
        if(strcmp(env, "ELPA_2STAGE_COMPLEX_GENERIC_SIMPLE") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_GENERIC_SIMPLE;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_BGP") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_BGP;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_BGQ") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_BGQ;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_SSE_ASSEMBLY") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_SSE_ASSEMBLY;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_SSE_BLOCK1") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_SSE_BLOCK1;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_SSE_BLOCK2") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_SSE_BLOCK2;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_AVX_BLOCK1") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_AVX_BLOCK1;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_AVX_BLOCK2") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_AVX_BLOCK2;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_AVX2_BLOCK1") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_AVX2_BLOCK1;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_AVX2_BLOCK2") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_AVX2_BLOCK2;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_AVX512_BLOCK1") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_AVX512_BLOCK1;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_AVX512_BLOCK2") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_AVX512_BLOCK2;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_SVE128_BLOCK1") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_SVE128_BLOCK1;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_SVE128_BLOCK2") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_SVE128_BLOCK2;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_SVE256_BLOCK1") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_SVE256_BLOCK1;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_SVE256_BLOCK2") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_SVE256_BLOCK2;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_SVE512_BLOCK1") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_SVE512_BLOCK1;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_SVE512_BLOCK2") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_SVE512_BLOCK2;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_NEON_ARCH64_BLOCK1") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_NEON_ARCH64_BLOCK1;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_NEON_ARCH64_BLOCK2") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_NEON_ARCH64_BLOCK2;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_NVIDIA_GPU") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_NVIDIA_GPU;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_AMD_GPU") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_AMD_GPU;
        else if(strcmp(env, "ELPA_2STAGE_COMPLEX_INTEL_GPU") == 0)
            kernel_id=ELPA_2STAGE_COMPLEX_INTEL_GPU;
        else
            kernel_id=ELPA_2STAGE_COMPLEX_GENERIC;
    }
    else
    {
        int cpuflag=read_cpuflag();
        switch (cpuflag)
        {
            case 4:
                kernel_id=ELPA_2STAGE_COMPLEX_AVX512_BLOCK2;
                break;
            case 3:
                kernel_id=ELPA_2STAGE_COMPLEX_AVX2_BLOCK2;
                break;
            case 2:
                kernel_id=ELPA_2STAGE_COMPLEX_AVX_BLOCK2;
                break;
            case 1:
                kernel_id=ELPA_2STAGE_COMPLEX_SSE_BLOCK2;
                break;
            default:
                kernel_id=ELPA_2STAGE_COMPLEX_GENERIC_SIMPLE;
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

// calculate cholesky factorization of matrix B
// B = U^T * U
// and calculate the inverse: U^{-1}
// input:
//      B: the right side matrix of generalized eigen equation
// output:
//      DecomposedState: the method used to decompose right matrix
//                  1 or 2: use cholesky decomposing, B=U^T*U
//                  3: if cholesky decomposing failed, use diagonalizing
//      B: decomposed right matrix
//           when DecomposedState is 1 or 2, B is U^{-1}
//           when DecomposedState is 3, B is B^{-1/2}
int ELPA_Solver::decomposeRightMatrix(double* B, double* EigenValue, double* EigenVector, int& DecomposedState)
{
    int info=0;
    int allinfo=0;
    if(loglevel>0)
    {
        std::stringstream outlog;
        outlog.str("");
        outlog<<"DEBUG: Process "<<myid<<" Enter decomposeRightMatrix"<<std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout<<outlog.str();
    }
    // first try cholesky decomposing
    if(nFull<CHOLESKY_CRITICAL_SIZE)
    {
        DecomposedState=1;
        info=Cpdpotrf('U', nFull, B, desc);
        MPI_Allreduce(&info, &allinfo, 1, MPI_INT, MPI_MAX, comm);
        if(allinfo != 0) //pdpotrf fail, try elpa_cholesky_real
        {
            DecomposedState=2;
            elpa_cholesky_d(handle_pool[handle_id], B, &info);
            MPI_Allreduce(&info, &allinfo, 1, MPI_INT, MPI_MAX, comm);
        }
    } else
    {
        DecomposedState=2;
        elpa_cholesky_d(handle_pool[handle_id], B, &info);
        MPI_Allreduce(&info, &allinfo, 1, MPI_INT, MPI_MAX, comm);
        if(allinfo != 0)
        {
            DecomposedState=1;
            info=Cpdpotrf('U', nFull, B, desc);
            MPI_Allreduce(&info, &allinfo, 1, MPI_INT, MPI_MAX, comm);
        }
    }

    if(allinfo!=0) // if cholesky decomposing failed, try diagonalize
    {
        DecomposedState=3;
        elpa_eigenvectors_all_host_arrays_d(handle_pool[handle_id], B, EigenValue, EigenVector, &info);
        MPI_Allreduce(&info, &allinfo, 1, MPI_INT, MPI_MAX, comm);
        // calculate B^{-1/2}_{i,j}=\sum_k q_{i,k}*ev_k^{-1/2}*q_{j,k} and put to b, which will be b^-1/2
        // calculate q*ev^{-1/2} and put to work
        for(int i=0; i<nacols; ++i)
        {
            int eidx=globalIndex(i, nblk, npcols, mypcol);
            //double ev_sqrt=1.0/sqrt(ev[eidx]);
            double ev_sqrt=EigenValue[eidx]>DBL_MIN?1.0/sqrt(EigenValue[eidx]):0;
            for(int j=0; j<narows; ++j)
                dwork[i*lda+j]=EigenVector[i*lda+j]*ev_sqrt;
        }

        // calculate work*q=q*ev^{-1/2}*q^T, put to b, which is B^{-1/2}
        char transa='N', transb='T';
        double alpha=1.0, beta=0.0;
        int isrc=1, jsrc=1;
        pdgemm_(&transa, &transb, &nFull, &nFull, &nFull,
                &alpha, dwork,          &isrc, &jsrc, desc,
                        EigenVector,    &isrc, &jsrc, desc,
                &beta,  B,              &isrc, &jsrc, desc);
    }
    else // calculate U^{-1}
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
        elpa_invert_trm_d(handle_pool[handle_id], B, &info);
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
    double _Complex* z = reinterpret_cast<double _Complex*>(zwork);

    int info=0;
    int allinfo=0;
    if(loglevel>0)
    {
        std::stringstream outlog;
        outlog.str("");
        outlog<<"DEBUG: Process "<<myid<<" Enter decomposeRightMatrix"<<std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout<<outlog.str();
    }
    // first try cholesky decomposing
    if(nFull<CHOLESKY_CRITICAL_SIZE)
    {
        DecomposedState=1;
        char uplo='U';
        int isrc=1, jsrc=1;
        pzpotrf_(&uplo, &nFull, b, &isrc, &jsrc, desc, &info);
        MPI_Allreduce(&info, &allinfo, 1, MPI_INT, MPI_MAX, comm);
        if(allinfo != 0) //pdpotrf fail, try elpa_cholesky_real
        {
            DecomposedState=2;
            elpa_cholesky_dc(handle_pool[handle_id], b, &info);
            MPI_Allreduce(&info, &allinfo, 1, MPI_INT, MPI_MAX, comm);
        }
    } else
    {
        DecomposedState=2;
        elpa_cholesky_dc(handle_pool[handle_id], b, &info);
        MPI_Allreduce(&info, &allinfo, 1, MPI_INT, MPI_MAX, comm);
        if(allinfo != 0)
        {
            DecomposedState=1;
            char uplo='U';
            int isrc=1, jsrc=1;
            pzpotrf_(&uplo, &nFull, b, &isrc, &jsrc, desc, &info);
            MPI_Allreduce(&info, &allinfo, 1, MPI_INT, MPI_MAX, comm);
        }
    }

    // if cholesky decomposing failed, try diagonalize
    if(allinfo!=0)
    {
        DecomposedState=3;
        elpa_eigenvectors_all_host_arrays_dc(handle_pool[handle_id], b,
                             EigenValue, q, &info);
        MPI_Allreduce(&info, &allinfo, 1, MPI_INT, MPI_MAX, comm);
        // calculate B^{-1/2}_{i,j}=\sum_k q_{i,k}*ev_k^{-1/2}*q_{j,k} and put to b, which will be b^-1/2
        // calculate q*ev and put to work
        for(int i=0; i<nacols; ++i)
        {
            int eidx=globalIndex(i, nblk, npcols, mypcol);
            //double ev_sqrt=1.0/sqrt(ev[eidx]);
            double ev_sqrt=EigenValue[eidx]>DBL_MIN?1.0/sqrt(EigenValue[eidx]):0;
            for(int j=0; j<narows; ++j)
                zwork[i*lda+j]=EigenVector[i*lda+j]*ev_sqrt;
        }

        // calculate qevq=qev*q^T, put to b, which is B^{-1/2}
        char transa='N', transb='C';
        double alpha=1.0, beta=0.0;
        int isrc=1, jsrc=1;
        pzgemm_(&transa, &transb, &nFull, &nFull, &nFull,
                &alpha, z, &isrc, &jsrc, desc,
                        q, &isrc, &jsrc, desc,
                &beta,  b, &isrc, &jsrc, desc);
    }
    return allinfo;
}

int ELPA_Solver::composeEigenVector(int DecomposedState, complex<double>* B, complex<double>* EigenVector)
{
    double _Complex* b = reinterpret_cast<double _Complex*>(B);
    double _Complex* q = reinterpret_cast<double _Complex*>(EigenVector);
    double _Complex* z = reinterpret_cast<double _Complex*>(zwork);

    if(DecomposedState==1 || DecomposedState==2)
    {
        // transform the eigenvectors to original general equation, let U^-1*q, and put to q
        char side='L', uplo='U', transa='N', diag='N';
        double alpha=1.0;
        int isrc=1, jsrc=1;
        pztrmm_(&side, &uplo, &transa,  &diag, &nFull, &nFull,
                &alpha, b, &isrc, &jsrc, desc,
                        q, &isrc, &jsrc, desc);
    } else
    {
        // transform the eigenvectors to original general equation, let b^T*q, and put to q
        char transa='T', transb='N';
        double alpha=1.0, beta=0.0;
        int isrc=1, jsrc=1;
        pzgemm_(&transa, &transb, &nFull, &nFull, &nFull,
                &alpha, b, &isrc, &jsrc, desc,
                        z, &isrc, &jsrc, desc,
                &beta,  q, &isrc, &jsrc, desc);

    }
    return 0;
}


//int ELPA_Solver::timer(int myid, const char function[], const char step[], double &t0)
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

// calculate error of sum_i{R(:,i)*R(:,i)}, where R = A*V - V*D
// V: eigenvector matrix
// D: Diaganal matrix of eigenvalue
// maxError: maximum error
// meanError: mean error
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
        maxError=maxError>E?maxError:E;
    }
    meanError=sumError/nFull;
    // global mean and max Error
    delete[] D;
}

// calculate remains of A*V - B*V*D
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

void ELPA_Solver::outputParameters()
{
    stringstream outlog;
    outlog.str("");
    outlog<<"myid "<<myid<<": comm id(in FORTRAN):"<<MPI_Comm_c2f(comm)<<endl;
    outlog<<"myid "<<myid<<": nprows: "<<nprows<<" npcols: "<<npcols<<endl;
    outlog<<"myid "<<myid<<": myprow: "<<myprow<<" mypcol: "<<mypcol<<endl;
    outlog<<"myid "<<myid<<": nFull: "<<nFull<<" nev: "<<nev<<endl;
    outlog<<"myid "<<myid<<": narows: "<<narows<<" nacols: "<<nacols<<endl;
    outlog<<"myid "<<myid<<": blacs parameters setting"<<endl;
    outlog<<"myid "<<myid<<": blacs ctxt:"<<cblacs_ctxt<<endl;
    outlog<<"myid "<<myid<<": desc: ";
    for(int i=0; i<9; ++i) outlog<<desc[i]<<" ";
    outlog<<endl;
    outlog<<"myid "<<myid<<": nblk: "<<nblk<<" lda: "<<lda<<endl;

    outlog<<"myid "<<myid<<": useQR: "<<useQR<<" wantDebug: "<<wantDebug<<endl;

    cout<<outlog.str();
}
