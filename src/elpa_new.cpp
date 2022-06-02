#include <complex>
#include <map>
#include <regex>
#include <fstream>
#include <cfloat>
#include <cstring>
#include <iostream>
#include <sstream>
#include <unistd.h>

#include <mpi.h>

#include "elpa_new.h"
#include "elpa_solver.h"

#include "my_math.hpp"
#include "utils.h"

using namespace std;

map<int, elpa_t> NEW_ELPA_HANDLE_POOL;

ELPA_Solver::ELPA_Solver(bool isReal, MPI_Comm comm, int nev,
                         int narows, int nacols, int* desc)
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
    NEW_ELPA_HANDLE_POOL[handle_id]=handle;

    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "na", nFull, &error);
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "nev", nev, &error);
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "local_nrows", narows, &error);
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "local_ncols", nacols, &error);
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "nblk", nblk, &error);
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "mpi_comm_parent", MPI_Comm_c2f(comm), &error);
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "process_row", myprow, &error);
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "process_col", mypcol, &error);

    error = elpa_setup(NEW_ELPA_HANDLE_POOL[handle_id]);
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "solver", ELPA_SOLVER_2STAGE, &error);
    if(isReal)
        elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "real_kernel", kernel_id, &error);
    else
        elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "complex_kernel", kernel_id, &error);
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
    MPI_Comm_rank(comm, &myid);
    Cblacs_gridinfo(cblacs_ctxt, &nprows, &npcols, &myprow, &mypcol);
    allocate_work(isReal);

    int error;
    static map<int, elpa_t> NEW_ELPA_HANDLE_POOL;
    static int total_handle;

    elpa_init(20210430);

    handle_id = ++total_handle;
    elpa_t handle;
    handle=elpa_allocate(&error);
    NEW_ELPA_HANDLE_POOL[handle_id]=handle;

    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "na", nFull, &error);
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "nev", nev, &error);
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "local_nrows", narows, &error);
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "local_ncols", nacols, &error);
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "nblk", nblk, &error);
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "mpi_comm_parent", MPI_Comm_c2f(comm), &error);
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "process_row", myprow, &error);
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "process_col", mypcol, &error);
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "blacs_context", cblacs_ctxt, &error);
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "solver", ELPA_SOLVER_2STAGE, &error);
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "debug", wantDebug, &error);
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "qr", useQR, &error);
    if(isReal)
        elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "real_kernel", kernel_id, &error);
    else
        elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "complex_kernel", kernel_id, &error);
}

void ELPA_Solver::setLoglevel(int loglevel)
{
    int error;
    this->loglevel=loglevel;
    if(loglevel>=2)
    {
        wantDebug=1;
        elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "verbose", 1, &error);
        elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "debug", wantDebug, &error);

        outputParameters();
    }
}

void ELPA_Solver::setKernel(bool isReal, int kernel)
{
    int error;
    if(isReal)
        elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "real_kernel", kernel_id, &error);
    else
        elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "complex_kernel", kernel_id, &error);
}

void ELPA_Solver::setKernel(bool isReal, int kernel, int useQR)
{
    int error;
    elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "qr", useQR, &error);
    if(isReal)
        elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "real_kernel", kernel_id, &error);
    else
        elpa_set_integer(NEW_ELPA_HANDLE_POOL[handle_id], "complex_kernel", kernel_id, &error);
}

void ELPA_Solver::exit()
{
    delete[] dwork;
    delete[] zwork;
    int error;
    elpa_deallocate(NEW_ELPA_HANDLE_POOL[handle_id], &error);
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

void ELPA_Solver::timer(int myid, const char function[], const char step[], double &t0)
{
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
    outlog<<"myid "<<myid<<": useQR: "<<useQR<<" kernel:"<<kernel_id<<endl;;
    outlog<<"myid "<<myid<<": wantDebug: "<<wantDebug<<" loglevel: "<<loglevel<<endl;
    cout<<outlog.str();
}
