#include <mpi.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <complex>
#include "my_math.hpp"
#include "utils.h"

void initBlacsGrid(int loglevel, MPI_Comm comm, int nFull, int nblk,
                         int& blacs_ctxt, int& narows, int& nacols, int desc[])
{
    std::stringstream outlog;
    char BLACS_LAYOUT='C';
    int ISRCPROC=0; // fortran array starts from 1
    int nprows, npcols;
    int myprow, mypcol;
    int nprocs, myid;
    int info;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &myid);
    // set blacs parameters
    for(npcols=int(sqrt(double(nprocs))); npcols>=2; --npcols)
    {
        if(nprocs%npcols==0) break;
    }
    nprows=nprocs/npcols;
    if( (loglevel>0 && myid==0) || loglevel>1)
    {
        outlog.str("");
        outlog<<"myid "<<myid<<": nprows: "<<nprows<<" ; npcols: "<<npcols<<std::endl;
        std::cout<<outlog.str();
    }

    //int comm_f = MPI_Comm_c2f(comm);
    blacs_ctxt=Csys2blacs_handle(comm);
    Cblacs_gridinit(&blacs_ctxt, &BLACS_LAYOUT, nprows, npcols);
    if( (loglevel>0 && myid==0) || loglevel>1)
    {
        outlog.str("");
        outlog<<"myid "<<myid<<": Cblacs_gridinit done, blacs_ctxt: "<<blacs_ctxt<<std::endl;
        std::cout<<outlog.str();
    }
    Cblacs_gridinfo(blacs_ctxt, &nprows, &npcols, &myprow, &mypcol);
    if( (loglevel>0 && myid==0) || loglevel>1)
    {
        int mypnum=Cblacs_pnum(blacs_ctxt, myprow, mypcol);
        int prow, pcol;
        Cblacs_pcoord(blacs_ctxt, myid, &prow, &pcol);
        outlog.str("");
        outlog<<"myid "<<myid<<": myprow: "<<myprow<<" ;mypcol: "<<mypcol<<std::endl;
        outlog<<"myid "<<myid<<": mypnum: "<<mypnum<<std::endl;
        outlog<<"myid "<<myid<<": prow: "<<prow<<" ;pcol: "<<pcol<<std::endl;
        std::cout<<outlog.str();
    }

    narows=numroc_(&nFull, &nblk, &myprow, &ISRCPROC, &nprows);
    nacols=numroc_(&nFull, &nblk, &mypcol, &ISRCPROC, &npcols);
    descinit_(desc, &nFull, &nFull, &nblk, &nblk, &ISRCPROC, &ISRCPROC, &blacs_ctxt, &narows, &info);

    if( (loglevel>0 && myid==0) || loglevel>1)
    {
        outlog.str("");
        outlog<<"myid "<<myid<<": narows: "<<narows<<" nacols: "<<nacols<<std::endl;
        outlog<<"myid "<<myid<<": blacs parameters setting"<<std::endl;
        outlog<<"myid "<<myid<<": desc is: ";
        for(int i=0; i<9; ++i) outlog<<desc[i]<<" ";
        outlog<<std::endl;
        std::cout<<outlog.str();
    }
}

// load matrix from the file
void loadMatrix(const char FileName[], int nFull, double* a, int* desca, int blacs_ctxt)
{
    int nprows, npcols, myprow, mypcol;
    Cblacs_gridinfo(blacs_ctxt, &nprows, &npcols, &myprow, &mypcol);
    int myid=Cblacs_pnum(blacs_ctxt, myprow, mypcol);

    const int ROOT_PROC=0;
    std::ifstream matrixFile;
    if(myid == ROOT_PROC)
        matrixFile.open(FileName);

    double *b; // buffer
    const int MAX_BUFFER_SIZE=1e9; // max buffer size is 1GB

    int N=nFull;
    int M=std::max( 1, std::min( nFull, (int)(MAX_BUFFER_SIZE/nFull/sizeof(double) ) ) ); // at lease 1 row, max size 1GB
    if(myid == ROOT_PROC)
        b=new double[M*N];
    else
        b=new double[1];

    //set descb, which has all elements in the only block in the root process
    // block size is M x N, so all elements are in the first process
    int descb[9]={1, blacs_ctxt, M, N, M, N, 0, 0, M};

    int ja=1, ib=1, jb=1;
    for(int ia=1; ia<nFull; ia+=M)
    {
        int thisM=std::min(M, nFull-ia+1); // nFull-ia+1 is number of the last few rows to be read from file
        // read from the file
        if(myid == ROOT_PROC)
        {
            for(int i=0; i<thisM; ++i)
            {
                for(int j=0; j<N; ++j)
                {
                    matrixFile>>b[i+j*M];
                }
            }
        }
        // gather data rows by rows from all processes
        Cpdgemr2d(thisM, N, b, ib, jb, descb, a, ia, ja, desca, blacs_ctxt);
    }

    if(myid == ROOT_PROC)
        matrixFile.close();

    delete[] b;
}

void saveLocalMatrix(const char filePrefix[], int narows, int nacols, double* a)
{
    using namespace std;
    char FileName[80];
    int myid;
    ofstream matrixFile;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    sprintf(FileName, "%s_%3.3d.dat", filePrefix, myid);
    matrixFile.open(FileName);
    matrixFile.flags(std::ios_base::scientific);
    matrixFile.precision(17);
    matrixFile.width(24);
    for(int i=0; i<narows; ++i)
    {
        for(int j=0; j<nacols; ++j)
        {
            matrixFile<<a[i+j*narows]<<" ";
        }
        matrixFile<<std::endl;
    }
    matrixFile.close();
}

int loadMatrix_v1(const char FileName[], int nFull, int nblk, int narows, int nacols, double* a, int lda, int blacs_ctxt)
{
    using namespace std;
    int nprows, npcols;
    int myprow, mypcol;
    double *loadBuffer=NULL, *sendBuffer=NULL, *recvBuffer=NULL;
    int ldBuffer;
    int nRow, nRowNow;
    int recvprow, recvpcol;
    int sendrow, recvrow;
    //int recvproc;
    int SENDPROW=0, SENDPCOL=0;
    int myid, nprocs;
    ifstream dataFile;
    stringstream outlog;

    Cblacs_pinfo(&myid, &nprocs);
    Cblacs_gridinfo(blacs_ctxt, &nprows, &npcols, &myprow, &mypcol);

    ldBuffer=ceil(double(nFull)/double(nblk*npcols))*nblk;

    if(myprow==SENDPROW && mypcol==SENDPCOL)
    {
        dataFile.open(FileName);
        loadBuffer=new double[nblk*nFull];
        sendBuffer=new double[nblk*ldBuffer];
    }else
    {
        recvBuffer=new double[nblk*ldBuffer];
    }

    nRow=nFull; // number of left rows to be loaded
    nRowNow=nblk; // number of current loading rows
    sendrow=0;
    do
    {
        nRowNow=nRow<nblk?nRow:nblk;
        if(myprow==SENDPROW && mypcol==SENDPCOL)
        {
            for(int i=0; i<nRowNow; ++i)
            {
                for(int j=0; j<nFull; ++j)
                {
                    dataFile>>loadBuffer[i*nFull+j];
                    //++iBuffer;
                }
                // cout<<endl;
            }
        }

        recvrow=localIndex(sendrow, nblk, nprows, recvprow);
        for(recvpcol=0; recvpcol<npcols; ++recvpcol)
        {
            if(myprow==SENDPROW && mypcol==SENDPCOL) // copy dato from loadBuffer to sub-matrix a directly
            {
                if(myprow==recvprow && mypcol==recvpcol)
                {
                    //for(int i=0; i<narows; ++i)
                    for(int j=0; j<nacols; ++j)
                    {
                        int jg=globalIndex(j, nblk, npcols, recvpcol);
                        const int BufferRow=nRowNow;
                        const double* pBuffer=&loadBuffer[jg];
                        const int inc_Buffer=nFull;
                        const int inc_a=1;
                        dcopy_(&BufferRow, pBuffer, &inc_Buffer, &a[j*narows+recvrow], &inc_a);
                    }
                }
                else// copy data from loadBuffer to sendBuffer
                {
                    for(int j=0; j<ldBuffer; ++j)
                    {
                        int jg=globalIndex(j, nblk, npcols, recvpcol);
                        if(jg>=nFull) break;
                        const int BufferRow=nRowNow;
                        const double* pBuffer=&loadBuffer[jg];
                        const int inc_Buffer=nFull;
                        const int inc_a=1;
                        dcopy_(&BufferRow, pBuffer, &inc_Buffer, &sendBuffer[j*nRowNow], &inc_a);
                    }
                    Cdgesd2d(blacs_ctxt, ldBuffer, nRowNow, sendBuffer, nRowNow, recvprow, recvpcol);
                }
            }
            else if(myprow==recvprow && mypcol==recvpcol)// receive sub-matrix data from SENDPROC
            {
                Cdgerv2d(blacs_ctxt, ldBuffer, nRowNow, recvBuffer, nRowNow, SENDPROW, SENDPCOL);
                const int BufferRow=nRowNow;
                const int inc=1;
                for(int j=0; j<nacols; ++j)
                {
                    const double* pBuffer=&recvBuffer[j*nRowNow];
                    dcopy_(&BufferRow, pBuffer, &inc, &a[j*narows+recvrow], &inc);
                }
            }
        }

        nRow-=nRowNow;
        sendrow+=nRowNow;
    } while(nRow>0);


    if(myprow==SENDPROW && mypcol==SENDPCOL)
    {
        dataFile.close();
        delete[] loadBuffer;
        delete[] sendBuffer;
    }else
    {
        delete[] recvBuffer;
    }
    return 0;
}

// use pdgemr2d to collect matrix from all processes to root process
// and save to one completed matrix file
void saveMatrix(const char FileName[], int nFull, double* a, int* desca, int blacs_ctxt)
{
    int nprows, npcols, myprow, mypcol;
    Cblacs_gridinfo(blacs_ctxt, &nprows, &npcols, &myprow, &mypcol);
    int myid=Cblacs_pnum(blacs_ctxt, myprow, mypcol);

    const int ROOT_PROC=0;
    std::ofstream matrixFile;
    if(myid == ROOT_PROC) // setup saved matrix format
    {
        matrixFile.open(FileName);
        matrixFile.flags(std::ios_base::scientific);
        matrixFile.precision(17);
        matrixFile.width(24);
    }

    double *b; // buffer
    const int MAX_BUFFER_SIZE=1e9; // max buffer size is 1GB

    int N=nFull;
    int M=std::max(1, std::min(nFull, (int)(MAX_BUFFER_SIZE/nFull/sizeof(double) ) ) ); // at lease 1 row, max size 1GB
    if(myid == ROOT_PROC)
        b=new double[M*N];
    else
        b=new double[1];

    //set descb, which has all elements in the only block in the root process
    int descb[9]={1, blacs_ctxt, M, N, M, N, 0, 0, M};

    int ja=1, ib=1, jb=1;
    for(int ia=1; ia<nFull; ia+=M)
    {
        int thisM=std::min(M, nFull-ia+1); // nFull-ia+1 is the last few row to be saved
        // gather data rows by rows from all processes
        Cpdgemr2d(thisM, N, a, ia, ja, desca, b, ib, jb, descb, blacs_ctxt);
        // write to the file
        if(myid == ROOT_PROC)
        {
            for(int i=0; i<thisM; ++i)
            {
                for(int j=0; j<N; ++j)
                {
                    matrixFile<<b[i+j*M]<<" ";
                }
                matrixFile<<std::endl;
            }
        }
    }

    if(myid == ROOT_PROC)
        matrixFile.close();

    delete[] b;
}

// load matrix from the file
void loadMatrix(const char FileName[], int nFull, std::complex<double>* a, int* desca, int blacs_ctxt)
{
    int nprows, npcols, myprow, mypcol;
    Cblacs_gridinfo(blacs_ctxt, &nprows, &npcols, &myprow, &mypcol);
    int myid=Cblacs_pnum(blacs_ctxt, myprow, mypcol);

    const int ROOT_PROC=0;
    std::ifstream matrixFile;
    if(myid == ROOT_PROC)
        matrixFile.open(FileName);

    std::complex<double> *b; // buffer
    const int MAX_BUFFER_SIZE=1e9; // max buffer size is 1GB

    int N=nFull;
    int M=std::max( 1, std::min( nFull, (int)(MAX_BUFFER_SIZE/nFull/(2*sizeof(double)) ) ) ); // at lease 1 row, max size 1GB
    if(myid == ROOT_PROC)
        b=new std::complex<double>[M*N];
    else
        b=new std::complex<double>[1];

    //set descb, which has all elements in the only block in the root process
    // block size is M x N, so all elements are in the first process
    int descb[9]={1, blacs_ctxt, M, N, M, N, 0, 0, M};

    int ja=1, ib=1, jb=1;
    for(int ia=1; ia<nFull; ia+=M)
    {
        int thisM=std::min(M, nFull-ia+1); // nFull-ia+1 is number of the last few rows to be read from file
        // read from the file
        if(myid == ROOT_PROC)
        {
            for(int i=0; i<thisM; ++i)
            {
                for(int j=0; j<N; ++j)
                {
                    matrixFile>>b[i+j*M];
                }
            }
        }
        // gather data rows by rows from all processes
        Cpzgemr2d(thisM, N, b, ib, jb, descb, a, ia, ja, desca, blacs_ctxt);
    }

    if(myid == ROOT_PROC)
        matrixFile.close();

    delete[] b;
}

void saveLocalMatrix(const char filePrefix[], int narows, int nacols, std::complex<double>* a)
{
    using namespace std;
    char FileName[80];
    int myid;
    ofstream matrixFile;

    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    sprintf(FileName, "%s_%3.3d.dat", filePrefix, myid);
    matrixFile.open(FileName);
    matrixFile.flags(std::ios_base::scientific);
    matrixFile.precision(17);
    matrixFile.width(24);
    for(int i=0; i<narows; ++i)
    {
        for(int j=0; j<nacols; ++j)
        {
            matrixFile<<a[i+j*narows]<<" ";
        }
        matrixFile<<std::endl;

    }
    matrixFile.close();
}


// use pzgemr2d to collect matrix from all processes to root process
// and save to one completed matrix file
void saveMatrix(const char FileName[], int nFull, std::complex<double>* a, int* desca, int blacs_ctxt)
{
    int nprows, npcols, myprow, mypcol;
    Cblacs_gridinfo(blacs_ctxt, &nprows, &npcols, &myprow, &mypcol);
    int myid=Cblacs_pnum(blacs_ctxt, myprow, mypcol);

    const int ROOT_PROC=0;
    std::ofstream matrixFile;
    if(myid == ROOT_PROC) // setup saved matrix format
    {
        matrixFile.open(FileName);
        matrixFile.flags(std::ios_base::scientific);
        matrixFile.precision(17);
        matrixFile.width(24);
    }

    std::complex<double> *b; // buffer
    const int MAX_BUFFER_SIZE=1e9; // max buffer size is 1GB

    int N=nFull;
    int M=std::max(1, std::min(nFull, (int)(MAX_BUFFER_SIZE/nFull/sizeof(double) ) ) ); // at lease 1 row, max size 1GB
    if(myid == ROOT_PROC)
        b=new std::complex<double>[M*N];
    else
        b=new std::complex<double>[1];

    //set descb, which has all elements in the only block in the root process
    int descb[9]={1, blacs_ctxt, M, N, M, N, 0, 0, M};

    int ja=1, ib=1, jb=1;
    for(int ia=1; ia<nFull; ia+=M)
    {
        int transM=std::min(M, nFull-ia+1); // nFull-ia+1 is the last few row to be saved
        // gather data rows by rows from all processes
        Cpzgemr2d(transM, N, a, ia, ja, desca, b, ib, jb, descb, blacs_ctxt);
        // write to the file
        if(myid == ROOT_PROC)
        {
            for(int i=0; i<transM; ++i)
            {
                for(int j=0; j<N; ++j)
                {
                    matrixFile<<b[i+j*M]<<" ";
                }
                matrixFile<<std::endl;
            }
        }
    }

    if(myid == ROOT_PROC)
        matrixFile.close();

    delete[] b;
}


int loadMatrix_v1(const char FileName[], int nFull, int nblk, int narows, int nacols, std::complex<double>* a, int lda, int blacs_ctxt)
{
    using namespace std;
    int nprows, npcols;
    int myprow, mypcol;
    double _Complex *aa=reinterpret_cast<double _Complex*>(a);
    double _Complex *loadBuffer, *sendBuffer, *recvBuffer;

    int ldBuffer;
    int nRow, nRowNow;
    int recvprow, recvpcol;
    int sendrow, recvrow;
    //int recvproc;
    int SENDPROW=0, SENDPCOL=0;
    int myid, nprocs;
    ifstream dataFile;
    stringstream outlog;

    Cblacs_pinfo(&myid, &nprocs);
    Cblacs_gridinfo(blacs_ctxt, &nprows, &npcols, &myprow, &mypcol);

    ldBuffer=ceil(double(nFull)/double(nblk*npcols))*nblk;
    if(myprow==SENDPROW && mypcol==SENDPCOL)
    {
        dataFile.open(FileName);
        loadBuffer=new double _Complex[nblk*nFull];
        sendBuffer=new double _Complex[nblk*ldBuffer];
        recvBuffer=new double _Complex[1];
    }else
    {
        loadBuffer=new double _Complex[1];
        sendBuffer=new double _Complex[1];
        recvBuffer=new double _Complex[nblk*ldBuffer];
    }

    std::complex <double> *loadBufferCPP=reinterpret_cast<std::complex <double>*>(loadBuffer);
    nRow=nFull; // number of left rows to be loaded
    nRowNow=nblk; // number of current loading rows
    sendrow=0;
    do
    {
        nRowNow=nRow<nblk?nRow:nblk;
        if(myprow==SENDPROW && mypcol==SENDPCOL)
        {
            for(int i=0; i<nRowNow; ++i)
            {
                for(int j=0; j<nFull; ++j)
                {
                    dataFile>>loadBufferCPP[i*nFull+j];
                    //dataFile>>loadBufferCPP[i+j*nRowNow];
                    //++iBuffer;
                }
                // cout<<endl;
            }
        }

        recvrow=localIndex(sendrow, nblk, nprows, recvprow);
        for(recvpcol=0; recvpcol<npcols; ++recvpcol)
        {
            if(myprow==SENDPROW && mypcol==SENDPCOL) // copy dato from loadBuffer to sub-matrix a directly
            {
                if(myprow==recvprow && mypcol==recvpcol)
                {
                    //for(int i=0; i<narows; ++i)
                    for(int j=0; j<nacols; ++j)
                    {
                        int jg=globalIndex(j, nblk, npcols, recvpcol);
                        int inc=1;
                        zcopy_(&nRowNow, &loadBuffer[jg], &nFull, &aa[j*narows+recvrow], &inc);
                    }
                }
                else// copy data from loadBuffer to sendBuffer and change from row first to column first
                {
                    for(int j=0; j<ldBuffer; ++j)
                    {
                        int jg=globalIndex(j, nblk, npcols, recvpcol);
                        if(jg>=nFull) break;
                        int inc=1;
                        zcopy_(&nRowNow, &loadBuffer[jg], &nFull, &sendBuffer[j*nRowNow], &inc);
                    }
                    Czgesd2d(blacs_ctxt, ldBuffer, nRowNow, sendBuffer, nRowNow, recvprow, recvpcol);
                }
            }
            else if(myprow==recvprow && mypcol==recvpcol)// receive sub-matrix data from SENDPROC
            {
                Czgerv2d(blacs_ctxt, ldBuffer, nRowNow, recvBuffer, nRowNow, SENDPROW, SENDPCOL);
                int inc=1;
                for(int j=0; j<nacols; ++j)
                {
                    zcopy_(&nRowNow, &recvBuffer[j*nRowNow], &inc, &aa[j*narows+recvrow], &inc);
                }
            }
        }

        nRow-=nRowNow;
        sendrow+=nRowNow;
    } while(nRow>0);


    if(myprow==SENDPROW && mypcol==SENDPCOL)
    {
        dataFile.close();
    }

    delete[] loadBuffer;
    delete[] sendBuffer;
    delete[] recvBuffer;
    return 0;
}
