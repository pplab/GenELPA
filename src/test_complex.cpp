//------------------------------------>8======================================
//  Copyright (c) 2016, Yu Shen (shenyu@ustc.edu.cn)
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//      * Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//      * Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//      * Neither the name of the <organization> nor the
//        names of its contributors may be used to endorse or promote products
//        derived from this software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT Yu Shen SHALL BE LIABLE FOR ANY
//  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//====================================8<----------------------------------------

#include <mpi.h>
#include <complex>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cstring>
extern "C"
{
    #include "blas.h"
    #include "pblas.h"
    #include "Cblacs.h"
    #include "scalapack.h"
    #include "my_elpa.h"
}
#include "GenELPA.h"
#include "utils.h"

using namespace std;

int main(int argc, char** argv)
{
    int nFull, nev, nblk, ntest, loglevel;
    int myid, nprocs, myprow, nprows, mypcol, npcols;
    int my_blacs_ctxt;
    int info;
    int MPIROOT=0;
    int ISRCPROC=0;
    char BLACS_LAYOUT='R';
    int narows, nacols;
    complex<double> *H, *S, *a, *b, *q, *work;
    
    double *ev;
    int desc[9];

    bool wantDebug=true;
    bool wantEigenVector=true;
    stringstream outlog;
    char filePrefix[40];
    char filename[40];

    clock_t t0, t1;

    // set mpi enviorment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);

    //load parameters
    if(myid==MPIROOT)
    {
        fstream inputFile("INPUT2");
        if(!inputFile)
        {
            cout<<"Cannot open INPUT file"<<endl;
            MPI_Finalize();
            return 1;
        }
        inputFile>>nFull>>nev>>nblk>>ntest>>loglevel;
        inputFile.close();
        usleep(20);
        outlog.str("");
        outlog<<"parameters loaded: "<<nFull<<" "<<nev<<" "<<nblk<<" "<<ntest<<" "<<loglevel<<endl;
        cout<<outlog.str();
    }
    
    MPI_Bcast(&nFull, 1, MPI_INT, MPIROOT, MPI_COMM_WORLD);
    MPI_Bcast(&nev, 1, MPI_INT, MPIROOT, MPI_COMM_WORLD);
    MPI_Bcast(&nblk, 1, MPI_INT, MPIROOT, MPI_COMM_WORLD);
    MPI_Bcast(&ntest, 1, MPI_INT, MPIROOT, MPI_COMM_WORLD);
    MPI_Bcast(&loglevel, 1, MPI_INT, MPIROOT, MPI_COMM_WORLD);

    if(loglevel>0)
    {
        outlog.str("");
        outlog<<"myid "<<myid<<": parameters synchonized";
        outlog<<" nFull: "<<nFull<<" nev: "<<nev<<" nblk: "<<nblk<<" loglevel: "<<loglevel<<endl;
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(myid*120);
        cout<<outlog.str();
    }

    // set blacs parameters
    for(npcols=int(sqrt(double(nprocs))); npcols>=2; --npcols)
    {
        if(nprocs%npcols==0) break;
    }
    nprows=nprocs/npcols;
    if(loglevel>0)
    {
        outlog.str("");
        outlog<<"myid "<<myid<<" nprows: "<<nprows<<" ; npcols: "<<npcols<<endl;
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(myid*120);
        cout<<outlog.str();
    }

    Cblacs_get(comm_f, 0, &my_blacs_ctxt);
    Cblacs_gridinit(&my_blacs_ctxt, &BLACS_LAYOUT, nprows, npcols);
    if(loglevel>0)
    {
        outlog.str("");
        outlog<<"myid "<<myid<<": Cblacs_gridinit done, my_blacs_ctxt: "<<my_blacs_ctxt<<endl;
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(myid*120);
        cout<<outlog.str();
    }
    Cblacs_gridinfo(my_blacs_ctxt, &nprows, &npcols, &myprow, &mypcol);
    if(loglevel>0)
    {
    	int mypnum=Cblacs_pnum(my_blacs_ctxt, myprow, mypcol);
    	int prow, pcol;
    	Cblacs_pcoord(my_blacs_ctxt, myid, &prow, &pcol);
        outlog.str("");
        outlog<<"myid "<<myid<<": myprow: "<<myprow<<" ;mypcol: "<<mypcol<<endl;
        outlog<<"myid "<<myid<<" ;mypnum: "<<mypnum<<endl;
        outlog<<"myid "<<myid<<" ;prow: "<<prow<<" ;pcol: "<<pcol<<endl;
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(myid*120);
        cout<<outlog.str();
    }

    narows=numroc_(&nFull, &nblk, &myprow, &ISRCPROC, &nprows);
    nacols=numroc_(&nFull, &nblk, &mypcol, &ISRCPROC, &npcols);
    descinit_(desc, &nFull, &nFull, &nblk, &nblk, &ISRCPROC, &ISRCPROC, &my_blacs_ctxt, &narows, &info);

    if(loglevel>0)
    {
        outlog.str("");
        outlog<<"myid "<<myid<<": narows: "<<narows<<" nacols: "<<nacols<<endl;
        outlog<<"myid "<<myid<<": blacs parameters setting"<<endl;
        outlog<<"myid "<<myid<<": desc is: ";
        for(int i=0; i<9; ++i) outlog<<desc[i]<<" ";
        outlog<<endl;
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(myid*120);
        cout<<outlog.str();
    }

    //init main matrices
    int subMatrixSize=narows*nacols;
    H=new complex<double>[subMatrixSize];
    S=new complex<double>[subMatrixSize];
    a=new complex<double>[subMatrixSize];
    b=new complex<double>[subMatrixSize];

    double _Complex *HH=reinterpret_cast<double _Complex*>(H);
    double _Complex *SS=reinterpret_cast<double _Complex*>(S);
    double _Complex *aa=reinterpret_cast<double _Complex*>(a);
    double _Complex *bb=reinterpret_cast<double _Complex*>(b);
        
    q=new complex<double>[subMatrixSize];
    work=new complex<double>[subMatrixSize];
    ev=new double[nFull];
    for(int i=0; i<subMatrixSize; ++i) q[i]=0;
    for(int i=0; i<nFull; ++i) ev[i]=0;

    //load input matrices and distribute to all processes
    MPI_Barrier(MPI_COMM_WORLD);
    t0=clock();

	strcpy(filename,"H.dat");
    loadMatrix(filename, nFull, nblk, narows, nacols, H, nacols, my_blacs_ctxt);
    //initPositiveDefiniteMatrix(1, nFull, narows, nacols, H, ISRC, ISRC, desc);
    if(loglevel>2)
    {
    	if(myid==0)
    	{
	        outlog.str("");
	        outlog<<"matrix H loaded"<<endl;
        	cout<<outlog.str();
    	}
		strcpy(filename,"H_load");
    	saveMatrix(filename, narows, nacols, H);
    }
	strcpy(filename,"S.dat");
    loadMatrix(filename, nFull, nblk, narows, nacols, S, nacols, my_blacs_ctxt);
    //initPositiveDefiniteMatrix(2, nFull, narows, nacols, S, ISRC, ISRC, desc);
    if(loglevel>2)
    {
    	if(myid==0)
    	{
	        outlog.str("");
	        outlog<<"matrix S loaded"<<endl;
        	cout<<outlog.str();
    	}
		strcpy(filename,"S_load");
    	saveMatrix(filename, narows, nacols, S);
    }
    t1=clock();
    if(myid==MPIROOT)
    {
        usleep(120);
        cout<<"load file time:"<<(t1-t0)/CLOCKS_PER_SEC<<"s"<<endl;
    }

    int mpi_comm_rows, mpi_comm_cols;
    info=get_elpa_communicators(comm_f, myprow, mypcol, &mpi_comm_rows, &mpi_comm_cols);

    // start testing
    for (int i=0; i<ntest; ++i)
    {
        for(int elpaKernel=1; elpaKernel<=11; ++elpaKernel)
        {
            for(int method=1; method<=3; ++method)
            {
                int inc=1;

                zcopy_(&subMatrixSize, HH, &inc, aa, &inc);
                zcopy_(&subMatrixSize, SS, &inc, bb, &inc);
                usleep(300);
                if(loglevel>0)
                {
                    if(myid==0)
                    {
                        outlog.str("");
                        outlog<<"matrix a and b are prepared"<<endl
                                <<"prepare to run with elpaKernel: "<<elpaKernel
                                <<" and method: "<<method<<endl;
                        cout<<outlog.str();
                    }
                }

                MPI_Barrier(MPI_COMM_WORLD);
                t0=clock();
                // info=pdSolveGenEigen1(nev, nFull, narows, nacols, desc,
                //                       a, b, ev, q, work,
                //                       my_mpi_comm, my_blacs_ctxt, method,
                //                       wantEigenVector, wantDebug);

                info=pzDecomposeRightMatrix2(nFull, narows, nacols, desc,
                                            b, ev, q, work,
                                            MPI_COMM_WORLD, comm_f, mpi_comm_rows, mpi_comm_cols, method,
                                            elpaKernel);
                t1=clock();
                if(info==0)
                {
                    if(myid==MPIROOT)
                    {
                        outlog.str("");
                        outlog<<"pdDecomposeRightMatrix2 time: "<<(t1-t0)/CLOCKS_PER_SEC<<" s, nCPU: "<<nprocs
                            <<", method is: "<<method<<" elpa2kernel: "<<elpaKernel<<endl;
                        cout<<outlog.str();
                    }
                }
                else
                {
                    if(myid==MPIROOT)
                    {
                        outlog.str("");
                        outlog<<"pdDecomposeRightMatrix2 error, method: "<<method<<" info: "<<info<<endl;
                        cout<<outlog.str();
                    }
                    continue;
                }
                if(loglevel>1)
                {
                    if(myid==0)
                    {
                        outlog.str("");
                        outlog<<"matrix b was decomposed"<<endl;
                        cout<<outlog.str();
                    }
                    sprintf(filePrefix, "T_%d_K_%d_M_%d_U", ntest, elpaKernel, method);
                    saveMatrix(filePrefix, narows, nacols, b);
                }
                MPI_Barrier(MPI_COMM_WORLD);
                t0=clock();
                info=pzSolveEigen2(nev, nFull, narows, nacols, desc,
                                a, b, ev, q, work,
                                MPI_COMM_WORLD, comm_f, mpi_comm_rows, mpi_comm_cols, method,
                                elpaKernel,
                                wantEigenVector, wantDebug);

                t1=clock();
                if(myid==MPIROOT)
                {
                    outlog.str("");
                    outlog<<"pdSolveEigen2 time: "<<(t1-t0)/CLOCKS_PER_SEC<<" s, nCPU: "<<nprocs
                            <<", method is: "<<method<<" elpa2kernel: "<<elpaKernel<<endl;
                    cout<<outlog.str();
                }
                if(info!=0)
                {
                    if(myid==MPIROOT)
                    {
                        outlog.str("");
                        usleep(300);
                        outlog<<"calculation with method "<<method<<" failed, info:"<<info<<endl;
                        cout<<outlog.str();
                    }
                    delete[] H;
                    delete[] S;
                    delete[] a;
                    delete[] b;
                    delete[] q;
                    delete[] work;
                    delete[] ev;
                    Cblacs_gridexit(my_blacs_ctxt);
                    MPI_Finalize();
                    return info;
                }

                sprintf(filePrefix, "T_%d_K_%d_M_%d_Q", ntest, elpaKernel, method);
                if(loglevel>1) saveMatrix(filePrefix, narows, nacols, q);
            } // for method
        } // for elpaKernel
    } // for ntest

    usleep(1000);
    MPI_Barrier(MPI_COMM_WORLD);
    if(myprow==0 && mypcol==0)
    {
        outlog.str("");
        for(int j=0; j<nev; ++j)
            outlog<<"ev_"<<j<<": "<<setprecision(16)<<ev[j]<<endl;
        cout<<outlog.str();
    }

    //finalize end exit
    delete[] H;
    delete[] S;
    delete[] a;
    delete[] b;
    delete[] q;
    delete[] work;
    delete[] ev;
    Cblacs_gridexit(my_blacs_ctxt);
    MPI_Finalize();
    return 0;
}

