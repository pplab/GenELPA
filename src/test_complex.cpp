#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include "elpa_solver.h"
#include "utils.h"
#include "my_math.hpp"

using namespace std;

int main(int argc, char** argv)
{
    int nFull, nev, nblk, ntest, loglevel;
    int myid, nprocs;
    int my_blacs_ctxt;
    int info;
    int MPIROOT=0;
    int narows, nacols;

    double *ev;
    complex<double> *H, *S, *a, *b, *q;

    int desc[9];
    stringstream outlog;
    char filename[40];

    clock_t t0, t1;

    // set mpi enviorment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    const int LOG_INTERVAL=min(int(1e6/nprocs), 500); // unit: micro seconds

    //load parameters
    if(myid==MPIROOT)
    {
        fstream inputFile("INPUT");
        if(!inputFile)
        {
            info=1;
            cout<<"Cannot open INPUT file"<<endl;
        }
        else
        {
            info=0;
            inputFile>>nFull>>nev>>nblk>>ntest>>loglevel;
            inputFile.close();
            usleep(20);
            outlog.str("");
            outlog<<"parameters loaded: "<<nFull<<" "<<nev<<" "<<nblk<<" "<<ntest<<" "<<loglevel<<endl;
            cout<<outlog.str();
        }
    }

    MPI_Bcast(&info, 1, MPI_INT, MPIROOT, MPI_COMM_WORLD);
    if(info != 0)
    {
        MPI_Finalize();
        return info;
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
        usleep(myid*LOG_INTERVAL);
        cout<<outlog.str();
    }

    // set blacs parameters
    MPI_Barrier(MPI_COMM_WORLD);
    initBlacsGrid(loglevel, MPI_COMM_WORLD, nFull, nblk,
                  my_blacs_ctxt, narows, nacols, desc);
    MPI_Barrier(MPI_COMM_WORLD);

    //init main matrices
    int subMatrixSize=narows*nacols;
    H=new complex<double>[subMatrixSize];
    S=new complex<double>[subMatrixSize];
    a=new complex<double>[subMatrixSize];
    b=new complex<double>[subMatrixSize];
    q=new complex<double>[subMatrixSize];
    ev=new double[nFull];
    for(int i=0; i<subMatrixSize; ++i) q[i]=0;
    for(int i=0; i<nFull; ++i) ev[i]=0;

    //load input matrices and distribute to all processes
    MPI_Barrier(MPI_COMM_WORLD);
    t0=MPI_Wtime();

	strcpy(filename,"H.dat");
    loadMatrix("H.dat", nFull, H, desc, my_blacs_ctxt);
    if(loglevel>2)
    {
    	if(myid==0)
    	{
	        outlog.str("");
	        outlog<<"matrix H loaded"<<endl;
        	cout<<outlog.str();
    	}
        if(loglevel>2) saveLocalMatrix("Loaded_H", narows, nacols, H);
    }

    loadMatrix("S.dat", nFull, S, desc, my_blacs_ctxt);
    if(loglevel>0)
    {
    	if(myid==0)
    	{
	        outlog.str("");
	        outlog<<"matrix S loaded"<<endl;
        	cout<<outlog.str();
    	}
        if(loglevel>2) saveLocalMatrix("Loaded_S", narows, nacols, S);
    }
    t1=MPI_Wtime();
    if(loglevel>0)
    {
        outlog.str("");
        outlog<<"myid "<<myid<<": load file time:"<<t1-t0<<"s"<<endl;
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(myid*LOG_INTERVAL);
        cout<<outlog.str();
    }

    //call elpa to do the testing
    const bool isReal=false;
    ELPA_Solver es(isReal, MPI_COMM_WORLD, nev, narows, nacols, desc);
    es.setLoglevel(loglevel);
    int DecomposedState=0;

    if(loglevel>0)
    {
        outlog.str("");
        outlog<<"myid "<<myid<<": ELPA_Solver is created."<<endl;
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(myid*LOG_INTERVAL);
        cout<<outlog.str();
    }
    // check elpa parameters
    if(loglevel>1) es.outputParameters();
    // start testing
    if(myid==MPIROOT)
    {
        usleep(5000);
        outlog.str("");
        outlog<<"Test eigenvector solver:"<<endl;
        cout<<outlog.str();
    }
    t0=MPI_Wtime();
    Czcopy(subMatrixSize, H, a);
    if(loglevel>0)
    {
        outlog.str("");
        outlog<<"myid "<<myid<<": input matrices are prepared, solver is running..."<<endl;
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(myid*LOG_INTERVAL);
        cout<<outlog.str();
    }
    es.eigenvector(a, ev, q);
    saveMatrix("eigenvector.dat", nFull, q, desc, my_blacs_ctxt);
    if(loglevel>0)
    {
        outlog.str("");
        outlog<<"myid "<<myid<<": a is solved"<<endl;
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(myid*LOG_INTERVAL);
        cout<<outlog.str();
    }
    t1=MPI_Wtime();

    // check result
    double maxError, meanError;
    es.verify(H, ev, q, maxError, meanError);
    if(myid==MPIROOT)
    {
        outlog.str("");
        outlog<<"eigenvector solving time:"<<t1-t0<<"s"<<endl;
        outlog<<"eigenvector result: maxError="<<maxError<<"; meanError="<<meanError<<endl;
        cout<<outlog.str();

        outlog.str("");
        outlog<<"ev:"; //<<setprecision(16);
        for(int j=0; j<nev; ++j)
            outlog<<' '<<ev[j];
        outlog<<endl;
        cout<<outlog.str();

        outlog.str("");
        outlog<<"Test generalized_eigenvector solver:"<<endl;
        cout<<outlog.str();
    }


    for (int i=0; i<ntest; ++i)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        t0=MPI_Wtime();
        Czcopy(subMatrixSize, H, a);
        if(i==0)
        {
            DecomposedState=0;
            Czcopy(subMatrixSize, S, b);
        }
        if(loglevel>0)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            outlog.str("");
            outlog<<"myid "<<myid<<": input matrices are prepared, solver is running..."<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            usleep(myid*LOG_INTERVAL);
            cout<<outlog.str();
        }
        es.generalized_eigenvector(a, b, DecomposedState, ev, q);
        if(loglevel>0)
        {
            outlog.str("");
            outlog<<"myid "<<myid<<": solver is done..."<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            usleep(myid*LOG_INTERVAL);
            cout<<outlog.str();
        }
        t1=MPI_Wtime();
        if(myid==MPIROOT)
        {
            outlog.str("");
            outlog<<"ntest: "<<i<<" solver time:"<<t1-t0<<"s"<<endl;
            cout<<outlog.str();

            outlog.str("");
            outlog<<"ntest_"<<i<<"_ev:"; //<<setprecision(16);
            for(int j=0; j<nev; ++j)
                outlog<<' '<<ev[j];
            outlog<<endl;
            cout<<outlog.str();
        }
        // check result
        if(i==0) saveMatrix("gen_eigenvector.dat", nFull, q, desc, my_blacs_ctxt);
        es.verify(H, S, ev, q, maxError, meanError);
        if(myid==MPIROOT)
        {
            outlog.str("");
            outlog<<"ntest: "<<i<<" maxError="<<maxError<<"; meanError="<<meanError<<endl;
            //outlog<<"ntest: "<<i<<" max error="<<maxError<<"; mean error="<<meanError<<endl;
            cout<<outlog.str();
        }
    } // for ntest

    //finalize end exit
    delete[] H;
    delete[] S;
    delete[] a;
    delete[] b;
    delete[] q;
    delete[] ev;
    es.exit();
    Cblacs_gridexit(my_blacs_ctxt);
    MPI_Finalize();
    return 0;
}

