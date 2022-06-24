#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "elpa_solver.h"
#include "utils.h"
#include "my_math.hpp"

using namespace std;

void diag(bool isNewRound, MPI_Comm comm, int nev, int narows, int nacols,		  
		  double *a, double *b, double *atmp, double *btmp, 
		  double* ev, double* q, int* desc)
{
	const int subMatrixSize=narows*nacols;
	const bool isReal=true;
	static ELPA_Solver es(isReal, comm, nev, narows, nacols, desc);
	int loglevel=1;
    es.setLoglevel(loglevel);
    
	Cdcopy(subMatrixSize, a, atmp);
    static int DecomposedState;
    if(isNewRound)
    {
		DecomposedState=0;
		Cdcopy(subMatrixSize, b, btmp);
    }
    es.generalized_eigenvector(atmp, btmp, DecomposedState, ev, q);

    double maxError, meanError;
	es.verify(a, b, ev, q, maxError, meanError);
	
	int myid;
	MPI_Comm_rank(comm, &myid);
    stringstream outlog;
	if(myid==0)
	{
		outlog.str("");
		outlog<<"Diag function max error="<<maxError<<"; mean error="<<meanError<<endl;
		cout<<outlog.str();
	}

}

int main(int argc, char** argv)
{
    int nFull, nev, nblk, ntest, loglevel;
    int myid, nprocs;
    int my_blacs_ctxt;
    int info;
    int narows, nacols;
    double *H, *S, *a, *b, *q, *ev;
    int desc[9];

    stringstream outlog;

    double t0, t1;

    // set mpi enviorment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    //load parameters
    if(myid==0)
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
            outlog.str("");
            outlog<<"parameters loaded: "<<nFull<<" "<<nev<<" "<<nblk<<" "<<ntest<<" "<<loglevel<<endl;
            cout<<outlog.str();
        }
    }

    MPI_Bcast(&info, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(info != 0)
    {
        MPI_Finalize();
        return info;
    }

    MPI_Bcast(&nFull, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nev, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nblk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ntest, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&loglevel, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(loglevel>1)
    {
        outlog.str("");
        outlog<<"myid "<<myid<<": parameters synchonized";
        outlog<<" nFull: "<<nFull<<" nev: "<<nev<<" nblk: "<<nblk<<" ntest: "<<ntest<<" loglevel: "<<loglevel<<endl;
        cout<<outlog.str();
    }

    // set blacs parameters
    initBlacsGrid(loglevel, MPI_COMM_WORLD, nFull, nblk,
                  my_blacs_ctxt, narows, nacols, desc);

    //init main matrices
    int subMatrixSize=narows*nacols;
    H=new double[subMatrixSize];
    S=new double[subMatrixSize];
    a=new double[subMatrixSize];
    b=new double[subMatrixSize];
    q=new double[subMatrixSize];
    ev=new double[nFull];
    for(int i=0; i<subMatrixSize; ++i) q[i]=0;
    for(int i=0; i<nFull; ++i) ev[i]=0;

    //load input matrices and distribute to all processes
    MPI_Barrier(MPI_COMM_WORLD);
    t0=MPI_Wtime();

    loadMatrix("H.dat", nFull, H, desc, my_blacs_ctxt);
    if( (loglevel>0 && myid==0) || loglevel>1)
    {
        outlog.str("");
        outlog<<"matrix H loaded"<<endl;
        cout<<outlog.str();
    }
    if(loglevel>2) saveLocalMatrix("Loaded_H", narows, nacols, H);

    loadMatrix("S.dat", nFull, S, desc, my_blacs_ctxt);
    if( (loglevel>0 && myid==0) || loglevel>1)
    {
        outlog.str("");
        outlog<<"matrix S loaded"<<endl;
        cout<<outlog.str();
    }
    if(loglevel>2) saveLocalMatrix("Loaded_S", narows, nacols, S);
    t1=MPI_Wtime();

    if( (loglevel>0 && myid==0) || loglevel>1)
    {
        outlog.str("");
        outlog<<"myid "<<myid<<": load file time:"<<t1-t0<<"s"<<endl;
        cout<<outlog.str();
    }

    //call elpa to do the testing
    const bool isReal=true;
    ELPA_Solver es(isReal, MPI_COMM_WORLD, nev, narows, nacols, desc);
    es.setLoglevel(loglevel);
    int DecomposedState=0;

    if( (loglevel>0 && myid==0) || loglevel>1)
    {
        outlog.str("");
        outlog<<"myid "<<myid<<": ELPA_Solver is created."<<endl;
        cout<<outlog.str();
    }
    // check elpa parameters
    if( (loglevel>0 && myid==0) || loglevel>1) es.outputParameters();

    // start testing
    double maxError, meanError;

    if(myid==0)
    {
        outlog.str("");
        outlog<<"Test eigenvector solver:"<<endl;
        cout<<outlog.str();
    }
    Cdcopy(subMatrixSize, H, a);
    if( (loglevel>0 && myid==0) || loglevel>1)
    {
        outlog.str("");
        outlog<<"myid "<<myid<<": input matrices are prepared, solver is running..."<<endl;
        cout<<outlog.str();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t0=MPI_Wtime();
    es.eigenvector(a, ev, q);
    t1=MPI_Wtime();
    if( (loglevel>0 && myid==0) || loglevel>1)
    {
        outlog.str("");
        outlog<<"myid "<<myid<<": a is solved"<<endl;
        cout<<outlog.str();
    }
    // check result
    es.verify(H, ev, q, maxError, meanError);
    if(myid==0)
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
        Cdcopy(subMatrixSize, H, a);
        if(i==0)
        {
            DecomposedState=0;
            Cdcopy(subMatrixSize, S, b);
        }
        if( (loglevel>0 && myid==0) || loglevel>1)
        {
            outlog.str("");
            outlog<<"myid "<<myid<<": input matrices are prepared, solver is running..."<<endl;
            cout<<outlog.str();
        }
        MPI_Barrier(MPI_COMM_WORLD);
        t0=MPI_Wtime();
        es.generalized_eigenvector(a, b, DecomposedState, ev, q);
        t1=MPI_Wtime();
        if( (loglevel>0 && myid==0) || loglevel>1)
        {
            outlog.str("");
            outlog<<"myid "<<myid<<": solver is done..."<<endl;
            cout<<outlog.str();
        }
        if(myid==0)
        {
            outlog.str("");
            outlog<<"ntest: "<<i<<" solving time:"<<t1-t0<<"s"<<endl;
            cout<<outlog.str();

            outlog.str("");
            outlog<<"ntest_"<<i<<"_ev:"; //<<setprecision(16);
            for(int j=0; j<nev; ++j)
                outlog<<' '<<ev[j];
            outlog<<endl;
            cout<<outlog.str();
        }
        // check result
        if(i==0) saveMatrix("eigenvector.dat", nFull, q, desc, my_blacs_ctxt);
        es.verify(H, S, ev, q, maxError, meanError);
        if(myid==0)
        {
            outlog.str("");
            outlog<<"ntest: "<<i<<" max error="<<maxError<<"; mean error="<<meanError<<endl;
            cout<<outlog.str();
        }
    } // for ntest
    
    // test diag
	if(myid==0)
	{
		outlog.str("");
		outlog<<"start test diag function"<<endl;
		cout<<outlog.str();
	}
    bool isNewRound=true;  
    diag(isNewRound, MPI_COMM_WORLD, nev, narows, nacols,		  
		  H, S, a, b, ev, q, desc);

	if(myid==0)
	{
		outlog.str("");
		outlog<<"start test diag function again"<<endl;
		cout<<outlog.str();
	}  
	isNewRound=false;
    diag(isNewRound, MPI_COMM_WORLD, nev, narows, nacols,		  
		  H, S, a, b, ev, q, desc);
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
