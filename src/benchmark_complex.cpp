#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "elpa_solver.h"
#include "utils.h"
#include "my_math.hpp"

using namespace std;

static inline void out_ev(int nev,complex<double> ev[])
{
	stringstream outlog;
	outlog.str("");
	outlog<<"ev:";
	for(int j=0; j<nev; ++j)
		outlog<<' '<<ev[j];
	outlog<<endl;
	cout<<outlog.str();
}

int main(int argc, char** argv)
{
    int nFull, nev, nblk, nkernels, loglevel;
    int myid, nprocs;
    int my_blacs_ctxt;
    int info;
    int narows, nacols;
    complex<double> *H, *S, *a, *b, *q, *ev;
    int desc[9];
    int kernel_id;
    
    vector<int> kernel_list;
    stringstream outlog;
    double t0, t1;

    // set mpi enviorment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    //load input parameters
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
            inputFile>>nFull>>nev>>nblk>>nkernels>>loglevel;
            for(int i=0; i<nkernels; ++i)
            {
				inputFile>>kernel_id;
				kernel_list.push_back(kernel_id);
			}
            inputFile.close();
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
    MPI_Bcast(&nkernels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&loglevel, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(loglevel>1)
    {
        outlog.str("");
        outlog<<"myid "<<myid<<": parameters synchonized";
        outlog<<" nFull: "<<nFull<<" nev: "<<nev<<" nblk: "<<nblk<<" nkernels: "<<nkernels<<" loglevel: "<<loglevel<<endl;
        cout<<outlog.str();
    }

    // set blacs parameters
    initBlacsGrid(loglevel, MPI_COMM_WORLD, nFull, nblk,
                  my_blacs_ctxt, narows, nacols, desc);

    //init main matrices
    int subMatrixSize=narows*nacols;
    H=new complex<double>[subMatrixSize];
    S=new complex<double>[subMatrixSize];
    a=new complex<double>[subMatrixSize];
    b=new complex<double>[subMatrixSize];
    q=new complex<double>[subMatrixSize];
    ev=new complex<double>[nFull];
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
    const bool isReal=false;
    ELPA_Solver es(isReal, MPI_COMM_WORLD, nev, narows, nacols, desc);
    es.setLoglevel(loglevel);

    if( (loglevel>0 && myid==0) || loglevel>1)
    {
        outlog.str("");
        outlog<<"myid "<<myid<<": ELPA_Solver is created."<<endl;
        cout<<outlog.str();
    }
    
    // start testing
    double maxError, meanError;
    
    // test each kernel and QR
    for(int i=0; i<nkernels; ++i)
    {
		for(int useqr=0; useqr<2; ++useqr)
		{
			// setup kernel and QR
			if(myid==0) kernel_id=kernel_list[i];
			MPI_Bcast(&kernel_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
			es.setKernel(kernel_id, useqr);
			// check elpa parameters
			if( (loglevel>0 && myid==0) || loglevel>1) es.outputParameters();
			// test eigen solver
			Czcopy(subMatrixSize, H, a);
			MPI_Barrier(MPI_COMM_WORLD);
			t0=MPI_Wtime();
			es.eigenvector(a, ev, q);
			t1=MPI_Wtime();
			es.verify(H, ev, q, maxError, meanError);
			if(myid==0)
			{
				if(loglevel>1) out_ev(nev, ev);	
				outlog.str("");		
				outlog<<"kernel: "<<i<<" elpa solving time:"<<t1-t0<<"s"<<endl;
				outlog<<"kernel: "<<i<<" elpa max error="<<maxError<<"; mean error="<<meanError<<endl;
				cout<<outlog.str();
			}			
			// test generalized eigen solver
			Czcopy(subMatrixSize, H, a);
            Czcopy(subMatrixSize, S, b);            
			int DecomposedState=0;
			MPI_Barrier(MPI_COMM_WORLD);
			t0=MPI_Wtime();
			es.generalized_eigenvector(a, b, DecomposedState, ev, q);
			t1=MPI_Wtime();
			es.verify(H, S, ev, q, maxError, meanError);
			if(myid==0)
			{
				if(loglevel>1) out_ev(nev, ev);	
				outlog.str("");
				outlog<<"kernel: "<<i<<" genelpa solving time:"<<t1-t0<<"s"<<endl;
				outlog<<"kernel: "<<i<<" genelpa max error="<<maxError<<"; mean error="<<meanError<<endl;
				cout<<outlog.str();
			}	
			// test generalized eigen solver with decomposed S;
			Czcopy(subMatrixSize, H, a);
			MPI_Barrier(MPI_COMM_WORLD);
			t0=MPI_Wtime();
			es.generalized_eigenvector(a, b, DecomposedState, ev, q);
			t1=MPI_Wtime();
			es.verify(H, S, ev, q, maxError, meanError);
			if(myid==0)
			{
				if(loglevel>1) out_ev(nev, ev);	
				outlog.str("");
				outlog<<"kernel: "<<i<<" genelpa solving time(decomposed S):"<<t1-t0<<"s"<<endl;
				outlog<<"kernel: "<<i<<" genelpa max error="<<maxError<<"; mean error="<<meanError<<endl;
				cout<<outlog.str();
			}
		} //useqr
    } // kernel
    
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
