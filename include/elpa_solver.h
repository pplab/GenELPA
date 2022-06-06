#pragma once
#include <complex>
using namespace std;
class ELPA_Solver
{
    public:
    ELPA_Solver(bool isReal, MPI_Comm comm, int nev, int narows, int nacols, int* desc);
    ELPA_Solver(bool isReal, MPI_Comm comm, int nev, int narows, int nacols, int* desc, int* otherParameter);

    int eigenvector(double* A, double* EigenValue, double* EigenVector);
    int generalized_eigenvector(double* A, double* B, int& DecomposedState,
                                double* EigenValue, double* EigenVector);
    int eigenvector(complex<double>* A, double* EigenValue, complex<double>* EigenVector);
    int generalized_eigenvector(complex<double>* A, complex<double>* B, int& DecomposedState,
                                double* EigenValue, complex<double>* EigenVector);
    void setLoglevel(int loglevel);
    void setKernel(bool isReal, int Kernel);
    void setQR(int useQR);
    void outputParameters();
    void verify(double* A, double* EigenValue, double* EigenVector,
                double &maxRemain, double &meanRemain);
    void verify(double* A, double* B, double* EigenValue, double* EigenVector,
                double &maxRemain, double &meanRemain);
	void verify(complex<double>* A, double* EigenValue, complex<double>* EigenVector,
                         double &maxError, double &meanError);
	void verify(complex<double>* A, complex<double>* B,
                double* EigenValue, complex<double>* EigenVector,
                double &maxError, double &meanError);
    void exit();

    private:
    const int  CHOLESKY_CRITICAL_SIZE=1000;
    bool isReal;
    MPI_Comm comm;
    int nFull;
    int nev;
    int narows;
    int nacols;
    int* desc;
    int method;
    int kernel_id;
    int cblacs_ctxt;
    int nblk;
    int lda;
    double* dwork=NULL;
    complex<double>* zwork=NULL;
    int myid;
    int nprows;
    int npcols;
    int myprow;
    int mypcol;
    int useQR;
    int wantDebug;
    int loglevel;
    // for legacy interface
    int comm_f;
    int mpi_comm_rows;
    int mpi_comm_cols;
    // for new elpa handle
    int handle_id;
    
    // toolbox
    int read_cpuflag();
    int read_real_kernel();
    int read_complex_kernel();
    int allocate_work();
    int decomposeRightMatrix(double* B, double* EigenValue, double* EigenVector, int& DecomposedState);
    int decomposeRightMatrix(complex<double>* B, double* EigenValue, complex<double>* EigenVector, int& DecomposedState);
    int composeEigenVector(int DecomposedState, double* B, double* EigenVector);
    int composeEigenVector(int DecomposedState, complex<double>* B, complex<double>* EigenVector);
    // debug tool
    void timer(int myid, const char function[], const char step[], double &t0);
};
