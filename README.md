# GenELPA: A easy-to-use ELPA interface to solve generalized eigenvalue problems

GenELPA is a tool for solving generalized eigenvalues using ELPA as a computational core.

[ELPA](https://elpa.mpcdf.mpg.de/index.html) is a high-performance and highly scalable direct eigensolvers solver, and is used in a variety of first-principles computing softwares. The program is developed at the Max Planck Institute, and is written in FORTRAN with a C interface.

Since the first version was released in 2011, ELPA has used two APIs, a traditional interface used in 2016.05.004 and earlier versions, and a new interface that uses a handle to pass computational configuration parameters used in 2016.11.001 and later versions. With the new interface, ELPA adds functions for solving generalized eigenvalues directly and support for GPUs.

We have previously made an extension to GenELPA for solving generalized eigenvalues, which provides functions for solving generalized eigenvalues directly as well as step-by-step solving and enhanced robustness in solving various matrices. The program is able to work stably in [ABACUS](https://abacus.ustc.edu.cn/).

The new interface of ELPA already supports direct solving of generalized eigenvalues, but we found the following problems in the process of using it.

1. less stable calculations when solving certain matrices.
2. unnecessary screen output in some cases (version 2021.05.002).

Therefore, we have rewritten the program that can support generalized eigenvalue solving based on the original GenELPA, and solved these two problems. At the same time, considering that ELPA of the old API is still widely used, we also supported ELPA of the old API in the new program. for compatibility and ease of use, we kept the invocation methods of the old and new versions the same, so that programs using ELPA do not need to change the source code, but only need to link different library files at compile time to support ELPA of the old and new different APIs.

We also added the function of automatically selecting the solution kernel according to the CPU parameters of the running server, so that users can get good computational efficiency by default.
