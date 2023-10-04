# Block Linear Algebra Algorithms

This file collection contains prototype realizations of linear algebra algorithms. All algorithms are expressed as block or tile algorithms, meaning that the bulk of the computation corresponds to matrix-matrix multiplications (level-3 BLAS).

 * Matrix Factorizations
   - Cholesky decomposition
   - QR decomposition
   - RQ decomposition
   - LQ decomposition
   - QL decomposition
   - LU decomposition with partial pivoting
 * Eigenvalue Problem
   - Reduction to block Hessenberg form
   - Solver of shifted Hessenberg systems
   - Eigenvalue reordering in a Schur decomposition
 * Solvers
   - Least Squares solver (via QR)
   - Triangular Solve
