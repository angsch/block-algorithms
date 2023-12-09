[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

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


This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
