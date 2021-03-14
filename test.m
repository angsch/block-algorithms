%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Block Cholesky decomposition.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display('Test Cholesky decomposition');

% Create symmetric diagonally dominant matrix.
n = 1000;
A = rand(n) + n * eye(n);
A = A + A';

% Compute Cholesky factorization.
L = tril(blockCholesky(A));

err = norm(L * L' - A);
disp(['  || L * L^T - A || = ', num2str(err)])

clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Block QR decomposition.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display('Test QR decomposition');

% Create random m-by-n matrix.
m = 2000;
n = 1000;
A = rand(m,n);

% Compute QR decomposition.
[Q, R] = blockQR(A);

err = norm(A - Q * R);
disp(['  ||A - Q * R|| = ', num2str(err)]);

err = norm(eye(m) - Q * Q');
disp(['  ||I - Q * Q^T|| = ', num2str(err)]);

clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Block RQ decomposition.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display('Test RQ decomposition');

% Create random m-by-n matrix.
m = 500;
n = 400;
A = rand(m,n);

% Compute RQ decomposition.
[R, Q] = blockRQ(A);

err = norm(A - R * Q);
disp(['  ||A - R * Q|| = ', num2str(err)]);

err = norm(eye(n) - Q * Q');
disp(['  ||I - Q * Q^T|| = ', num2str(err)]);

clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Block eigenvalue reordering.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display('Test eigenvalue reordering');

% Create a non-symmetric A and compute its Schur decomposition A := Q T Q^H.
n = 200;
A = rand(n,n) + i * rand(n,n);
[Q, T] = schur(A, 'complex');

% Randomly select positions affiliated with the eigenvalues of A.
select = randi([0 1], 1, n);

% Reorder.
[QS, TS] = blockReorderSchur(Q, T, select);

err = norm(A - QS * TS * QS');
disp(['  ||A - Q * T * Q^H|| = ', num2str(err)])
err = norm(eye(n) - QS * QS');
disp(['  ||I - Q * Q^H|| = ', num2str(err)])

clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Block LU decomposition with partial pivoting.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display('Test LU decomposition');

% Create a random matrix.
n = 1000;
A = rand(n,n);
[L, U, P] = blockLU(A);

err = norm(P * A - L * U);
disp(['  ||P * A - L * U|| =', num2str(err)]);
