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
A = rand(m,n) + 1i * rand(m,n);

% Compute QR decomposition.
[Q, R] = blockQR(A);

err = norm(A - Q * R);
disp(['  ||A - Q * R|| = ', num2str(err)]);

err = norm(eye(m) - Q * Q');
disp(['  ||I - Q * Q^H|| = ', num2str(err)]);

clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Block RQ decomposition.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display('Test RQ decomposition');

% Create random m-by-n matrix.
m = 500;
n = 400;
A = rand(m,n) + 1i * rand(m,n);

% Compute RQ decomposition.
[R, Q] = blockRQ(A);

err = norm(A - R * Q);
disp(['  ||A - R * Q|| = ', num2str(err)]);

err = norm(eye(n) - Q * Q');
disp(['  ||I - Q * Q^H|| = ', num2str(err)]);

clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Block LQ decomposition.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display('Test LQ decomposition');

% Create random m-by-n matrix.
m = 1234;
n = 123;
A = rand(m,n) + 1i * rand(m,n);

% Compute LQ decomposition.
[L, Q] = blockLQ(A);

err = norm(A - L * Q);
disp(['  ||A - L * Q|| = ', num2str(err)]);

err = norm(eye(n) - Q * Q');
disp(['  ||I - Q * Q^H|| = ', num2str(err)]);

clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Block QL decomposition.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display('Test QL decomposition');

% Create random m-by-n matrix.
m = 1000;
n = 500;
A = rand(m,n) + 1i * rand(m,n);

% Compute QL decomposition.
[Q, L] = blockQL(A);

err = norm(A - Q * L);
disp(['  ||A - Q * L|| = ', num2str(err)]);

err = norm(eye(m) - Q * Q');
disp(['  ||I - Q * Q^H|| = ', num2str(err)]);

clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Block reduction to block Hessenberg form.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display('Test reduction to block Hessenberg form');

n = 1000;
A = rand(n,n);
[P, H] = blockReduceToBlockHess(A, 60);

err = norm(A - P * H * P');
disp(['  ||A - Q * T * Q^H|| = ', num2str(err)])

err = norm(eye(n) - P * P');
disp(['  ||I - P * P^H|| = ', num2str(err)])

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

clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simultaneous solution of shifted Hessenberg systems.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 1000;
num_rhs = 0.25 * n;

% Create a random Hessenberg matrix, random shifts and right-hand sides.
Lambda = rand(num_rhs,1);
B = ones(n,num_rhs);
A = rand(n,n);
H = hess(A);

% Solve (H - Lambda(k) * eye(n)) * X(:,k) = B(:,k).
X = blockShiftedHessenbergSolve(H, Lambda, B);

err = norm(H * X - X * diag(Lambda) - B);
disp(['  ||H * X - X * diag(Lambda) - B|| = ', num2str(err)]);
