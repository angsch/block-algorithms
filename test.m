%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Block Cholesky decomposition.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display('Test Cholesky decomposition');

% Create symmetric diagonally dominant matrix.
n = 1000;
A = rand(n) + n * eye(n);
A = A + A';

% Compute Cholesky factorization A = L * L'.
L = tril(blockCholesky(A));

err = norm(L * L' - A);
disp(['  || L * L^T - A || = ', num2str(err)])

% Compute Cholesky factorization A = R' * R.
R = triu(blockCholesky(A, 'upper'));

err = norm(R' * R - A);
disp(['  || R^T * R - A || = ', num2str(err)])

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
% Bulge chasing with offdiagonal block updates.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generate random Hessenberg matrix.
n = 1111;
H = triu(randn(n), -1);

% Set number of bulges such that half the matrix is filled.
nb = floor(n / 2 / 3);

% Place nb tightly packed bulges in top left corner.
for bulge = 1:nb
    k = 1+(bulge-1)*3;
    H(k+1:k+3, k:k+2) = randn(3);
end

% Call the chase routine.
[A, Q] = blockBulgeChase(H, nb);

err = norm(A - Q' * H * Q);
disp(['  ||A - Q^T * H * Q|| = ', num2str(err)])

err = norm(eye(n) - Q * Q');
disp(['  ||I - Q * Q^T|| = ', num2str(err)])

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

display('Test shifted Hessenberg system solve');

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

clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Blocked least squares via a QR factorization.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display('Test least squares via QR factorization');

t = 0:0.25:3';
b = [2.9986, 2.6569, 2.4291, 2.2664, 2.1444, 2.0495, 1.9736, 1.9115, 1.8597, 1.8159, 1.7783, 1.7458, 1.7173]';
% Number of data points
m = size(b)(1);
% Number of parameters
n = 2;
A = zeros(m, n);
A(:,1) = arrayfun(@(t) 1 / (1+t), t)';
A(:,2) = ones(m, 1);
X = blockLeastSquares(A, b, 'notranspose');
% Fitted model
f = @(x) X(1) / (1 + x) + X(2);
% Plot fitted model
residuals = b - arrayfun(f, t)';
norm(residuals)
% plot(t, residuals, '*')

clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Blocked Triangular Solve.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display('Test triangular solve');

n = 234; nrhs = 10;

% Create a diagonally dominant matrix.
A = rand(n) + eye(n) * 2 * n;
B = rand(n, nrhs);

% Solve lower triangular system L * X = B.
L = tril(A);
X = blockTriangularSolve(L, B);
err = norm(L * X - B);
disp(['  || L * X - B || = ', num2str(err)])

% Solver upper triangular system U * X = B.
U = triu(A);
X = blockTriangularSolve(U, B, 'upper');
err = norm(U * X - B);
disp(['  || U * X - B || = ', num2str(err)])

clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bidiagonal SVD (unblocked)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display('Test SVD of bidiagonal matrix by implicitly shifted QR algorithm');

n = 6;
d = rand(n,1); %diagonal entries
e = rand(n-1,1); %superdiagonal entries

B = diag(d) + diag(e,1);
[U, S, V] = bidiagSVD(B);

err = norm(U*S*V'-B);
disp(['  || U*S*V^T-B || = ', num2str(err)])
err = norm(U*U' - eye(n));
disp(['  || U * U^T - I || = ', num2str(err)])
err = norm(V*V' - eye(n));
disp(['  || V * V^T - I || = ', num2str(err)])

clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reduction to triangular band form.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display('Test reduction to triangular band form');

bandwidth = 8;
m = 100;
n = 60;
A = rand(m, n);
[B, Q, P] = blockReduceToTriangularBand(A, bandwidth);

err = norm(Q'*A*P-B);
disp(['  || Q^T*A*P-B || = ', num2str(err)])
err = norm(Q*Q' - eye(m));
disp(['  || Q * Q^T - I || = ', num2str(err)])
err = norm(P*P' - eye(n));
disp(['  || P * P^T - I || = ', num2str(err)])

clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test SVD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display('Test SVD via Polar decomposition');

n = 100;
A = rand(n,n) + 1i * rand(n,n);
[U, S, V] = blockSVD(A, 'polar');

err = norm(U * S * V' - A);
disp(['  || U * S * V^H - A || = ', num2str(err)]);
err = norm(U*U' - eye(n));
disp(['  || U * U^H - I || = ', num2str(err)])
err = norm(V*V' - eye(n));
disp(['  || V * V^H - I || = ', num2str(err)])

% Plot singular value distribution
% semilogy(S, '-o')
