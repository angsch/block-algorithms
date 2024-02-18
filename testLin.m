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
