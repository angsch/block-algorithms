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
% Reduction to upper bidiagonal form.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

display('Test 2-stage reduction to bidiagonal form');

blksz = 8;
m = 100;
n = 60;
A = rand(m, n);
[B, Q, P] = blockReduceToBidiagonal(A, blksz);

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

clear all;


display('Test SVD via bidiagonal QR iterations');
n = 8;
A = rand(n,n);
[U, S, V] = blockSVD(A, 'bidiag');

err = norm(U * S * V' - A);
disp(['  || U * S * V^H - A || = ', num2str(err)]);
err = norm(U*U' - eye(n));
disp(['  || U * U^H - I || = ', num2str(err)])
err = norm(V*V' - eye(n));
disp(['  || V * V^H - I || = ', num2str(err)])

%err = norm(svd(A) - diag(S))/norm(svd(A))
%disp(['  || S - S_ref ||/||S|| = ', num2str(err)]);
