% Assume square matrix
% This is a viable assumption after QR processing
m = 10;
n = m;
A = rand(m,n);

options = struct("oneSided", true, ...
                 "precond", true);
%[U, S, V] = svdJacobi(A, options);
[U, S, V] = svdBlockJacobi(A);


err = norm(A - U * diag(S) * V')/(norm(A) * max(m,n));
disp(['| A - U * S * V^T| / ( |A| max(m,n) ) = ', num2str(err)]);

err = norm(eye(n) - U' * U) / m;
disp(['| I - U^T * U | / m  = ', num2str(err)]);

err = norm(eye(n) - V * V') / n;
disp(['| I - V * V^T | / n = ', num2str(err)]);
