function [Q, R] = cholQR(A)
%CHOLQR   QR decomposition via Cholesky
%    [Q, R] = cholQR(A) computes a QR decomposition
%    of A using the Cholesky decomposition. This
%    requires A to be sufficiently well conditioned.
%

    % Extract dimensions.
    [m, n] = size(A);
    
    % Form Gram matrix.
    G = A' * A;
    
    % Factor G = R' * R, where R is upper triangular.
    R = chol(G, 'upper');

    % Solve Q * R = A for Q.
    Q = A / R;
end
