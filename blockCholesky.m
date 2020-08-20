function L = blockCholesky(A)
%BLOCKCHOLESKY   Cholesky factorization.
%    BLOCKCHOLESKY(A) uses the lower triangular part of A.
%    The upper triangular part is assumed to be the transpose
%    of the lower triangular part.  If A is positive definite, 
%    then L = CHOL(A) produces a lower triangular L so that
%    L'*L = A. If A is not positive definite, an error message
%    is printed.

    % Extract dimension.
    [n,~] = size(A);

    blksz = 120;

    for j = 1:blksz:n
        % Compute block width.
        b = min(n-j+1, blksz);

        % Given the partitioning,
        % [A11    ]
        % [A21 A22]
        % complete Cholesky decomposition in panel [A11; A21].
        % A11 := chol(A11).
        A(j:j+b-1, j:j+b-1) = chol(A(j:j+b-1,j:j+b-1),'lower');
        % A21 := A21 * tril(A11)^-T.
        A(j+b:n, j:j+b-1) = A(j+b:n, j:j+b-1) / tril(A(j:j+b-1, j:j+b-1))';

        % Trailing matrix update.
        % A22 := A22 - tril(A21 * A21^T).
        A(j+b:n, j+b:n) = A(j+b:n, j+b:n) - tril(A(j+b:n, j:j+b-1) * A(j+b:n, j:j+b-1)');
    end
    L = A;         
end
