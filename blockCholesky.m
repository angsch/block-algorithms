function L = blockCholesky(A, shape)
%BLOCKCHOLESKY   Cholesky factorization.
%    BLOCKCHOLESKY(A) uses the lower triangular part of A. The upper triangular
%    part is assumed to be the transpose of the lower triangular part.  If A
%    is positive definite, then L = BLOCKCHOLESKY(A) produces a lower
%    triangular L so that L'*L = A. If A is not positive definite, an error
%    message is printed.
%
%    R = BLOCKCHOLESKY(A, 'upper') uses the upper triangular part of A and
%    produces an upper triangular R so that R*R' = A. If A is not positive
%    definite, an error message is printed.

    if nargin < 2
        shape = 'lower';
    end

    if shape ~= 'lower' && shape ~= 'upper'
        error('Shape must upper or lower.');
    end

    % Extract dimension.
    [n,~] = size(A);

    blksz = 120;

    if shape == 'lower'
        for j = 1:blksz:n
            % Compute block width.
            b = min(n-j+1, blksz);

            % Partition [A11    ]
            %           [A21 A22] and compute A11 := chol(A11).
            A(j:j+b-1, j:j+b-1) = chol(A(j:j+b-1,j:j+b-1),'lower');

            % A21 := A21 * tril(A11)^-T.
            A(j+b:n, j:j+b-1) = A(j+b:n, j:j+b-1) / tril(A(j:j+b-1, j:j+b-1))';

            % Trailing matrix update.
            % A22 := A22 - tril(A21 * A21^T).
            A(j+b:n, j+b:n) = A(j+b:n, j+b:n) - tril(A(j+b:n, j:j+b-1) * A(j+b:n, j:j+b-1)');
        end
    else % shape == 'upper'
        for j = 1:blksz:n
            % Compute block width.
            b = min(n-j+1, blksz);

            % Partition [A11 A12 ]
            %           [    A22 ] and compute A11 := chol(A11).
            A(j:j+b-1, j:j+b-1) = chol(A(j:j+b-1,j:j+b-1));

            % A12 := triu(A11)^-T * A21.
            A(j:j+b-1,j+b:n) = triu(A(j:j+b-1,j:j+b-1))' \ A(j:j+b-1,j+b:n);

            % Trailing matrix update.
            % A22 := A22 - triu(A12^T * A12).
            A(j+b:n, j+b:n) = A(j+b:n, j+b:n) - triu(A(j:j+b-1, j+b:n)' * A(j:j+b-1, j+b:n));
        end
    end
    L = A;
end
