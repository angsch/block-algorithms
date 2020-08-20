function [Q, R] = blockQR(A)
%BLOCKQR   Orthogonal-triangular decomposition.
%    [Q, R] = BLOCKQR(A), where A is an m-by-n matrix, produces an m-by-n upper
%    triangular matrix R and an m-by-m orthogonal matrix Q so that A = Q*R.

    % Extract dimensions.
    [m, n] = size(A);
    k = min(m, n);

    blksz = 100;

    Q = eye(m);

    if blksz > k
        % Use Matlab's qr factorization for small matrices.
        [Q, R] = qr(A);
    else
        % Use blocked qr.
        for j=1:blksz:k
            % Compute width of k-th panel.
            b = min(n-j+1, blksz);
            
            % QR decomposition of panel.
            [Qj, A(j:m, j:j+b-1)] = qr(A(j:m, j:j+b-1));
            
            % A := Q^T * A
            A(j:m, j+b:n) = Qj' * A(j:m, j+b:n);
            
            % Update Q.
            Q = Q * [eye(j-1),         zeros(j-1,m-j+1);
                     zeros(m-j+1,j-1), Qj               ];
    end
    R = triu(A);
end
