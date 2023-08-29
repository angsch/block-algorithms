function X = blockLeastSquares(A, B, transA)
%BLOCKLEASTSQUARES   Solves a least squares problem
%    X = BLOCKLEASTSQUARES(A, B) finds the X that minimizes
%    ||A * X - B|| or ||A**T * X - B|| using an orthogonal-
%    triangular factorization. Possible values of transA
%    are 't' (transpose) and 'n' (no transpose).

    % Extract dimensions.
    [m, n] = size(A);
    [m1, nrhs] = size(B);
    notrans = strncmp(transA, 'n', 1) || strncmp(transA, 'N', 1)
    trans = strncmp(transA, 't', 1) || strncmp(transA, 'T', 1)

    if not(notrans) && not(trans)
        error('transA has to be N or T');
    end

    if (notrans && m ~= m1) || (trans && n ~= m1)
        error('A and B do not have conforming dimensions');
    end

    blksz = 32;
    if m > n
        for j=1:blksz:n
            % Compute width of j-th panel.
            b = min(n-j+1, blksz);

            % QR decomposition of panel.
            [Qj, A(j:m, j:j+b-1)] = qr(A(j:m, j:j+b-1));

            % Trailing matrix update A := Q^T * A
            A(j:m, j+b:n) = Qj' * A(j:m, j+b:n);

            % B := Q^T * B
            B(j:m, 1:nrhs) = Qj' * B(j:m, 1:nrhs);
        end
        R = triu(A);
        X = R(1:n,1:n)\B(1:n, 1:nrhs);
    else % m < n
        for i=1:blksz:n
            % LQ decomposition of the current block row.
            [A(i:min(m,i+blksz-1),i:n), Qi] = blockLQ(A(i:min(m,i+blksz-1),i:n));

            % Trailing matrix update: A := A * Q^T
            A(i+blksz:m,i:n) = A(i+blksz:m,i:n) * Qi';

            % B := Q * B
            B(i:n,1:nrhs) = Qi * B(i:n, 1:nrhs);
        end
        L = tril(A(1:m,1:m))
        X = transpose(L)\B(1:m,1:nrhs);
    end
end
