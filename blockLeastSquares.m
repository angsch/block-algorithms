function X = blockLeastSquares(A, B)
%BLOCKLEASTSQUARES   Finds the X that minimizes ||A * X - B||
%    X = BLOCKLEASTSQUARES(A, B) computes the least squares solution
%    of an overdetermined system using a QR factorization. A is an
%    m-by-n matrix of full rank, B is n-by-nrhs and m >= n.

    % Extract dimensions.
    [m, n] = size(A);
    [m1, nrhs] = size(B);

    if m < n
        error('Expect A to have more rows than columns');
    end

    if m ~= m1
        error('A and B do not have the conforming dimensions');
    end

    blksz = 32;

    for j=1:blksz:n
        % Compute width of j-th panel.
        b = min(n-j+1, blksz);
        
        % QR decomposition of panel.
        [Qj, A(j:m, j:j+b-1)] = qr(A(j:m, j:j+b-1));
        
        % A := Q^T * A
        A(j:m, j+b:n) = Qj' * A(j:m, j+b:n);

        % B := Q^T * B
        B(j:m, 1:nrhs) = Qj' * B(j:m, 1:nrhs);
    end
    R = triu(A);
    X = R(1:n,1:n)\B(1:n, 1:nrhs);
end
