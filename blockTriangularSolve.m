function X = blockTriangularSolve(T, B, shape)
%BLOCKTRIANGUALRSOLVE   Solves a triangular system T * X = B.
%    BLOCKTRIANGUALRSOLVE(T, B) solves T * X = B for X when T is a square
%    upper triangular matrix with the same number of rows as B.
%
%    X = BLOCKTRIANGUALRSOLVE(T, B, 'lower') solves T * X = B when T
%    is a square lower triangular matrix.
%
%    Right-looking variant.

    if nargin < 3
        shape = 'lower';
    end

    if shape ~= 'lower' && shape ~= 'upper'
        error('Shape must upper or lower.');
    end

    % Extract dimensions.
    [m, n] = size(T);
    if m ~= n
        error('The matrix T must be square.');
    end
    [m, nrhs] = size(B);
    if m ~= n
        error('Number or columns of T and number of rows of B are nonconformant.');
    end

    blksz = 60;
    if shape == 'lower'
        for j = 1:blksz:n
            % Compute block width.
            b = min(n-j+1, blksz);

            % Partition [T11    ], [B1]
            %           [T21 T22]  [B2] and solve T11 * X = B1 for X.
            B(j:j+b-1,:) = T(j:j+b-1,j:j+b-1) \ B(j:j+b-1,:);

            % Trailing matrix update B2 := B2 - T21 * B1.
            B(j+b:n,:) = B(j+b:n,:) - T(j+b:n,j:j+b-1) * B(j:j+b-1,:);
        end
    else % shape == 'upper'
        for j = n-blksz+1:-blksz:-blksz+2
            % Compute block width.
            jold = j+blksz;
            j = max(1,j);
            b = min(jold-j, blksz);

            % Partition [T11 T12], [B1]
            %           [    T22]  [B2] and solve T22 * X = B2 for X.
            B(j:j+b-1,:) = T(j:j+b-1,j:j+b-1) \ B(j:j+b-1,:);

            % Trailing matrix update B1 := B1 - T12 * B2.
            B(1:j-1,:) = B(1:j-1,:) - T(1:j-1,j:j+b-1) * B(j:j+b-1,:);
        end
    end
    X = B;
end
