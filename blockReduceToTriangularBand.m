function [B, Q, P] = blockReduceToTriangularBand(A, blksz)
%BLOCKREDUCETOTRIANGULARBAND   Reduction to triangular band form.
%    [B, Q, P] = BLOCKREDUCETOTRIANGULARBAND(A) reduces an m-by-n matrix A, m >= n, to
%    upper triangular band form B by an orthogonal transformation Q' * A * P = B.
%    The width of the band corresponds to blksz.

    if nargin < 2
        blksz = 4;
    end

    B = A;

    % Extract dimensions.
    [m, n] = size(A);

    Q = eye(m);
    P = eye(n);

    if m >= n
        for j=1:blksz:n
            % Compute width of j-th panel.
            b = min(n-j+1, blksz);

            % QR factorization of current block column.
            [Qj, B(j:m, j:j+b-1)] = qr(B(j:m, j:j+b-1));
            
            % Trailing matrix update.
            B(j:m, j+b:n) = Qj' * B(j:m, j+b:n);
            
            % Update Q.
            Q = Q * [eye(j-1),         zeros(j-1,m-j+1);
                     zeros(m-j+1,j-1), Qj               ];

            % LQ factorization of the current block row.
            [B(j:min(m,j+b-1),j+b:n), Pj] = blockLQ(B(j:min(m,j+b-1),j+b:n));
           
            % Trailing matrix update.
            B(j+b:m,j+b:n) = B(j+b:m,j+b:n) * Pj';

            % Update P.
            P = P * [ eye(j+b-1),            zeros(j+b-1,n-j-b+1);
                      zeros(n-j-b+1,j+b-1),  Pj'                  ];
        end
    else
        % TODO: reduce to lower triangular band form
    end
end
