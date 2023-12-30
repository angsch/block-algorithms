function [B, Q, P] = blockReduceToTriangularBand(A, blksz)
%BLOCKREDUCETOTRIANGULARBAND   Reduction to triangular band form.
%    [B, Q, P] = BLOCKREDUCETOTRIANGULARBAND(A) reduces an m-by-n matrix A to
%    triangular band matrix B by an orthogonal transformation Q' * A * P = B.
%    If m >= n, B is upper triangular; if m < n, B is lower triangular.
%    The width of the band corresponds to blksz.

    if nargin < 2
        blksz = 20;
    end

    B = A;

    % Extract dimensions.
    [m, n] = size(A);

    Q = eye(m);
    P = eye(n);

    if m >= n
        % Reduce to upper triangular band form.
        for j=1:blksz:n
            % Compute width of j-th panel.
            b = min(n-j+1, blksz);

            % QR factorization of current block column.
            [Qj, B(j:m, j:j+b-1)] = qr(B(j:m, j:j+b-1));
            
            % Trailing matrix update with Q.
            B(j:m, j+b:n) = Qj' * B(j:m, j+b:n);
            
            % Update Q.
            Q = Q * [eye(j-1),         zeros(j-1,m-j+1);
                     zeros(m-j+1,j-1), Qj               ];

            % LQ factorization of the current block row.
            [B(j:min(m,j+b-1),j+b:n), Pj] = blockLQ(B(j:min(m,j+b-1),j+b:n));
           
            % Trailing matrix update wih P.
            B(j+b:m,j+b:n) = B(j+b:m,j+b:n) * Pj';

            % Update P.
            P = P * [ eye(j+b-1),            zeros(j+b-1,n-j-b+1);
                      zeros(n-j-b+1,j+b-1),  Pj'                  ];
        end
    else
        % Reduce to lower triangular band form.
        for i=1:blksz:m
            % Compute height of i-th block row.
            b = min(m-i+1, blksz);

            % LQ decomposition of the current block row.
            [B(i:i+b-1,i:n), Pi] = blockLQ(B(i:i+b-1,i:n));

            % Trailing matrix update with P.
            B(i+b:m,i:n) = B(i+b:m,i:n) * Pi';

            % Update P.
            P = P * [ eye(i-1),          zeros(i-1,n-i+1);
                      zeros(n-i+1,i-1),  Pi'               ];

            % QR factorization of current block column.
            [Qi, B(i+b:m, i:i+b-1)] = qr(B(i+b:m, i:i+b-1));

            % Trailing matrix update with Q.
            B(i+b:m, i+b:n) = Qi' * B(i+b:m, i+b:n);

            % Update Q.
            Q = Q * [eye(i+b-1),           zeros(i+b-1,m-i-b+1);
                     zeros(m-b-i+1,i+b-1), Qi                    ];
        end
    end
end
