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

    % Stage 1: Reduce the general matrix to a triangular band matrix.
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

    % Stage 2: Reduce the triangular band matrix to a bidiagonal matrix.
    [B, Q, P] = bandToBidiagonal(B, Q, P);
end


function [B, Q, P] = bandToBidiagonal(A, Q, P)
    % Extract dimensions.
    [m, n] = size(A);

    if m >= n
        % Extract the bandwidth.
        bw = nnz(A(1, :)) - 1;

        % Reduce band matrix to upper bidiagonal form row by row.
        for row = 1:n-2
            % Generate elementary reflector to annihilate A(row,row+2:n)
            % and introduce bulge.
            r = -sign(A(row,row+1)) * norm(A(row, row+1:n));
            [v, tau] = gallery('house', A(row, row+1:n));
            A(row, row+1:n) = [r, zeros(1,n-row-1)];

            % Right update.
            A(row+1:m,row+1:n) = ...
                A(row+1:m,row+1:n) - tau * (A(row+1:m,row+1:n) * v) * v';

            % Accumulate reflectors for P.
            P(:,row+1:n) = P(:,row+1:n) - tau * (P(:,row+1:n) * v) * v';

            % Chase the bulge off.
            j1 = row+1;
            for i = row+1:bw:m
                if j1 >= n
                    continue;
                end

                % Generate elementary reflector to annihilate fill-in A(i+1:m,j1).
                r = -sign(A(i,j1)) * norm(A(i:m, j1));
                [v, tau] = gallery('house', A(i:m, j1));
                A(i:m, j1) = [r; zeros(m-i,1)];

                % Left update.
                A(i:m,j1+1:n) = A(i:m,j1+1:n) - tau * v * (v' * A(i:m,j1+1:n));

                % Accumulate reflectors for Q.
                Q(:,j1:m) = Q(:,j1:m) - tau * (Q(:,j1:m) * v) * v';

                % Advance j1 to newly generated fill-in.
                j1 = j1 + bw;

                if j1 < n
                    % Generate elementary reflector to annihilate fill-in A(i,j1:n).
                    r = -sign(A(i,j1)) * norm(A(i, j1:n));
                    [v, tau] = gallery('house', A(i, j1:n));
                    A(i, j1:n) = [r, zeros(1,n-j1)];

                    % Right update.
                    A(i+1:m,j1:n) = A(i+1:m,j1:n) - tau * (A(i+1:m,j1:n) * v) * v';

                    % Accumulate reflectors for P.
                    P(:,j1:n) = P(:,j1:n) - tau * (P(:,j1:n) * v) * v';
                end
            end
        end
    else
        % Extract the bandwidth.
        bw = nnz(A(:, 1)) - 1;

        % Reduce band matrix to upper bidiagonal form column by column.
        for col = 1:m-2
            % Generate elementary reflector H to annihilate A(col+2:m,col)
            % and introduce bulge.
            r = -sign(A(col+1,col)) * norm(A(col+1:m, col));
            [v, tau] = gallery('house', A(col+1:m,col));
            A(col+1:m,col) = [r; zeros(m-col-1,1)];

            % Left update.
            A(col+1:m,col+1:n) = ...
                A(col+1:m,col+1:n) - tau * v * (v' * A(col+1:m,col+1:n));

            % Accumulate reflectors for Q.
            Q(:,col+1:m) = Q(:,col+1:m) - tau * (Q(:,col+1:m) * v) * v';

            % Chase the bulge off.
            i = col+1;
            for j1 = col+1:bw:n
                if i >= m
                    continue;
                end

                % Generate elementary reflector to annihilate fill-in A(i,j1:n).
                r = -sign(A(i,j1)) * norm(A(i, j1:n));
                [v, tau] = gallery('house', A(i, j1:n));
                A(i, j1:n) = [r, zeros(1,n-j1)];

                % Right update.
                A(i+1:m,j1:n) = A(i+1:m,j1:n) - tau * (A(i+1:m,j1:n) * v) * v';

                % Accumulate reflectors for P.
                P(:,j1:n) = P(:,j1:n) - tau * (P(:,j1:n) * v) * v';
                
                i = i + bw;
                if i < m
                    % Generate elementary reflector to annihilate fill-in A(i+1:m,j1).
                    r = -sign(A(i,j1)) * norm(A(i:m, j1));
                    [v, tau] = gallery('house', A(i:m, j1));
                    A(i:m, j1) = [r; zeros(m-i,1)];

                    % Left update.
                    A(i:m,j1+1:n) = A(i:m,j1+1:n) - tau * v * (v' * A(i:m,j1+1:n));

                    % Accumulate reflectors for Q.
                    Q(:,i:m) = Q(:,i:m) - tau * (Q(:,i:m) * v) * v';
                end
            end
        end
        % Transform to upper bidiagonal form.
        for i = 1:m-1
            % Apply Givens rotation annihilating A(i+1,i) from the left.
            [c, s] = givens(A(i,  i), A(i+1,i));
            A(i:i+1,i:i+1) = [ c  s;
                              -s  c] * A(i:i+1,i:i+1);
            A(i+1,i) = 0.0;
            Q(:,i:i+1) = Q(:,i:i+1) * [ c -s;
                                        s  c ];
        end
    end

    B = A;
end

