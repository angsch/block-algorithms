function [Q, L] = blockQL(A)
%BLOCKQL   Orthogonal-triangular decomposition.
%    [Q, L] = BLOCKQL(A), where A is an m-by-n matrix, produces an m-by-m
%    unitary matrix Q and an m-by-n lower triangular matrix L so that A = Q*L.

    % Extract dimensions.
    [m, n] = size(A);
    k = min(m, n);

    blksz = 100;

    if blksz > k
        [Q, L] = smallQL(A);
    else
        % Blocked algorithm.
        Q = eye(m);

        % Range of block columns to be transformed with the blocked algorithm.
        num_blk_cols = floor(k / blksz);
        last = n - num_blk_cols * blksz + 1;

        % Set the row update range 1:i.
        i = m;
        for j = n-blksz+1:-blksz:last
            % QL decomposition of panel.
            [Qj, A(1:i, j:j+blksz-1)] = smallQL(A(1:i, j:j+blksz-1));

            % Trailing matrix update.
            A(1:i, 1:j-1) = Qj' * A(1:i, 1:j-1);

            % Update Q.
            Q = Q * [Qj,            zeros(i, m-i)
                     zeros(m-i, i), eye(m-i)     ];

            % Shrink the row update range.
            i = i - blksz;
        end

        % Remaining columns.
        [Q1, A(1:i, 1:last-1)] = smallQL(A(1:i, 1:last-1));
        Q = Q * [Q1,            zeros(i, m-i)
                 zeros(m-i, i), eye(m-i)     ];
        L = A;
    end
end

function [Q, L] = smallQL(A)
    A1 = flipud(transpose(A));
    [Q1, R1] = qr(transpose(A1));
    Q = flipud(transpose(Q1));
    R1 = flipud(transpose(R1));
    R = fliplr(R1);
    Q = transpose(Q);
    L = transpose(R);
end
