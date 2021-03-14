function [R, Q] = blockRQ(A)
%BLOCKRQ   Triangular-orthogonal decomposition.
%    [R, Q] = BLOCKRQ(A), where A is an m-by-n matrix, produces an m-by-n upper
%    triangular matrix R and an n-by-n orthogonal matrix Q so that A = R*Q.

    % Extract dimensions.
    [m, n] = size(A);
    k = min(m, n);

    blksz = 120;

    Q = eye(n);

    if blksz > k
        [R, Q] = smallRQ(A);
    else
        % Blocked algorithm.

        % Range of block rows that are transformed with the blocked algorithm.
        num_blk_rows = floor(k / blksz);
        last = m - num_blk_rows * blksz + 1;

        % Set the column update range 1:j.
        j = n;
        for i = m - blksz + 1 : -blksz : last
            % RQ decomposition of current block row.
            [A(i:i+blksz-1, 1:j), Qj] = smallRQ(A(i:i+blksz-1, 1:j));

            % Trailing matrix update.
            A(1:i-1, 1:j) = A(1:i-1, 1:j) * Qj';

            % Update Q.
            Q = [Qj,           zeros(j,n-j);
                 zeros(n-j,j), eye(n-j,n-j)] * Q;

            % Shrink the column update range.
            j = j - blksz;
        end

        % Remaining rows.
        [A(1:last-1, 1:j), Q2] = smallRQ(A(1:last-1, 1:j));
        Q = [Q2,           zeros(j,n-j);
             zeros(n-j,j), eye(n-j,n-j)] * Q;
        R = A;
    end
end

function [R, Q] = smallRQ(A)
    % Reverse the rows of A.
    A1 = flipud(A);

    % Compute QR decomposition.
    [Q1, R1] = qr(transpose(A1));

    % Reverse the rows of Q1.
    Q = flipud(transpose(Q1));

    % Reverse rows of R1^T.
    R1 = flipud(transpose(R1));

    % Reverse columns of R1.
    R = fliplr(R1);
end
