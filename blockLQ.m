function [L, Q] = blockLQ(A)
%BLOCKLQ   Triangular-orthogonal decomposition.
%    [L, Q] = BLOCKLQ(A), where A is an m-by-n matrix, produces an m-by-n lower
%    triangular matrix L and an n-by-n orthogonal matrix Q so that A = L*Q.

    % Extract dimensions.
    [m, n] = size(A);
    k = min(m, n);

    blksz = 120;

    if blksz > k
        [L, Q] = smallLQ(A);
    else
        % Blocked algorithm.
        Q = eye(n);
        for i=1:blksz:k
            % LQ decomposition of the current block row.
            [A(i:min(m,i+blksz-1),i:n), Qj] = smallLQ(A(i:min(m,i+blksz-1),i:n));

            % Trailing matrix update. A * Qj'
            A(i+blksz:m,i:n) = A(i+blksz:m,i:n) * Qj';

            % Update Q.
            Q = [ eye(i-1),          zeros(i-1,n-i+1);
                  zeros(n-i+1,i-1),  Qj                ] * Q;
        end
        L = tril(A);
    end
end

function [L, Q] = smallLQ(A)
    [Q, R] = qr(transpose(A));
    L = transpose(R);
    Q = transpose(Q);
end
