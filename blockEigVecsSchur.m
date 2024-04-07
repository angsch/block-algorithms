function V = blockEigVecsSchur(T)
% BLOCKEIGVECSSCHUR Computes right eigenvectors of the Schur matrix T.
%   V = BLOCKEIGVECSSCHUR(T) computes eigenvectors of a triangular matrix
%   T such that T*V(:,k) = T(k,k) * V(:,k).

    [n, _] = size(T);
    blksz = 32;

    % Allocate memory.
    V = zeros(n,n);

    num_blks = ceil(n/blksz);
    for k = num_blks:-1:1
        % Locate k-th block column spanning k1:k2
        k1 = (k-1)*blksz+1;
        k2 = min(k*blksz,n);

        % Compute eigenvectors corresponding to eigenvalues
        % of current block column.
        shifts = diag(T(k1:k2,k1:k2));

        for j = k:-1:1
            j1 = (j-1)*blksz+1;
            j2 = min(j*blksz,n);
            if j == k
                % Find eigenvectors of block of T.
                V(j1:j2,k1:k2) = eigVecs(T(k1:k2,k1:k2));
            else
                % Triangular solve.
                V(j1:j2,k1:k2) = ...
                    shiftedSolve(T(j1:j2,j1:j2), shifts, V(j1:j2,k1:k2));
            end
            % Linear update.
            V(1:j1-1,k1:k2) = ...
                V(1:j1-1,k1:k2) - T(1:j1-1,j1:j2) * V(j1:j2,k1:k2); 
        end
    end

    % Normalize eigenvectors.
    for k = 1:n
        V(:,k) = V(:,k) / norm(V(:,k));
    end
end

function V = eigVecs(T)
    [n,_] = size(T);
    % Form right-hand side.
    V = eye(n) - triu(T,1);

    % Solve small eigenvector problem.
    for k = n:-1:1
        lambda = T(k,k);
        for j = k-1:-1:1
            V(j,k) = V(j,k) / (T(j,j)-lambda);
            V(1:j-1,k) = V(1:j-1,k) - T(1:j-1,j)*V(j,k);
        end
    end
end

function V = shiftedSolve(T, shifts, V)
    [m,n] = size(V);
    for k = 1:n
        V(:,k) = (T - shifts(k)*eye(m)) \ V(:,k);
    end
end
