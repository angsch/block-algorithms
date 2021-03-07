function [L, U, P] = blockLU(A)
% BLOCKLU  Compute an LU factorization with partial pivoting.
%    [L,U,P] = BLOCKLU(A) returns a unit lower triangular matrix L, an upper
%    triangular matrix U, and a permutation matrix P so that P*A = L*U.
%    All matrices a square.

    [n, ~] = size(A);
    blksz = 120;

    P = eye(n);
    L = eye(n);
    U = zeros(n, n);

    if blksz > n
        % Use Matlab's LU decomposition for small matrices.
        [L, U, P] = lu(A);
    else
        % Compute a panel factorization and update the trailing matrix with
        % matrix-matrix multiplications.

        % Loop over block columns.
        for k = 1:blksz:n
            % Compute width of k-th panel.
            b = min(n-k+1, blksz);

            % LU decomposition of panel.
            [L(k:n, k:k+b-1), U(k:k+b-1, k:k+b-1), Pk] = lu(A(k:n, k:k+b-1));

            % Apply permutation to trailing matrix. (naive)
            A(k:n, k+b:n) = Pk * A(k:n, k+b:n);

            % Apply permutation to already processed part of L. (naive)
            L(k:n, 1:k-1) = Pk * L(k:n, 1:k-1);

            % Merge permutations: P = Pk * P.
            P = [eye(k-1),          zeros(k-1,n-k+1); 
                 zeros(n-k+1, k-1), Pk               ] * P;

            % Right update.
            for j = k+b:b:n
                % Compute dimensions of tile kj.
                w = min(n-j+1, blksz);

                % Ukj = inv(Lkk) * Akj.
                U(k:k+b-1, j:j+w-1) = L(k:k+b-1, k:k+b-1) \ A(k:k+b-1, j:j+w-1);
            end

            % Tiled trailing matrix update.
            for i = k+b:b:n
                for j = k+b:b:n
                    % Compute dimensions of tile ij.
                    h = min(n-i+1, blksz);
                    w = min(n-j+1, blksz);

                    % Aij := Aij - Lik * Ukj.
                    A(i:i+h-1, j:j+w-1) = A(i:i+h-1, j:j+w-1) ...
                        - L(i:i+h-1, k:k+b-1) * U(k:k+b-1, j:j+w-1);
                end
            end
        end
    end
end
