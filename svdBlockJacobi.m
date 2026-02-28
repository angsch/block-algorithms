function [U, S, V] = svdBlockJacobi(A)
%SVDBLOCKJACOBI   Singular value decomposition
%    [U, S, V] = SVDBLOCKJACOBI(A), computes a singular value
%    decomposition of A using a block Jacobi's method. That is,
%    A = U * S * V'. The matrix A is assumed to be square.

    % Extract dimensions.
    [m, n] = size(A);
    if m ~= n
        error('The matrix A must be square.');
    end

    tol = sqrt(n) * eps;

    V = eye(n);
    [U, S, V] = blockSvdJacobi1Sided(A, V, tol);
end


function [V, wasUpdated] = smallBlockJacobi1Sided(pIdxSet, qIdxSet, A, tol)
%SMALLBLOCKJACOBI1SIDED   One-sided Jacobi's method on a small block
%    [V, wasUpdated] = SMALLBLOCKJACOBI1SIDED(pIdxSet,qIdxSet, A), computes
%    the unitary update matrix to be applied to A(:,[pIdxSet,qIdxSet]).
%    The flag wasUpdated signals if the block is converged or has to be
%    revisited.

    % Compute B, the (p,q) submatrix of the Gram matrix A'*A.
    Ap = A(:, pIdxSet);
    Aq = A(:, qIdxSet);
    B = [Ap'*Ap, Ap'*Aq;
         Aq'*Ap, Aq'*Aq];

    % Extract dimensions.
    [n, ~] = size(B);

    V = eye(n);

    nsweep = -1;
    maxIter = 2;  % let's not accumulate too much error
    done = false;
    wasUpdated = false;
    while ~done && nsweep < maxIter
        nsweep = nsweep + 1;
        done = true;

        % Row cyclic Jacobi
        for p = 1:n
            for q = p+1:n
                % Work on (p,q) submatrix [ a  c ]
                %                         [ c' b ].
                a = B(p,p); c = B(p,q);
                            b = B(q,q);

                % Convergence criterion as in LAWN15.
                if abs(c) > tol*sqrt(abs(a))*sqrt(abs(b))
                    done = false;
                    wasUpdated = true;

                    % Form Jacobi rotation.
                    J = jacobi_rotation( [a c;
                                          c b ]);

                    % Apply rotation to B from the right.
                    B(:,[p,q]) = B(:,[p,q]) * J;

                    % Apply rotation to B from the left.
                    B([p,q],:) = J' * B([p,q],:);

                    % Update V from the right.
                    V(:,[p,q]) = V(:,[p,q]) * J;
                end
            end
        end
    end
end

function [U, S, V] = blockSvdJacobi1Sided(A, V, tol)
%BLOCKSVDJACOBI1SIDED   One-sided block Jacobi's method
%    [U, S, V] = BLOCKSVDJACOBI1SIDED(A,V,tol), computes
%    a singular value decomposition of A using a one-sided
%    block Jacobi's method.

    % Extract dimensions.
    [n,~] = size(A);

    blksz = 8;
    num_blk_rows = ceil(n/blksz);
    num_blk_cols = ceil(n/blksz);

    nsweep = -1;
    maxIter = 30;
    done = false;

    % TODO: scale A to prevent overflow in the largest singular values.

    while ~done && nsweep < maxIter
        nsweep = nsweep + 1;
        rotationsThisSweep = 0;

        % Block row cyclic Jacobi.
        for p = 1:num_blk_rows
            for q = p+1:num_blk_cols
                % Compute the index sets for the (p,q) submatrix (1-based)
                first = (p-1) * blksz+1;
                last = min(n, p * blksz);
                pIdxSet = first:last;
                first = (q-1) * blksz+1;
                last = min(n, q * blksz);
                qIdxSet = first:last;

                % Compute block Jacobi transformation.
                [J, wasUpdated] = smallBlockJacobi1Sided(pIdxSet, qIdxSet, A, tol);

                % Track how many blocks are converged/are updated
                if wasUpdated
                    rotationsThisSweep = rotationsThisSweep + 1;
                end

                % Apply rotation to A from the right.
                A(:,[pIdxSet,qIdxSet]) = A(:,[pIdxSet,qIdxSet]) * J;

                % Update V from the right.
                V(:,[pIdxSet,qIdxSet]) = V(:,[pIdxSet,qIdxSet]) * J;

            end
        end

        % Convergence check.
        done = (rotationsThisSweep == 0);
    end
    % A now holds U * S. Extract singular values S.
    S = zeros(n,1);

    % Preallocate U.
    U = zeros(n,n);

    % TODO: guard against zero singular values
    for j = 1:n
        S(j) = norm(A(:,j));
        U(:,j) = A(:,j) / S(j);
    end

    % Sort singular values and corresponding singular vectors.
    [~,ind] = sort(S, 'descend');
    S = S(ind);
    V = V(:,ind);
    U = U(:,ind);
end

function J = jacobi_rotation(A)
% JACOBI   Compute a Jacobi rotation.
%    J = JACOBI(A) computes a 2-by-2 similarity
%    transformation (Jacobi rotation) such that
%    J' * A * J is diagonal.

    if A(1,2) ~= 0.0
        [J, ~] = eig(A);
    else
        % A is already diagonal.
        J = eye(2,2);
    end
end
