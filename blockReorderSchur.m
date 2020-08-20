function [QS, TS] = blockReorderSchur(Q, T, select)
% BLOCKREORDERSCHUR  Reorder eigenvalues of a Schur factorization.
%    [QS,TS] = BLOCKREORDERSCHUR(Q,T,SELECT) reorders the complex Schur
%    factorization A = Q*T*Q' of a non-symmetric square matrix A so that
%    the selected eigenvalues appear tightly packed in the upper left corner
%    of the complex upper triangular Schur matrix T. The corresponding
%    invariant subspace is spanned by the leading columns of QS. The logical
%    vector SELECT marks the 1-by-1 blocks on the diagonal of T that shall 
%    appear in the top-left corner after reordering. The unitary matrix QS
%    accumulates all similarity transformations applied during the reordering.
   
    [n, ~] = size(T);

    blksz = 120;

    if blksz > n
        % Use Matlab's reorder function for small matrices.
        [QS, TS] = ordschur(Q, T, select);
    else
        % Use reordering in window + DGEMM for off-diagonal matrix updates.
        reordered = false;

        QS = Q;

        while ~reordered

            % Odd windows.
            for k = 1:blksz:n
                % Compute width of window.
                b = min(n-k+1, blksz);

                % Reorder window unless if it already reordered.
                if ~is_reordered(select(k:k+b-1))
                    % Reorder window.
                    Qk = eye(b);
                    [Qk, T(k:k+b-1, k:k+b-1)] = ...
                        ordschur(Qk, T(k:k+b-1, k:k+b-1), select(k:k+b-1));
                    
                    % Reorder select accordingly.
                    select(k:k+b-1) = sort(select(k:k+b-1), 'descend');
                    
                    % Update from the left T := Qk^H * T.
                    T(k:k+b-1, k+b:n) = Qk' * T(k:k+b-1, k+b:n);
                    
                    % Update from the right T := T * Qk.
                    T(1:k-1,k:k+b-1) = T(1:k-1,k:k+b-1) * Qk;
                    
                    % Update Q.
                    QS(1:n, k:k+b-1) = QS(1:n, k:k+b-1) * Qk;
                end
            end

            % Even windows.
            for k = blksz/2+1:blksz:n-blksz/2+1
                % Compute width of window.
                b = min(n-k+1, blksz);

                % Reorder window unless if it already reordered.
                if ~is_reordered(select(k:k+b-1))
                    % Reorder window.
                    Qk = eye(b);
                    [Qk, T(k:k+b-1, k:k+b-1)] = ...
                        ordschur(Qk, T(k:k+b-1, k:k+b-1), select(k:k+b-1));
                    
                    % Reorder select accordingly.
                    select(k:k+b-1) = sort(select(k:k+b-1), 'descend');
                    
                    % Update from the left T := Qk^H * T.
                    T(k:k+b-1, k+b:n) = Qk' * T(k:k+b-1, k+b:n);
                    
                    % Update from the right T := T * Qk.
                    T(1:k-1,k:k+b-1) = T(1:k-1,k:k+b-1) * Qk;
                    
                    % Update Q.
                    QS(1:n, k:k+b-1) = QS(1:n, k:k+b-1) * Qk;
                end
            end

            % Check if we are done.
            reordered = is_reordered(select);
        end
        TS = T;
    end
end

function ns = count_selected(select)
    ns = sum(select);
end

function reordered = is_reordered(select)
    reordered = false;

    % Count selected in the entire array.
    ns = count_selected(select);

    % Check if all selected are in the first ns positions.
    if count_selected(select(1:ns)) == ns
        reordered = true;
    end
end

