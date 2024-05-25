function varargout = blockEigVecsSchur(T)
% BLOCKEIGVECSSCHUR Computes eigenvectors of the Schur matrix T.
%   --VR = BLOCKEIGVECSSCHUR(T) computes right eigenvectors of a
%     triangular matrix T such that T*VR(:,k) = T(k,k) * VR(:,k).
%   --[VR, VL] = BLOCKEIGVECSSCHUR(T) computes right eigenvectors VR
%     and left eigenvectors VL such that T*VR(:,k) = T(k,k) * VR(:,k)
%     and VL(:,k)'*T = T(k,k)*VL(:,k)'.


    [n, _] = size(T);
    blksz = 32;

    % Right eigenvectors
    VR = zeros(n,n);
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
                VR(j1:j2,k1:k2) = rightEigVecs(T(k1:k2,k1:k2));
            else
                % Triangular solve.
                VR(j1:j2,k1:k2) = ...
                    shiftedSolve(T(j1:j2,j1:j2), shifts, VR(j1:j2,k1:k2));
            end
            % Linear update.
            VR(1:j1-1,k1:k2) = ...
                VR(1:j1-1,k1:k2) - T(1:j1-1,j1:j2) * VR(j1:j2,k1:k2);
        end
    end

    % Normalize right eigenvectors.
    for k = 1:n
        VR(:,k) = VR(:,k) / norm(VR(:,k));
    end
    varargout{1} = VR;

    % Left eigenvectors
    if nargout > 1
        VL = zeros(n,n);
        for j = 1:num_blks
            j1 = (j-1)*blksz+1;
            j2 = min(j*blksz,n);
            for k = 1:j
                % Locate block column k1:k2
                k1 = (k-1)*blksz+1;
                k2 = min(k*blksz,n);
                if j == k
                    VL(k1:k2,k1:k2) = leftEigVecs(T(k1:k2,k1:k2));
                else
                    shifts = diag(T(k1:k2,k1:k2));
                    VL(j1:j2,k1:k2) = ...
                        shiftedSolveVL(T(j1:j2,j1:j2), shifts, VL(j1:j2,k1:k2));
                end
                % Linear update.
                VL(j2+1:n,k1:k2) = ...
                    VL(j2+1:n,k1:k2) - T(j1:j2,j2+1:n)' * VL(j1:j2,k1:k2);
            end
        end

        % Normalize
        for k = 1:n
            VL(:,k) = VL(:,k) / norm(VL(:,k));
        end

        varargout{2} = VL;
    end
end

function VR = rightEigVecs(T)
    [n,_] = size(T);
    % Form right-hand side.
    VR = eye(n) - triu(T,1);

    % Solve small eigenvector problem.
    for k = n:-1:1
        lambda = T(k,k);
        for j = k-1:-1:1
            VR(j,k) = VR(j,k) / (T(j,j)-lambda);
            VR(1:j-1,k) = VR(1:j-1,k) - T(1:j-1,j)*VR(j,k);
        end
    end
end

function VL = leftEigVecs(T)
    [n,_] = size(T);
    % [ 0 1 y ] * [ T11  t12   T13 ] = lambda * [0 1 y ]
    %             [     lambda t23 ]
    %             [            T33 ]

    % Form right-hand side.
    VL = (eye(n)-triu(T,1))';

    for k = 1:n
        lambda = T(k,k);
        % Forward solve.
        for j = k+1:n
            VL(j,k) = (T(j,j)-lambda)' \ VL(j,k);
            VL(j+1:n,k) = VL(j+1:n,k) - VL(j,k) * T(j,j+1:n)';
        end
    end
end

function VR = shiftedSolve(T, shifts, VR)
    [m,n] = size(VR);
    for k = 1:n
        VR(:,k) = (T - shifts(k)*eye(m)) \ VR(:,k);
    end
end

function VL = shiftedSolveVL(T, shifts, VL)
    [m,n] = size(VL);
    for k = 1:n
        VL(:,k) = (T-shifts(k)*eye(m))' \ VL(:,k);
    end
end
