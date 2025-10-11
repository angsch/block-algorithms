function [V, LAMBDA, nsweep] = eigJacobi(A, tol)
%EIGJACOBI   Symmetric eigenvalue decomposition
%    [V, LAMBDA, nsweep] = EIGJACOBI(A), where A is a symmetric
%    or Hermitian matrix, computes the eigendecompositions V'*A*V=LAMBDA
%    using Jacobi's method.
%

    if ~ishermitian(A)
        error('A must be symmetric/Hermitian');
    end

    % Extract dimensions.
    [n, ~] = size(A);

    if nargin < 2
        tol = sqrt(n) * eps; % Tolerance when to stop
    end

    %
    % Single precision preconditioner.
    %
    Afp32 = single(A);
    [Vfp32, Dfp32] = eig(Afp32);
    % Reorthogonalize Vfp32 to have an orthogonal basis in FP64.
    [V,~] = qr(double(Vfp32));
    Aprecond = V' * A * V;
    
    [V, LAMBDA, nsweep] = jacobi(Aprecond, V, tol);
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

function [V, LAMBDA, nsweep] = jacobi(A, V, tol)
    if nargin < 3
        tol = eps;
    end

    [n, ~] = size(A);
    nsweep = -1;
    maxIter = 30;
    done = false;

    while ~done && nsweep < maxIter
        nsweep = nsweep + 1;
        done = true;

        % Row cyclic Jacobi (upper triangular)
        for p = 1:n-1
            for q = p+1:n
                if abs(A(p,q)) > tol*sqrt(abs(A(p,p)))*sqrt(abs(A(q,q)))
                    done = false;

                    % Form Jacobi rotation.
                    J = jacobi_rotation([A(p,p), A(p,q);
                                         A(p,q)', A(q,q) ]);
                    
                    % Apply rotation to A as similarity transformation.
                    A([p,q],:) = J' * A([p,q],:);
                    A(:,[p,q]) = A(:,[p,q]) * J;
                    A(p,q) = 0.0;
                    A(q,p) = 0.0;

                    % Update V from the right.
                    V(:,[p,q]) = V(:,[p,q]) * J;
                end
            end
        end
    end
    LAMBDA = diag(A);

    % Sort eigenvalues.
    [~,ind] = sort(LAMBDA, 'ascend');
    LAMBDA = LAMBDA(ind);
    V = V(:,ind);
end
