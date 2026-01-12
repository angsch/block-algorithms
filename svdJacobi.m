function [U, S, V] = svdJacobi(A, options)
%SVDJACOBI   Singular value decomposition
%    [U, S, V] = SVDJACOBI(A), computes a singular value
%    decomposition of A using Jacobi's method. That is,
%    A = U * S * V'. The matrix A is assumed to be square.
%    The optional argument options is a structure that can be
%    used to configure the algorithm by setting the following
%    fields:
%    - tol:      Tolerance when to stop.
%    - oneSided: If set to true, use the one-sided Jacobi algorithm,
%                otherwise, use two-sided Jacobi algorithm.
%    - precond:  If set to true, use preconditioning,
%                otherwise, no preconditioning is used.
%

    % Extract dimensions.
    [m, n] = size(A);
    if m ~= n
        error('The matrix A must be square.');
    end

    if nargin < 2
        options = struct();
    end
    % Populate all options
    if ~isfield(options, 'tol'),      options.tol = sqrt(n) * eps; end
    if ~isfield(options, 'oneSided'), options.oneSided = false;    end
    if ~isfield(options, 'precond'),  options.precond = true;      end

    if options.precond
        % Adjust tolerance to single precision preconditioner.
        tol = sqrt(n) * eps(single(1));

        % Single-precision preconditioner.
        Afp32 = single(A);
        if options.oneSided
            V = single(eye(n));
            [Ufp32, Sfp32, Vfp32] = svdJacobi1Sided(Afp32, V, tol);
        else
            U = single(eye(n));
            V = single(eye(n));
            [Ufp32, Sfp32, Vfp32] = svdJacobi2Sided(Afp32, U, V, tol);
        end
        % Reorthogonalize Vfp32, Ufp32.
        V0 = newtonSchulz(double(Vfp32));
        U0 = newtonSchulz(double(Ufp32));
        Aprecond = U0' * A * V0;
    else
        Aprecond = A;
        U0 = eye(n);
        V0 = eye(n);
    end


    % Set the tolerance to the user-defined precision
    tol = options.tol;
    if options.oneSided
        [U, S, V] = svdJacobi1Sided(Aprecond, V0, tol);
        U = U0 * U;
    else
        [U, S, V] = svdJacobi2Sided(Aprecond, U0, V0, tol);
    end

    % Sort singular values and corresponding singular vectors.
    [~,ind] = sort(S, 'descend');
    S = S(ind);
    V = V(:,ind);
    U = U(:,ind);
end


function [U, S, V] = svdJacobi1Sided(A, V, tol)
    % Extract dimensions.
    [n, ~] = size(A);

    nsweep = -1;
    maxIter = 30;
    done = false;

    % TODO: scale A to prevent overflow in the largest singular values

    while ~done && nsweep < maxIter
        nsweep = nsweep + 1;
        done = true;

        % Row cyclic Jacobi
        for p = 1:n
            for q = p+1:n
                % Compute [ a c ], the (p,q) submatrix of the Gram matrix A'*A.
                %         [ c b ]
                a = A(:,p)'*A(:,p);
                c = A(:,p)'*A(:,q);
                b = A(:,q)'*A(:,q);

                % Convergence criterion as in LAWN15                
                if abs(c) > tol*sqrt(abs(a))*sqrt(abs(b))
                    done = false;

                    % Form Jacobi rotation.
                    J = jacobi_rotation( [a c;
                                          c b ]);
                    
                    % Apply rotation to A from the right.
                    A(:,[p,q]) = A(:,[p,q]) * J;

                    % Update V from the right.
                    V(:,[p,q]) = V(:,[p,q]) * J;
                end
            end
        end
    end
    display(["Number of sweeps in ", class(A), ": ",num2str(nsweep)]);

    % A now holds U * S. Extract singular values S.
    S = zeros(n,1);

    % TODO: guard against zero singular values
    for j = 1:n
        S(j) = norm(A(:,j));
        U(:,j) = A(:,j) / S(j);
    end
end


function [U, S, V] = svdJacobi2Sided(A, U, V, tol)
    % Extract dimensions.
    [n,~] = size(A);

    nsweep = -1;
    maxIter = 30;
    done = false;

    while ~done && nsweep < maxIter
        nsweep = nsweep + 1;
        done = true;

        % Row cyclic Jacobi
        for p = 1:n
            for q = p+1:n
                if abs(A(p,q)) > tol*sqrt(abs(A(p,p)))*sqrt(abs(A(q,q))) || ...
                   abs(A(q,p)) > tol*sqrt(abs(A(p,p)))*sqrt(abs(A(q,q)))
                    done = false;

                    % Compute SVD of 2x2 matrix.
                    [Jl, ~, Jr] = svd([A(p,p), A(p,q);
                                       A(q,p), A(q,q) ]);
                    
                    % Apply rotations to A annihilating (p,q), (q,p).
                    A([p,q],:) = Jl' * A([p,q],:);
                    A(:,[p,q]) = A(:,[p,q]) * Jr;
                    A(p,q) = 0.0;
                    A(q,p) = 0.0;

                    % Update V from the right.
                    V(:,[p,q]) = V(:,[p,q]) * Jr;
                    
                    % Update U from the right.
                    U(:,[p,q]) = U(:,[p,q]) * Jl;
                end
            end
        end
    end
    S = diag(A);
end

function Q = newtonSchulz(Q)
% NEWTONSCHULZ   Reorthogonalize the matrix Q.
%    Q = newtonSchulz(Q) orthogonalizes a matrix Q that
%    satisfies norm(Q'*Q-eye(n)) < 1e-5.
    [n, ~] = size(Q);
    for iter = 1:3
        Q = 0.5 * Q * (3*eye(n) - Q' * Q);
    end
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
