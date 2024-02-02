function [U, H] = polar(A, algorithm)
%POLAR Computes a Polar decomposition of A.
%    [U,H] = POLAR(A) factors a full-rank matrix A into a unitary U
%    and a Hermitian positive definite matrix H such that A = U*H.

    if nargin < 2
        algorithm = 'Newton';
    end

    if strcmp(algorithm,'Newton')
        [U, H] = polarNewton(A);
    else
        error("Supported 'algorithm' options: 'Newton'");
    end
end

function [U, H] = polarNewton(A)
%POLARNEWTON Polar decomposition by Newton's iteration.
%    [U,H] = POLARNEWTON(A) factors the square full-rank matrix A
%    into a unitary U and a Hermitian positive definite matrix
%    H such that A = U*H.

    % Newton's iteration with BX scaling. See Section 2.1 in
    % https://web.cs.ucdavis.edu/~bai/Winter09/nakatsukasabaigygi09.pdf
    X = A;
    alpha = norm(A, 'fro');
    beta = 1.0 / norm(inv(A), 'fro');
    scale = 1.0 / sqrt(alpha * beta);

    % Loop control
    maxIter = 30;
    iter = 1;

    % Convergence threshold
    tol = nthroot(4 * eps,3);

    while iter < maxIter
        Xold = X;

        Xinv = inv(X);
        X = 0.5 * (scale*X + (1.0/scale)*Xinv');
        if iter == 1
            scale = sqrt(2.0 * sqrt(alpha * beta) / (alpha + beta));
        else
            scale = sqrt(2.0 / (scale+1/scale));
        end
        
        % Convergence test
        if norm(Xold - X, 'fro') <= tol
            U = X;
            H = U' * A;
            return;
        end
        iter = iter + 1;
    end

    error('Not converged, condition number = %.2e', cond(A));
end

