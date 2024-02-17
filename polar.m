function [U, H] = polar(A, algorithm)
%POLAR Computes a Polar decomposition of A.
%    [U,H] = POLAR(A) factors a full-rank matrix A into a unitary U
%    and a Hermitian positive definite matrix H such that A = U*H.

    if nargin < 2
        algorithm = 'Newton';
    end

    if strcmp(algorithm,'Newton')
        [U, H] = polarNewton(A);
    elseif strcmp(algorithm,'QDWH')
        [U, H] = QDWH(A);
    else
        error("Supported 'algorithm' options: 'Newton', 'QDWH'");
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

function [U, H] = QDWH(A)
% QDWH Polar decomposition by QR-based dynamically weighted Halley iteration
%    [U,H] = QDWH(A) factors the square full-rank matrix A
%    into a unitary U and a Hermitian positive definite matrix
%    H such that A = U*H.

    % Section4, QDWH algorithm as presented in
    % https://web.cs.ucdavis.edu/~bai/Winter09/nakatsukasabaigygi09.pdf
    % X_{k+1} = X_k (a_k I + b_k X_k^H * X_k)(I + c_k X_k^HX_k)^-1, X_0 = A/alpha

    % Extract dimension.
    [m, n] = size(A);

    % Transform the problem into one for square matrices.
    [Q1, R] = qr(A);

    % Convergence threshold
    tol = nthroot(4 * eps,3);

    % Estimate largest singular value.
    alpha = norm(A, 'fro');
    X = R / alpha;

    % Estimate reciprocal condition number.
    l = 1/condest(R);

    maxIter = 8;
    iter = 1;
    while iter < maxIter
        Xold = X;

        % Compute weights dynamically.
        d = nthroot(4 * (1-l^2)/l^4, 3);
        a = sqrt(1+d) + 0.5*sqrt(8-4*d+8*(2-l^2)/(l^2 * sqrt(1+d)));
        b = (a-1)^2 / 4;
        c = a+b-1;

        % (a,b,c) should converge to Halley's iteration (3,1,3).
        assert(l <= 1);
        assert(3 <= a && a <= (2+l)/l);
        assert(b >= 1);
        assert(c >= a);

        % Compute QR factorization.
        [Q, R] = qr([ sqrt(c)*Xold;
                      eye(n,n)     ]);

        % Update X.
        X = (b/c) * Xold + (1/sqrt(c))*(a-b/c)*Q(1:n,1:n)*Q(n+1:2*n,1:n)';
        l = l * (a+b*l^2)/(1+c*l^2);

        % Convergence test
        if norm(Xold - X, 'fro') <= tol
            U = Q1*X;
            H = U' * A;
            disp(['  Iterations until convergence reached: ', num2str(iter)]);
            return;
        end
        iter = iter + 1;
    end
end
