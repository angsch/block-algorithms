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
    elseif strcmp(algorithm,'Zolo')
        [U, H] = zolo(A);
    else
        error("Supported 'algorithm' options: 'Newton', 'QDWH', 'Zolo'");
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

    % Estimate sigma_max(A) = norm(A,2) <= norm(A, 'fro')
    alpha = normest(A); % approximates 2-norm
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

function [U, H] = zolo(A)
% ZOLO Polar decomposition by Zolotarev's functions.
%    [U,H] = zolo(A) factors the m-by-n full-rank matrix A
%    into a unitary U and a Hermitian positive definite matrix
%    H such that A = U*H.
%
%    This is an improved version of Algorithm 1 in
%    https://people.maths.ox.ac.uk/nakatsukasa/publishedpdf/zoloeigsvd.pdf

    % Number of factors to use, chosen such that two iterations suffice.
    r = 8;

    [~,m] = size(A);

    % Transform the problem into one for square matrices.
    [Q0, R] = qr(A);

    % Estimate largest singular value.
    alpha = norm(A, 'fro');
    X = R/alpha;

    %disp(['  cond(X) at the beginning = ', num2str(cond(X))]);

    % Estimate reciprocal condition number.
    l = 1/condest(R);

    lprime = sqrt(1-l^2);
    assert(real(lprime^2) <= 1 || iscomplex(lprime^2));
    Kprime = ellipke(lprime^2/(lprime^2-1)) / sqrt(1-lprime^2);
    % Matlab's ellipke only works for input in the range [0,1]
    % Use ellipticK from the symbolic toolbox instead
    % Kprime = ellipticK(lprime^2/(lprime^2-1)) / sqrt(1-lprime^2)

    c = zeros(2*r,1);
    for j = 1:2*r
        % Note second input parameter should be lprime^2 as Matlab defines
        % the Jacobi elliptic functions using parameter M=lprime^2
        [sn, cn] = ellipj(j*Kprime/(2*r+1), lprime^2);
        c(j) = l^2 * (sn^2/cn^2);
    end

    % Note bug in paper, which says
    %            a(j) = -prod(some c) * prod(some c);
    % should be  a(j) = -prod(some c) / prod(some c);
    a = zeros(r,1);
    for j = 1:r
        a(j) = -prod(c(2*j-1) - c(2:2:2*r) ) / prod(c(2*j-1) - c(setdiff(1:2:2*r-1,2*j-1)) );
    end


    Mhat = prod( (1 + c(1:2:2*r-1)) ./ (1 + c(2:2:2*r) ) );

    Q = cell(r,1); % Use a cell array in case we want to go back and check...
    Xupdate = zeros(size(X));
    for j = 1:r

        [Q{j},~] = qr([X; sqrt(c(2*j-1)) * eye(m)]);

        % Mimic economy mode
        Q1 = Q{j}(1:m,1:m);
        Q2 = Q{j}(m+1:end,1:m);

        Xupdate = Xupdate + a(j)/sqrt(c(2*j-1)) * Q1 * Q2';
    end
    X = Mhat*(X+Xupdate);

    % Step 1 is complete. The conditioning has improved so that
    % we can use Cholesky-based iterations.

    %disp(['  cond(X) after one iteration = ', num2str(cond(X))]);

    l = Mhat * l * prod( (l^2 + c(2:2:2*r)) ./ (l^2 + c(1:2:2*r-1)) );
    lprime = sqrt(1-l^2);
    Kprime = ellipke(lprime^2/(lprime^2-1)) / sqrt(1-lprime^2);
    % Matlab's ellipke only works for input in the range [0,1]
    % Use ellipticK from the symbolic toolbox
    % Kprime = double(ellipticK(sym(lprime^2/(lprime^2-1))) / sqrt(1-lprime^2))

    c = zeros(2*r,1);
    for j = 1:2*r
        % Note second input parameter should be lprime^2 as Matlab defines
        % the Jacobi elliptic functions using parameter M=lprime^2
        [sn, cn] = ellipj(j*Kprime/(2*r+1),lprime^2);
        c(j) = l^2 * (sn^2/cn^2);
    end

    a = zeros(r,1);
    for j = 1:r
        a(j) = -prod(c(2*j-1) - c(2:2:2*r) ) / ...
            prod(c(2*j-1) - c(setdiff(1:2:2*r-1,2*j-1)) );
    end
    Mhat = prod( (1 + c(1:2:2*r-1)) ./ (1 + c(2:2:2*r) ) );

    R = cell(r,1);
    Xupdate = zeros(size(X));
    for j = 1:r
        Z = X'*X+c(2*j-1)*eye(m);
        R{j} = chol(Z);

        Xupdate = Xupdate + a(j) * (X * inv(R{j})) * inv(R{j})'; 
    end
    % Note typo in paper, says X_2 = Mhat*(X_2 + update)
    %                should be X_2 = Mhat*(X_1 + update)
    X = Mhat*(X+Xupdate);

    % Step 2 is complete. In double precision and an initial condition
    % number of less than 1e+16, the algorithm has converged and
    % we can compute the Hermitian factor.
    U = Q0 * X;
    H = (U'*A+(U'*A)')/2;
end
