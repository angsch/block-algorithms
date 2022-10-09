function X = blockShiftedHessenbergSolve(H, Mu, B)
% BLOCKSHIFTEDHESSENBERGSOLVE  Solves a shifted Hessenberg system.
%    X = BLOCKSHIFTEDHESSENBERGSOLVE(H, Mu, B) returns the m-by-n solution X to
%    H * X - X * diag(Mu) = B, where H is an upper Hessenberg matrix,
%    Mu is an n-vector of shifts, and B in an m-by-n matrix. All data is real.
%
%    This protype demonstrates the core ideas presented in
%    https://doi.acm.org?doi=3544789 (available as preprint: arXiv:2101.05063)
%    for a solving a series of shifted Hessenberg systems such that the
%    majority of the flops is matrix-matrix multiplications.

    [m, n] = size(H);
    if m ~= n
        error('Hessenberg matrix must be square');
    end

    [m, num_rhs] = size(B);
    if m ~= n
      error('Number of rows in B and number of column in H are not conforming');
    end

    if num_rhs ~= length(Mu)
        error('Number of shifts and number of right-hand sides mismatch');
    end

    % Compute tiling.
    blksz = min(140, n/2);
    num_blks = floor((n + blksz - 1) / blksz);
    partitioning = zeros(num_blks+1,1);
    for i = num_blks-1:-1:0
        % Compute indices embracing diagonal block (i,i) as (l:r,l:r).
        l = ceil(i * blksz+1);
        r = ceil(min((i+1)*blksz, n));
        partitioning(i+1) = min(l, n);
    end
    partitioning(num_blks+1) = n+1;

    % Allocate workspace.
    X = zeros(n, num_rhs);
    Z = zeros(n,num_rhs);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Reduction phase, yielding Givens rotations and cross-over columns s.t.
    % (H - mu * I) * Q^T = R, where Q^T is a product of Givens rotations.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [c, s, Rtildes] = reduce(H, Mu, partitioning);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Backward substitution phase. On exit, b holds the solution to R * y = b.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = num_blks:-1:1
        % Compute indices embracing diagonal block (i,i) as (l:r,l:r).
        l = partitioning(i);
        r = partitioning(i+1)-1;

        % Solve a small shifted Hessenberg system.
        for rhs = 1:num_rhs
            mu = Mu(rhs);
            if i == 1
                % Pad with zero column.
                Hii = [ zeros(r-l+1,1), H(l:r,l:r-1), Rtildes(l:r,rhs,i) ];
            else
                Hii = [ H(l:r,l-1:r-1), Rtildes(l:r,rhs,i) ];
            end
            for j = 1:r-l
                Hii(j,j+1) = Hii(j,j+1) - mu;
            end
            Rii = generateR(Hii, c(l:r,rhs), s(l:r,rhs));
            B(l:r,rhs) = Rii\B(l:r,rhs);
        end

        % Update without using shift-dependent R.
        if l > 1
            for rhs = 1:num_rhs
                rho = s(l,rhs) * B(l,rhs);
                B(1:l-1,rhs) = B(1:l-1,rhs) + rho * H(1:l-1,l-1);
                B(l-1,rhs) = B(l-1,rhs) - Mu(rhs) * rho;
            end
            for rhs = 1:num_rhs
                z = [zeros(l-1,1);
                     B(l:r,rhs);
                     zeros(n-r,1) ];
                for j = max(2,l):r
                    G = [ c(j,rhs) -s(j,rhs);
                          s(j,rhs) c(j,rhs) ];
                    z(j-1:j) = G * z(j-1:j);
                end
                Z(:,rhs) = z;
            end
            B(1:l-1,:) = B(1:l-1,:) - H(1:l-1,l:r-1) * Z(l:r-1,:); % DGEMM
            for rhs = 1:num_rhs
                B(1:l-1,rhs) = B(1:l-1,rhs) - Rtildes(1:l-1,rhs,i) * Z(r,rhs);
            end
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Backtransform x = Q^T * b.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for rhs = 1:num_rhs
        tau1 = B(1,rhs);
        for i = 2:n
            tau2 = B(i,rhs);
            X(i-1,rhs) = c(i,rhs) * tau1 - s(i,rhs) * tau2;
            tau1 = c(i,rhs) * tau2 + s(i,rhs) * tau1;
        end
        X(n,rhs) = tau1;
    end
end

function R = generateR(H, c, s)
% Reduce a shifted Hessenberg matrix with cross-over column to triangular form.
%     1    n                          1    n
% 1  HSHHHHr                         0RRRRRR
%     HSHHHr                           RRRRR
%      HSHHr                            RRRR
%       HSHr  * G_{n} * ... * G_1 =>     RRR
%        HSr                              RR
% n       Hr                               R
    [n, ~] = size(H);

    R = zeros(n,n);
    R(:,n) = H(:,end);

    for k=n:-1:2
        % Restore Givens rotation.
        G = [ c(k) -s(k);
              s(k)  c(k) ];

        % Apply Givens rotation from the right.
        R(1:k,k-1:k) = [ H(1:k,k), R(1:k,k) ] * G;
    end

    % Apply final rotation to R(1,1).
    R(1,1) = [H(1,1) R(1,1)] * [ -s(1);
                                  c(1) ];
end

function [c, s, Rtildes] = reduce(H, Mu, partitioning)
% Compute a sequence of Givens rotations that reduces (H-mu*I) to a
% triangular matrix.
% R = (H-mu*I) * [ c(n) -s(n) ] * [ c(n-1) -s(n-1) ] * ... * [ c(2) -s(2) ]
%                [ s(n)  c(n) ]   [ s(n-1)  c(n-1) ]         [ s(2)  c(2) ]

    [n, ~] = size(H);
    num_rhs = length(Mu);
    num_blks = length(partitioning) - 1;

    % Allocate space to record the cosine and sine components.
    c = zeros(n,num_rhs);
    s = zeros(n,num_rhs);

    % Allocate space to record the cross-over columns.
    Rtildes = zeros(n, num_rhs, num_blks);

    % The rightmost cross-over column corresponds to H(:,n) - e_n * mu.
    cur_blk_col = num_blks;
    for rhs = 1:num_rhs
        Rtildes(:,rhs,cur_blk_col) = H(:,n);
        Rtildes(n,rhs,cur_blk_col) = Rtildes(n,rhs,cur_blk_col) - Mu(rhs);
    end

    % Quick access to Cartesian standard basis vectors.
    E = eye(n);

    % Running column.
    r = zeros(n,1);

    % Compute all cross-over columns.
    for rhs = 1:num_rhs
        cur_blk_col = num_blks;
        r = Rtildes(:,rhs,cur_blk_col);

        % Reduce H - mu * I to triangular form.
        for k=n:-1:2
            % Compute and record a Givens rotation [ c s
            %                                       -s c ].
            G = planerot([ r(k); H(k,k-1) ]);
            c(k,rhs) = G(1,1);
            s(k,rhs) = G(2,1);

            % Check if we cross a block column boundary.
            if k == partitioning(cur_blk_col) - 1
                % Record the cross-over column. Note that only G_{k+1}^T has
                % transformed r.
                Rtildes(1:k,rhs,cur_blk_col-1) = r(1:k);
                cur_blk_col = cur_blk_col - 1;
            end

            % Reduce H by applying a Givens rotation from the right.
            r(1:k) = [ H(1:k,k-1) - Mu(rhs) * E(1:k,k-1), r(1:k) ] * G(:,1);
        end

        % G_1^T is not defined. Pad c(1) and s(1) to match the identity matrix.
        c(1,rhs) = 1.;
        s(1,rhs) = 0.;
    end
end
