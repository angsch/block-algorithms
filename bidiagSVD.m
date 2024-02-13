function [U, S, V] = bidiagSVD(B)
%BIDIAGSVD   Singular value decomposition of upper bidiagonal matrix.
%    [U, S, V] = bidiagSVD(B) computes a singular value decomposition
%    B = U*S*V', where B is an upper bidiagonal matrix, U and V
%    are unitary matrices and S is the diagonal matrix with the
%    singular values of B.
%    This routine realizes the implicit non-zero shift bidiagonal
%    QR algorithm. It assumes that B is not close-to-rank deficient.

% This code is based on the bidiagonal SVD with shift by Eddie Wadbro
% under the CC-BY-SA-4.0 license. The following changes have been made:
% - Revision of the function interface
% - Replacement of the function rot() with the octave-builtin
%   function givens()
% - Change V generation and application.
% - Add postprocessing: convert negative singular values into
%   non-negative ones
% - Limit iteration count

    % Extract dimension.
    [m,n] = size(B)
    if m ~= n
        error("B must be square");
    end

    % Want to find U, Sigma, and V so that B = U*Sigma*V^T
    U = eye(n);
    V = eye(n);

    % Loop control
    done = false;
    maxIter = 30 * n;
    iter = 1;
    nmax = n;
    
    % Convergence threshold
    tol = 1e-12;
    
    while ~done && iter < maxIter
        iter = iter + 1;

        % Compute the SVD of the bottom 2-by-2 block.
        lls = svd(B(nmax-1:nmax,nmax-1:nmax));

        % The SVD returns the singular values in sorted order.
        % Pick minimum singular value.
        shift = lls(2);
        
        for i=1:nmax-1
            % Transformation from the right.
            if i > 1
                f = B(i-1,i);
                g = B(i-1,i+1);
            else %Note upper part of B^T B is [s1^2 s1e1 0 ...]
                f = B(1,1)^2 - shift^2;
                g = B(1,1)*B(1,2); 
            end
            [cs, sn] = givens(f,g);

            % Embed transposed Givens rotation and transform B.
            Q = eye(n);
            Q(i:i+1,i:i+1) = [ cs -sn;
                               sn  cs ];
            B = B*Q;

            % Accumulate the unitary transformations from the right.
            V = V*Q;

            % Transformation from the left.
            %Here: f = B(i,i), g = B(i+1,i), and h = B(i+1,i+1)
            f = B(i,i);
            g = B(i+1,i);        
            [cs, sn] = givens(f,g);

            % Embed Givens rotation and transform B.
            Q = eye(n);
            Q(i:i+1,i:i+1) = [ cs sn;
                              -sn cs ];
            B = Q*B;

            % Accumulate the unitary transformations from the left.
            U = U*Q';
            % Note that the matrix B([i i+1],[i+1 i+2]) has rank-1
            % with entries [h*sn e(i+1)*sn; h*cs e(i+1)*cs]
        end
        
        % Reset B to bitriangular form:
        B = triu(B)-triu(B,2);
        % Convergence test
        if abs(B(nmax-1,nmax)) < tol
            B(nmax-1,nmax) = 0.0;
            if nmax > 2
                nmax = nmax - 1;
            else
                done = true;
            end
        end
    end
    
    % Make all singular values non-negative
    for i=1:n
        if B(i,i) < 0
            B(i,i) = abs(B(i,i));
            V(:,i) = V(:,i)* (-1.0);
        end
    end

    S = B;
end

