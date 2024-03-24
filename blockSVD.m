function [U, S, V] = blockSVD(A, algorithm)
%BLOCKSVD Compute the singular value decomposition of A.
%    [U, S, V] = BLOCKSVD(A) computes a singular value decomposition
%    A = U*S*V'. Supported algorithmic choices are:
%    -- [U, S, V] = BLOCKSVD(A, 'bidiag') bidiagonal QR iterations (default)
%    -- [U, S, V] = BLOCKSVD(A, 'polar') polar decomposition followed
%       by a symmetric eigenvalue problem solver.
%

    if nargin < 2
        algorithm = 'bidiag';
    end
 
    if ~strcmp(algorithm, 'bidiag') && ~strcmp(algorithm, 'polar')
        error("Supported 'algorithm' options: 'bidiag', 'polar'");
    end

    if strcmp(algorithm, 'polar')
        % Factor U * H = A.
        [U, H] = polar(A);

        % Factor V' * H * V = S.
        [V, S] = eig(H);

        % Accumulate unitary matrix.
        U = U * V;
    else if strcmp(algorithm, 'bidiag')
        % Golub-Reinsch QR algorithm (GESVD)

        % Compute bidiagonal decomposition Q' * A * P = B.
        [B, Q, P] = blockReduceToBidiagonal(A);

        % Bidiagonal QR iterations.
        [U, S, V] = bidiagSVD(B);

        % Accumulate unitary matrices.
        U = Q * U;
        V = P * V;
        % norm((Q*U)*S*(V'*P')-A)
    end
end
