function [P, H] = blockReduceToBlockHess(A, b)
%BLOCKREDUCETOBLOCKHESS   Reduction to block Hessenberg form.
%    [P, H] = BLOCKREDUCETOBLOCKHESS(A) produces an n-by-n unitary P and
%    an n-by-n block Hessenberg matrix
%    H = [ H11 H12 ...  ...   H1n ]
%        [ H21 H22 ...  ...   H2n ]
%        [  0                 ... ]
%        [ ...                ... ]
%        [  0   0  ...  Hnn-1 Hnn ]
%    with lower bandwidth b and upper triangular subdiagonal blocks so that
%    A = P*H*P'.

    if b < 1
        error('b has to be at least 1');
    end
    [n, ~] = size(A);
    P = eye(n);

    % Loop over block columns.
    for k = 1:b:n-b
        % Compute width of k-th block.
        r = min(n-k+1, b);

        % Partition
        % A = [ A11 A12 A13 ]    k
        %     [ A21 A22 A23 ]    r
        %     [  0  A32 A33 ]  n-k-r
        %        k   r  n-k-r

        % Compute [Q, R] = qr(A32).
        [Q, A(k+r:n, k:k+r-1)] = qr(A(k+r:n, k:k+r-1));

        %                 [ A11, A12,  A13 * Q       ]
        % Qj^H * A * Qj = [ A21, A22,  A23 * Q       ] where Qj := diag(I, Q)
        %                 [   0,   R,  Q^H * A33 * Q ]
        % [ A13 ] := [ A13 ] * Q
        % [ A23 ]    [ A23 ]
        A(1:k+r-1, k+r:n) = A(1:k+r-1, k+r:n) * Q;

        % A33 := Q^H * A33 * Q.
        A(k+r:n,k+r:n) = Q' * A(k+r:n,k+r:n) * Q;

        % Accumulate block reflectors.
        P = P * [eye(k+r-1),             zeros(k+r-1, n-k-r+1);
                 zeros(n-k-r+1, k+r-1),  Q                     ];
    end
    H = A;

    % Reduce block Hessenberg matrix to upper Hessenberg form.
    [Q, H] = mHessToHess(H);
    P = P * Q;
end

function [P, H] = mHessToHess(H)
    [n, ~] = size(H);
    P = eye(n);

    % Extract the bandwidth.
    bw = nnz(H(:, 1)) - 1

    % Introduce bulge
    for col = 1:1 %n-2
        % Generate elementary reflector to annihilate H(col+2:n,col)
        % and introduce bulge.
        r = -sign(H(col+1,col)) * norm(H(col+1:col+bw, col));
        [v, tau] = gallery('house', H(col+1:col+bw, col));
        H(col+1:col+bw, col) = [r, zeros(1,bw-1)];

        % Left update.
        H(col+1:col+bw, col+1:end) = ...
            H(col+1:col+bw, col+1:end) - tau * v * (v' * H(col+1:col+bw, col+1:end))

        % Right update.
        H(1:end, col+1:col+bw) = H(1:end, col+1:col+bw) - tau * (H(1:end, col+1:col+bw)*v)*v';

        % Accumulate reflectors.
        P(:, col+1:col+bw) = P(:, col+1:col+bw) - tau * (P(:, col+1:col+bw) * v) * v';

        % TODO: Chase off the bulge
        % TODO: Introduce more than one bulge
    end

end
