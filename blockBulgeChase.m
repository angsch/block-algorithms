function [H, Q] = blockBulgeChase(H, nb)
% BLOCKBULGECHASE  Chase tightly packed bulges.
%    [A, Q] = BLOCKBULGECHASE(H, nb) chases the nb 3-by-3 bulges tightly packed
%    in the top left corner of the upper Hessenberg matrix H to the bottom
%    right corner. The orthogonal matrix Q accumulates all similarity
%    transformations applied during the chase and A = Q*H*Q'.

    [m, n] = size(H);

    if m ~= n
        error('H must be square');
    end

    % Expect chain spanning at most half the diagonal.
    if nb * 6 > n || nb < 1
        error('Set bulge count to any number between 1 and n / 6.');
    end

    % Window size divisible by 6.
    ws = 120;

    if ws > n
        % Unblocked bulge chase.
        [H, Q] = chase(H, nb);
    else
        % Use small bulge chase routine in window and matrix-matrix
        % multiplications for off-diagonal updates.

        Q = eye(n);

        % Last bulge in the chain of tightly packed bulges yet to be moved.
        bottommost_bulge = nb;

        % First column of the bottommost bulge.
        pos_bulge = bottommost_bulge * 3 - 2;
        final_pos = n;

        % Center window around bottommost bulge.
        first = max(1, pos_bulge - ws / 2);
        last = min(first + ws, final_pos) + 2;

        % Number of bulges in the window.
        nb_window = (pos_bulge + 3 - first) / 3;

        nMoves = 1;
        while bottommost_bulge > 0
            while last > first && nMoves > 0
                % Chase bulges in window.
                [H(first:last, first:last), Qj, nMoves] = ...
                    chase(H(first:last, first:last), nb_window); 

                % Left update.
                H(first:last, last+1:end) = Qj' * H(first:last, last+1:end);

                % Right update.
                H(1:first-1, first:last) = H(1:first-1, first:last) * Qj;

                % Window size.
                w = last - first + 1;

                % Accumulate reflectors.
                Q = Q * [eye(first-1),             zeros(first-1, n-first+1);
                         zeros(w, first-1),        Qj, zeros(w, n-last);
                         zeros(n-last, first-1+w), eye(n-last, n-last)       ];

                % Move window down.
                first = first + nMoves;
                last = min(last + nMoves, final_pos);
            end

            % Prepare next window.
            bottommost_bulge = max(0, bottommost_bulge - nb_window);
            pos_bulge = bottommost_bulge * 3 - 2;
            final_pos = final_pos - 3 * nb_window;
            first = max(1, pos_bulge - ws / 2);
            last = min(first + ws, final_pos) + 2;
            nb_window = (pos_bulge + 3 - first) / 3;
            nMoves = 1;
        end
    end
end

function [H, Q, nMoves] = chase(H, nb)
    [n, ~] = size(H);
    Q = eye(n);

    % Number of moves along the diagonal (same for all bulges).
    nMoves = n - 4 - (nb - 1) * 3;

    for i = 1:nMoves
        % Move bulges one step down along the diagonal, starting from the bottom.
        for bulge = nb:-1:1
            % Index of the first column of a bulge.
            k = 1 + (bulge - 1) * 3 + i - 1;

            % Reduce the first column of the bulge.
            [v, tau, r] = gallery('house', H(k+1:k+3, k));
            H(k+1:k+3, k) = [r; 0; 0];

            % Left update.
            H(k+1:k+3, k+1:end) = ...
                H(k+1:k+3, k+1:end) - tau * v * (v' * H(k+1:k+3, k+1:end));

            % Right update.
            H(1:k+4, k+1:k+3) = ...
                H(1:k+4, k+1:k+3) - tau * (H(1:k+4, k+1:k+3) * v) * v';

            % Accumulate reflectors.
            Q(:, k+1:k+3) = Q(:, k+1:k+3) - tau * (Q(:, k+1:k+3) * v) * v';
        end
    end
end
