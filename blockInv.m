function X = blockInv(A)
%BLOCKINV Computes the inverse of an upper triangular matrix A.

% This is the left-looking variant.

    % Extract dimension.
    [n,~] = size(A);

    blksz = 80;
    for j = 1:blksz:n
        % Compute block column width.
        b = min(n-j+1, blksz);

        % Partition [A11 A12]
        %           [    A22], where A11 holds the block inverse inv(A11).
        % Compute current block colum [A12]
        %                             [A22].
        if j > 1
            % Compute upper triangular block
            % A12 := (-inv(A11)*A12)*inv(A22), where A11 holds inv(A11).

            % A12 := -A11 * A12.
            A(1:j-1,j:j+b-1) = -A(1:j-1,1:j-1) * A(1:j-1,j:j+b-1);

            % A12 := A12 * inv(A22).
            A(1:j-1,j:j+b-1) = A(1:j-1,j:j+b-1) / A(j:j+b-1,j:j+b-1);
        end

        % Use built-in to compute inverse of small block A22.
        A(j:j+b-1,j:j+b-1) = inv(A(j:j+b-1,j:j+b-1));
    end

    X = A;
end
