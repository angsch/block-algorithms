function count = eigCountCircle(A, x0, r)
%EIGCOUNTCIRCLE   Count number of eigenvalues in a circle.
%    count = EIGCOUNTCIRCLE(A), where A is a square matrix,
%    returns the number of eigenvalues in a circle.
%
%    Calling the routine with the optional "x0" and "r" arguments 
%    specifies the circle. Specifically, x0 is the center point
%    of the circle and r is the radius r.
%
%    This routine is based on Cauchy's argument principle.
%    If A does not have any zeros or poles on the contour C,
%       1/(2*pi*i) ∮_C (f'(z)/f(z))dz = Z + P,
%    where f(z) = det(A-z*I), and Z and P are the number of
%    zeros and poles of f inside C. Since f is the
%    characteristic polynomial, P = 0 and the number of zeros
%    corresponds to the number of eigenvalues in C.
%    This routine implements the mathematically equivalent
%    formulation
%      1/(2*pi*i) ∮_C trace(inv(A-z*I)) dz = Z.
%    

    if nargin < 2
       % Unit circle.
       x0 = complex(0,0);
       r = 1;
    end

    % Extract dimensions.
    [m, ~] = size(A);

    % Number of quadrature points.
    N = 100;

    % Generate quadrature points on the contour of the circle.
    t = linspace(0, 2*pi, N+1);
    z = x0 + r * exp(1i * t(1:N));

    % Evaluate the integrand on the N integration points.
    result = zeros(N, 1);
    for k = 1:N
        result(k) = -trace(inv(A - z(k) * eye(m)));
    end

    % Integrate using trapezoidal rule.
    sum = 0.0;
    for k = 1:N-1
        dz = z(k+1) - z(k);
        sum = sum + dz * (result(k+1) + result(k)) / 2.0;
    end
    % Close the contour.
    dz = z(N) - z(1);
    sum = sum + dz * (result(1) + result(N)) / 2.0;

    count = max(0, floor(imag(sum)/(2*pi) + 0.5));
end
