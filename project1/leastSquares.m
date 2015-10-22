function beta = leastSquares(y, tX)
% using normal equation
beta = pinv(tX' * tX) * tX' * y;