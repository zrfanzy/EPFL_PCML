function beta = ridgeRegression(y, tX, lambda)
beta = pinv(tX' * tX + lambda * eye(length(tX' * tX))) * tX' * y;

