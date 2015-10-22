function beta = ridgeRegression(y, tX, lambda)
lambdaI = lambda * eye(length(tX' * tX));
lambdaI(1,1) = 0;
beta = pinv(tX' * tX + lambdaI) * tX' * y;
end
