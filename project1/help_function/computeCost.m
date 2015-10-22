function cost = computeCost(y,tX,beta)
% computer the cost by MSE
e = y - tX * beta;
mse = 1/(2*length(y)) * (e)' * e;
cost = sqrt(2 * mse);
end