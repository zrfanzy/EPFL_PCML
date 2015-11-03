function cost = computeCost_classi(y,tX,beta)
% computer the cost by 0/1 loss
sigmoid = @(x) exp(x)./(1+exp(x));
e = y - sigmoid(tX * beta);
mse = 1/(2*length(y)) * (e)' * e;

ty = sigmoid(tX * beta)>0.5;

cost = sum(ty~=y);
% fprintf('computeCost: for %d data, %d wrong %.3f \n',size(y,1),cost,cost/size(y,1));
end