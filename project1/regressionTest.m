% load data
clear all;
load('Shanghai_regression.mat');

X = normalizeFeature(X_train);

%meanX = mean(X);
%stdX = std(X);
%for i = 1:length(meanX)
%    X(:,i) = X(:,i) - meanX(i);
%    X(:,i) = X(:,i) ./ stdX(i);
%end

N = length(y_train);
tX = [ones(N, 1) X];
beta = leastSquaresGD(y_train, tX, 0.1)
%beta = leastSquares(y_train, tX)