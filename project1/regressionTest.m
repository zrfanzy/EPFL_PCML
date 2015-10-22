% load data
clear all;
load('Shanghai_regression.mat');

% normalize features
meanX = mean(X_train);
for i = 1:length(meanX)
    X_train(:,i) = X_train(:,i) - meanX(i);
end

N = length(y_train);
tX = [ones(N, 1) X_train];
beta = leastSquaresGD(y_train, tX, 0.1)
%beta = leastSquares(y_train, tX)