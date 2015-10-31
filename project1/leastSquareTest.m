% load data
clear all;
load('Shanghai_regression.mat');

X = normalizeFeature(X_train);

N = length(y_train);
tX = [ones(N, 1) X];
beta = leastSquares(y_train, tX);

testX = normalizeFeature(X_test);

M = length(testX);
tX = [ones(M, 1) testX];
predY = tX * beta;

csvwrite('predictions_regressions.csv', predY);