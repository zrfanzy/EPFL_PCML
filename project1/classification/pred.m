% ridgeRegression
load('Shanghai_classification.mat');

sigmoid = @(x) exp(x)./(1+exp(x));

classlabel = X_test(:,35) > -10;

tX = normalizeFeature(X_test);
X = mypoly(tX,3);
tXX = [ones(length(X_test),1) X];
class predY;
predY = zeros(length(X_test),1);
for i = 1 : length(X_test)
    if classlabel(i) == 1
        beta = classbeta1;
    else
        beta = classbeta2;
    end
    %sigmoid(tXX(i,:) * beta)
    predY(i) = sigmoid(tXX(i,:) * beta);
end
