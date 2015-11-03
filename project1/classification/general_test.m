clear all
% ridgeRegression
load('Shanghai_classification.mat');

sigmoid = @(x) exp(x)./(1+exp(x));
y = normalization_claasi(y_train);

classlabel = X_train(:,35) > -10;

classes1 = find(classlabel==1);
classes2 = find(classlabel==0);

tX = normalizeFeature(X_train);
X = mypoly(tX,3);
alpha = 0.05;

for class = 1 : 2
    
if class == 1
    yTr = y(classes1);
    XTr = X(classes1,:);
else
    yTr = y(classes2);
    XTr = X(classes2,:);
end

tXTr = [ones(length(yTr), 1) XTr];

lambda = 0.0001;

beta = penLogisticRegression(yTr,tXTr,alpha,lambda);

if class == 1
    classbeta1 = beta;
else
    classbeta2 = beta;
end
msegenTr(class) = computeCost_classi(yTr, tXTr, beta)/length(yTr);

end
