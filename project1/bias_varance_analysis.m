% for regulazation analysis, typically lambda
% ridgeRegression or penLogisticRegression
% plot the figure in bias-variance decomposition

% change the value in setSeed can generate 
% independent set of data split
clear all
% ridgeRegression
load('Shanghai_regression.mat');

% penLogisticRegression
% load('Shanghai_classificaion.mat');

for s = 1:50 % # of seeds
    setSeed(s);
    
    % randomly permute training data
    N = length(X_train);
    idx = randperm(N);
    y = y_train(idx);
    X = X_train(idx,:);

    % split data, portion of data for traning
    portion = 0.8;
    [XTr, yTr, XTe, yTe] = split(y,X,portion);
    
    % size of the training set and validation set
    size_training = length(yTr); 
    size_test = length(yTe);
    
    % for different lambda
    lambda = logspace(-2, 8,50);
    
    for k = 1:length(lambda)
        % expansion
    	tXTr = [ones(size_training, 1) XTr];
    	tXTe = [ones(size_test, 1) XTe];      
 
%% for ridge regression
        beta = ridgeRegression(y,tXTr, lambda(k));
        
%% for penLogisticRegression
        % set stepsize alpha
%         alpha = 0.003;
%         beta = penLogisticRegression(y,tX,alpha,lambda);

        % compute train and test RMSE
        rmseTr(s,k) = computeCost(yTr,tXTr,beta);
        rmseTe(s,k) = computeCost(yTe,tXTe,beta);
    end

    end
    
% compute expected train and test error
rmseTr_mean = mean(rmseTr);
rmseTe_mean = mean(rmseTe);
% rmseTe_mean_re = reshape(rmseTe_mean,[i,k]);
% rmseTr_mean_re = reshape(rmseTr_mean,[i,k]);

subplot(2,1,1);

% rmseTe_resize = reshape(rmseTe,[s * i,k]);
% rmseTr_resize = reshape(rmseTr,[s * i,k]);

plot(lambda, rmseTe,'r-', 'color', [1 0.7 0.7]);
hold on;
plot(lambda, rmseTr, 'b-', 'color', [0.7 0.7 1]);
hold on;
legend('test','train');

plot(lambda, rmseTe_mean, 'r-', 'linewidth', 3);
hold on;
plot(lambda, rmseTr_mean,'b-', 'linewidth',3);
ylabel('error');

subplot(2,1,2);
boxplot(rmseTe,'boxstyle','filled');
ylim('auto');
