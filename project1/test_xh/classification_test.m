clear all
% ridgeRegression
load('Shanghai_classification.mat');
% X = normalizeFeature(X_train);
X = X_train;
% y = normalizeFeature(y_train);
y = normalization_claasi(y_train);
X = [X expandx(X)];

prop = 0.8;
[XTr, yTr, XTe, yTe] = split(y, X, prop);
tXTr = [ones(length(yTr), 1) XTr];
tXTe = [ones(length(yTe), 1) XTe]; 

K = 5;
N = size(yTr,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
	idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

% lambda values found for each degree
% lambda = logspace(-5,15 ,20);
lambda = logspace(-2,6,100);
lambda = [0,lambda];
    for i = 1:length(lambda)
    % K-fold cross validation for each lambda
        for k = 1:K
        % get k'th subgroup in test, others in train

            idxTe = idxCV(k,:);

            idxTr = idxCV([1:k-1 k+1:end],:);
            idxTr = idxTr(:);

            yTe_cv = yTr(idxTe);
            XTe_cv = XTr(idxTe,:);

            yTr_cv = yTr(idxTr);
            XTr_cv = XTr(idxTr,:);

            tXTr_cv = [ones(length(yTr_cv), 1) XTr_cv];

            tXTe_cv = [ones(length(yTe_cv), 1) XTe_cv]; 

            % ridge regression    
            alpha = 0.005;
            beta = penLogisticRegression(yTr_cv,tXTr_cv,alpha,lambda(i));

            mseTrSub(k) = computeCost(yTr_cv, tXTr_cv, beta); 
            mseTeSub(k) = computeCost(yTe_cv, tXTe_cv, beta);
        end
            % compute the mean error for k cross validation of the same lambda
            fprintf('%dfold varErrTe: %.4f for lambda %.2f  ',k,var(mseTeSub),lambda(i));
            
            rmseTr_lamb(i) = mean(mseTrSub);
            rmseTe_lamb(i) = mean(mseTeSub);
            fprintf('tr %.4f te %.4f\n ',rmseTr_lamb(i),rmseTe_lamb(i));
    %         box(:,i) = mseTeSub;
    end % end of runing for different lambda
    
[numb,index_best_lambda] = min(rmseTe_lamb);

beta = penLogisticRegression(yTr_cv,tXTr_cv,alpha,lambda(index_best_lambda));
mseTr = computeCost(yTr, tXTr, beta);
mseTe = computeCost(yTe, tXTe, beta);
fprintf('tr %.4f te %.4f and lambda%.3f \n',mseTr,mseTe,lambda(index_best_lambda));