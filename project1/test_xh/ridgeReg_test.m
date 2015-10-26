% for regulazation analysis, typically lambda
% ridgeRegression or penLogisticRegression
% plot the figure in bias-variance decomposition

% change the value in setSeed can generate 
% independent set of data split
clear all
% ridgeRegression
load('Shanghai_regression.mat');
% X = normalizeFeature(X_train);
 X = X_train;
 binary_ft = [18,19,20,35,47];
 categori_ft = [13,16,17,22,36,49,52,54];
 non_norm = [13,16,17,22,36,18,19,20,35,47,49,52,54];


y = normalizeFeature(y_train);

% X = X.^5;
% for i=1:71
%      if(~isempty(find(non_norm == i)))
%          X(:,i) = normalizeFeature(X_train(:,i));
%      else
%          X(:,i) = normalizeFeature(X(:,i));
%      end
% end

seeds = 1;
for s = 1 : length(seeds)
setSeed(seeds(s));

% test on degree: manipulate the features
degree =  1:9;
for i_degree  = 1:length(degree)
    X_expand = zeros(size(y_train,1),1);
for i=1:71
     if(~isempty(find(categori_ft == i)))
%            append_cols = normalizeFeature(X_train(:,i));
         append_cols = dummy_encoding(X_train(:,i)');
          X_expand = [X_expand append_cols];
          
     elseif(~isempty(find(binary_ft == i))) 
         append_cols = X_train(:,i);
         X_expand = [X_expand append_cols];
     elseif(i==48)
         append_cols = split_feature48(X_train(:,i));
         display(size(X_expand,2));
         X_expand = [X_expand append_cols];
     elseif(i==12)
         append_cols = split_feature12(X_train(:,i));
         append_cols = normalizeFeature(append_cols);
         X_expand = [X_expand append_cols];
      elseif(i==57 || i ==65 ||i ==21||i==27||i==69||i==34||i==51||i==5||i==45)
          continue;
     else 
%        append_cols = remove_outlier(X_train(:,i));
         append_cols = X_train(:,i);
         append_cols = mypoly(append_cols,degree(i_degree));
         append_cols = normalizeFeature(append_cols);


         X_expand = [X_expand append_cols];
          

     end
end
X = X_expand(:,2:size(X_expand,2));
fprintf('featue input %d',size(X,2));

% split data in K fold (we will only create indices)

prop = 0.7;
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
lambda = logspace(-2,4,10);
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
            
            beta = ridgeRegression(yTr_cv,tXTr_cv,lambda(i));

            mseTrSub(k) = computeCost(yTr_cv, tXTr_cv, beta); 
            mseTeSub(k) = computeCost(yTe_cv, tXTe_cv, beta);
        end
            % compute the mean error for k cross validation of the same lambda
            fprintf('dg %d mean%dfold varErrTe: %.4f for lambda %.2f  ',degree(i_degree),k,var(mseTeSub),lambda(i));
            
            rmseTr_lamb(i) = mean(mseTrSub);
            rmseTe_lamb(i) = mean(mseTeSub);
            fprintf('tr %.4f te %.4f\n ',rmseTr_lamb(i),rmseTe_lamb(i));
    %         box(:,i) = mseTeSub;
    end % end of runing for different lambda
    
[numb,index_best_lambda] = min(rmseTe_lamb);
% by cv find lambda, use the lambda to train and test, get the performance
% of the degree(diff model complexity)
beta = ridgeRegression(yTr,tXTr,lambda(index_best_lambda));
mseTr_degree(s,i_degree) = computeCost(yTr, tXTr, beta);
mseTe_degree(s,i_degree) = computeCost(yTe, tXTe, beta);

fprintf('\ndegree %d ; lambda %f;  test: %.4f; train:%.4f \n',...
    degree(i_degree),lambda(index_best_lambda),...
    mseTe_degree(s,i_degree),mseTr_degree(s,i_degree));

end

end% for different seed, repeated again

for s = 1:length(seeds)
plot(degree,mseTr_degree(s,:),'b');hold on;
plot(degree,mseTe_degree(s,:),'r');hold on;grid on;
end

% fprintf('for different seed, the mean test error is %.4f \n',mean(mesTest));
