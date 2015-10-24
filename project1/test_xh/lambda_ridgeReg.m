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
 non_norm = [13,16,17,18,19,20,22,35,36,47,49,52,54];
for i = 1 : 71

    
    if i==6
        temp = X_train(:,i);
        [a,index] = min(temp);
        temp(index) = 0;
        X(:,i) = temp;
    end
    if i==8
        temp = X_train(:,i);
        [a,index] = max(temp);
        temp(index) = 5;
%         second_extre= max(temp);
        
        X(:,i) = temp;
    end
    if i==15
        temp = X_train(:,i);
        [a,index] = min(temp);
        temp(index) = 2;
        X(:,i) = temp;
    end      
    
    if i==28
        temp = X_train(:,i);
        [a,index] = max(temp);
        temp(index) = 5;
%         second_extre= max(temp);
        
        X(:,i) = temp;
    end
    
    if i==32
        temp = X_train(:,i);
        [a,index] = max(temp);
        temp(index) = 5;
%         second_extre= max(temp);
        
        X(:,i) = temp;
    end    
    
    if i==42    
        temp = X_train(:,i);
        [a,index] = min(temp);
        temp(index) = 3;
        X(:,i) = temp;
    end   
    
    if i==42    
        temp = X_train(:,i);
        [a,index] = min(temp);
        temp(index) = 0;
        X(:,i) = temp;
    end
    if i==62    
        temp = X_train(:,i);
        [a,index] = min(temp);
        temp(index) = 1;
        X(:,i) = temp;
    end  
    
    if i==64    
        temp = X_train(:,i);
        [a,index] = min(temp);
        temp(index) = 2;
        X(:,i) = temp;
    end  
    
    if i==71    
        temp = X_train(:,i);
        [a,index] = min(temp);
        temp(index) = 2;
        X(:,i) = temp;
    end  
end

y = normalizeFeature(y_train);

% X = X.^5;
% for i=1:71
%      if(~isempty(find(non_norm == i)))
%          X(:,i) = normalizeFeature(X_train(:,i));
%      else
%          X(:,i) = normalizeFeature(X(:,i));
%      end
% end

X_expand = zeros(size(y_train,1),1);
seeds = 1:10;
for s = 1 : length(seeds)
setSeed(seeds(s));

% test on degree: manipulate the features
degree = 6:10;
for i_degree  = 1:length(degree)
for i=1:71
     if(~isempty(find(non_norm == i)))
%           append_cols = normalizeFeature(X_train(:,i));
        append_cols = dummy_encoding(X_train(:,i)');
          X_expand = [X_expand append_cols];
        % X_expand(:,size(X_expand,2)+1:size(append_cols,2)+size(X_expand,2)+1) = append_cols;
     elseif(i==12||i==48)
         append_cols = normalizeFeature(X_train(:,i));
         X_expand = [X_expand append_cols];
%          X_expand(:,size(X_expand,2)+1:size(append_cols,2)+size(X_expand,2)+1) = append_cols;
     else 
         append_cols = normalizeFeature(X_train(:,i));
         append_cols = mypoly(append_cols,degree(i_degree));
         X_expand = [X_expand append_cols];
     end
end
X = X_expand(:,2:size(X_expand,2));
fprintf('featue input %d',size(X,1));

% split data in K fold (we will only create indices)

prop = 0.8;
[XTr, yTr, XTe, yTe] = split(y, X, prop,1);
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
lambda = logspace(-1,10,20);

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
            fprintf('dg %d mean%dfold var: %.4f for lambda %.2f  ',degree(i_degree),k,lambda(i));
            
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
semilogx(degree,mseTr_degree(s,:),'b');hold on;
semilogx(degree,mseTe_degree(s,:),'r');hold on;
end
% legend('train','test');
% rmseTr_diffSeed(s) = rmseTr;
% rmseTe_diffSeed(s) = rmseTe;

% semilogx(lambda,mean(rmseTr_diffSeed),'b','linewidth',3);hold on;
% semilogx(lambda,rmseTe,'r','linewidth',3);hold on;


    % find which lambda is mot unsful
%     [mseTe_min,index] = min(rmseTe(s,:));
%     fprintf('lambda %.4f  croess validation: test error: %.4f trainig error: %.4f \n ', ...
%             lambda(index), mseTe_min,rmseTr(s,index));
%     beta = ridgeRegression(yTr,tXTr,lambda(index));
%     mesTest(s) = computeCost(yTe,tXTe,beta);
%     fprintf('test on validation set error: %.4f \n',mesTest(s))
    

% fprintf('for different seed, the mean test error is %.4f \n',mean(mesTest));
