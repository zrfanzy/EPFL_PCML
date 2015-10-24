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
 non_norm = [12,13,16,17,18,19,20,22,35,36,47,49,52,54];
 
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
% first col is dummy
X_expand = zeros(size(y_train,1),1);

% test on degree: manipulate the features
% degree = 1;
% for i_degree  = 1:length(degree)
for i=1:71
     if(~isempty(find(non_norm == i)))
          append_cols = dummy_encoding(X_train(:,i)');
          X_expand = [X_expand append_cols];
        % X_expand(:,size(X_expand,2)+1:size(append_cols,2)+size(X_expand,2)+1) = append_cols;
     elseif(i==12||i==48)
         append_cols = normalizeFeature(X_train(:,i));
         X_expand = [X_expand append_cols];
%          X_expand(:,size(X_expand,2)+1:size(append_cols,2)+size(X_expand,2)+1) = append_cols;
     else 
         append_cols = normalizeFeature(X_train(:,i));
         append_cols = mypoly(append_cols,1); % in base line degree = 1
         X_expand = [X_expand append_cols];
     end
end
X = X_expand(:,2:size(X_expand,2));













lambda = logspace(-3,2,10);
for i = 1:length(lambda)
for s = 1
% split data in K fold (we will only create indices)
setSeed(s);
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

% lambda values (INSERT CODE)
% K-fold cross validation

  lambda = logspace(1,6 ,10);
%   lambda = 0;
%lambda = 50;
% for i = 1:length(lambda)
    
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
        rmseTr(s,i) = mean(mseTrSub);
        rmseTe(s,i) = mean(mseTeSub);
%         box(:,i) = mseTeSub;
end % end of runing for different lambda

    % find which lambda is mot unsful
    [mseTe_min,index] = min(rmseTe(s,:));
    fprintf('lambda %.4f  croess validation: test error: %.4f trainig error: %.4f \n ', ...
            lambda(index), mseTe_min,rmseTr(s,index));
    beta = ridgeRegression(yTr,tXTr,lambda(index));
    mesTest(s) = computeCost(yTe,tXTe,beta);
    fprintf('test on validation set error: %.4f \n',mesTest(s))
    
end 
fprintf('for different seed, the mean test error is %.4f \n',mean(mesTest));
%     figure
% % boxplot(box,lambda);
% % xlabel('lambda');
% % ylabel('test error');
% % title('show the variance between different fold');
% 
% % compute expected train and test error
% rmseTr_mean = mean(rmseTr);
% rmseTe_mean = mean(rmseTe);
% 
% subplot(2,1,1);
% plot(lambda, rmseTe,'r-', 'color', [1 0.7 0.7]);
% hold on;
% plot(lambda, rmseTr, 'b-', 'color', [0.7 0.7 1]);
% hold on;
% legend('test','train');
% 
% plot(lambda, rmseTe_mean, 'r-', 'linewidth', 1);
% hold on;
% plot(lambda, rmseTr_mean,'b-', 'linewidth',1);
% xlabel('lambda');
% ylabel('error');
% 
% subplot(2,1,2);
% boxplot(rmseTe,'boxstyle','filled');
% ylim('auto');
% 
% 
% 
% %     boxp(i) = mseTe_lam;
% %     subplot(121);
% %     semilogx(lambda,mseTr_lam);
% %     hold on;
% %     semilogx(lambda,mseTe_lam);     
% %     
% %     if(k~=K)
% %         hold on;
% %     end
% %     
% %     xlabel('lambda ');
% %     legend(['training e degree ',degree(i)] ,['test e degree ',degree(i)]);
