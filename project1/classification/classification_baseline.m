% clear all
% ridgeRegression
load('Shanghai_classification.mat');

sigmoid = @(x) exp(x)./(1+exp(x));
%X = X_train;
y = normalization_claasi(y_train);
degree =[1/2,6];

for i = 1:length(degree)
X = mypoly(X_train,degree(i));
X = normalizeFeature(X);
for s = 1:10

prop = 0.8;
[XTr, yTr, XTe, yTe] = split_setseed(y, X, prop,s);
% for differnet train set selection:
% size1 = size(XTr,1)*0.1*p;
% [XTr, yTr, nump2, nump3] = split_setseed(yTr, XTr, 0.1*p,s);

tXTr = [ones(length(yTr), 1) XTr];
tXTe = [ones(length(yTe), 1) XTe]; 
alpha = 0.05;

K = 5;
N = size(yTr,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
	idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

% lambda values found for each degree
lambda = logspace(-3,1 ,6);
lambda = [0.0001,lambda];

    for l = 1:length(lambda)
    % K-fold cross validation for each lambda
        for k = 1:K
    %        get k'th subgroup in test, others in train
            idxTe = idxCV(k,:);
            idxTr = idxCV([1:k-1 k+1:end],:);
            idxTr = idxTr(:);
            yTe_cv = yTr(idxTe);            XTe_cv = XTr(idxTe,:);
            yTr_cv = yTr(idxTr);            XTr_cv = XTr(idxTr,:);
            tXTr_cv = [ones(length(yTr_cv), 1) XTr_cv];
            tXTe_cv = [ones(length(yTe_cv), 1) XTe_cv]; 

%             ridge regression    
            
            beta = penLogisticRegression(yTr_cv,tXTr_cv,alpha,lambda(l));
%              beta = logisticRegression(yTr_cv,tXTr_cv,alpha);
          
            mseTrSub(s,i,l,k) = computeCost_classi(yTr_cv, tXTr_cv, beta); 
            mseTeSub(s,i,l,k) = computeCost_classi(yTe_cv, tXTe_cv, beta);
         end
%             compute the mean error for k cross validation of the same lambda
%              fprintf('%dfold varErrTe: %.4f for lambda %.2f  ',k,var(mseTeSub),lambda(l));
%             
             rmseTr_lamb(s,i,l) = mean(mseTrSub(s,i,l,:));
             rmseTe_lamb(s,i,l) = mean(mseTeSub(s,i,l,:));
             fprintf('tr %.4f te %.4f\n ',rmseTr_lamb(s,i,l),rmseTe_lamb(s,i,l));
%             box(:,i) = mseTeSub;
     end % end of runing for different lambda
    
[numb,index_best_lambda] = min(rmseTe_lamb(s,i,:));

beta = penLogisticRegression(yTr,tXTr,alpha,lambda(index_best_lambda));
mseTr(s,i) = computeCost_classi(yTr, tXTr, beta)/length(yTr);
mseTe(s,i) = computeCost_classi(yTe, tXTe, beta)/length(yTe);

end % end of seeds

end % end of degree


%% some plotting
% figure;box1 = [];box2 = [];for i = 1:10
% box1 = [box1 mseTe(:,i)/((10-i)*150)];box2 = [box2 mseTr(:,i)/(150*i)];
% end
% boxplot(box1);figure;boxplot(box2);
% box1 = [];box2 = [];for i = 1:50
% box1 = [box1 mseTe(i,i)];box2 = [box2 mseTr(:,i)];
% end
% boxplot(box1);figure;boxplot(box2);
% figure;subplot(211); boxplot(box1);hy = ylabel('test error');set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
% set(hy,'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);grid on;subplot 212;boxplot(box2);hy = ylabel('test error');grid on;hx = xlabel('portion');set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
% set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
% print -dpdf 'image/claas_learning.pdf'
% % figure;
% for s1 = 1:s
%     scatter(degree, lambda_store(s1,:),'r');hold on;
%     title('lambda');
% %     plot(degree, lambda_store(s1,:),'b');
% end
% 
% figure;
% for s1 = 1:s
% %     scatter(degree, lambda_store(s1,:),'r');
%     title('degree');
%      plot(degree, mseTr(s1,:)/length(yTr),'color',[1 0.7 0.7]);hold on;
%      plot(degree, mseTe(s1,:)/length(yTe),'color',[0.7 0.7 1]);
% end
% plot(degree,mean(mseTr(:,degree)),'-b');hold on;
% plot(degree,mean(mseTe(:,degree)),'-r');hold off;
% % fprintf('tr %.4f te %.4f and lambda%.3f \n',mseTr,mseTe,lambda(index_best_lambda));