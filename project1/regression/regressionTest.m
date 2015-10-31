load('Shanghai_regression.mat');
labely = zeros(length(y_train), 1);

for i = 1 : length(y_train)
    if y_train(i) < 4000
        labely(i) = 1;
    elseif y_train(i) > 7600
        labely(i) = 3;
    else
        labely(i) = 2;
    end
end


X = normalizeFeature(X_train);

% find outlier with easy leastSquares & remove
figure;
for label = 1 : 3
    ind = find(labely == label);
    N = length(y_train(ind));
    tX = [ones(N, 1) X(ind)];
    beta = leastSquares(y_train(ind), tX);
    
    tY = tX * beta;
    % histogram errors by leastsquares
    subplot(3, 1, label);
    hist(abs(tY - y_train(ind)), 100);
    
    % remove outliers
    %for j = length(ind):-1:1
    %    if abs(tY(j) - y_train(ind(j))) > 2000
    %        X(ind(j),:)=[];
    %        y_train(ind(j),:)=[];
    %        labely(ind(j)) = [];
    %    end
    %end
end

% train for each label
for label = 1 : 3 
    figure;
    clear idxCV;
    clear mseTr_degree;
    clear mseTe_degree;
    
    ind = find(labely == label);
    thisY = y_train(ind);
    N = length(thisY);

    seeds = 1:10;
    for s = 1 : length(seeds)
        setSeed(seeds(s));

        % degrees
        for degree  = 1:9
            tX = [ones(N, 1) mypoly(X(ind), degree)];

            % K fold
            K = 5;
            Nk = floor(N/K);
            idx = randperm(Nk * K);

            for k = 1:K
                idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
            end

            lambda = logspace(-5,15,10);
            lambda = [0,lambda];
            
            for i = 1:length(lambda)
                % K-fold cross validation for each lambda
                for k = 1:K
                % get k'th subgroup in test, others in train
                    idxTe = idxCV(k,:);
                    idxTr = idxCV([1:k-1 k+1:end],:);
                    idxTr = idxTr(:);

                    yTe_cv = thisY(idxTe);
                    XTe_cv = tX(idxTe,:);

                    yTr_cv = thisY(idxTr);
                    XTr_cv = tX(idxTr,:);

                    tXTr_cv = [XTr_cv];
                    tXTe_cv = [XTe_cv]; 

                    % ridge regression    

                    beta = ridgeRegression(yTr_cv,tXTr_cv,lambda(i));

                    mseTrSub(k) = computeCost(yTr_cv, tXTr_cv, beta); 
                    mseTeSub(k) = computeCost(yTe_cv, tXTe_cv, beta);
                end
                % compute the mean error for k cross validation of the same lambda
                fprintf('dg %d mean%dfold varErrTe: %.4f for lambda %.2f  ',degree,k,var(mseTeSub),lambda(i));

                rmseTr_lamb(label, i) = mean(mseTrSub);
                rmseTe_lamb(label, i) = mean(mseTeSub);
                fprintf('tr %.4f te %.4f\n ',rmseTr_lamb(label, i),rmseTe_lamb(label, i));
            %         box(:,i) = mseTeSub;
            end % end of runing for different lambda

            [numb(label),index_best_lambda(label)] = min(rmseTe_lamb(label));
            % by cv find lambda, use the lambda to train and test, get the performance
            % of the degree(diff model complexity)
            prop = 0.7;
            [XTr, yTr, XTe, yTe] = split(y_train(ind), tX, prop);
            tXTr = [ones(length(yTr), 1) XTr];
            tXTe = [ones(length(yTe), 1) XTe]; 
            beta = ridgeRegression(yTr,tXTr,lambda(index_best_lambda(label)));
            if label == 1
                ridgebeta1 = beta;
                ridgelambda1 = lambda(index_best_lambda(label));
            elseif label == 2
                ridgebeta2 = beta;
                ridgelambda2 = lambda(index_best_lambda(label));
            else
                ridgebeta3 = beta;
                ridgelambda3 = lambda(index_best_lambda(label));
            end
            mseTr_degree(s,degree) = computeCost(yTr, tXTr, beta);
            mseTe_degree(s,degree) = computeCost(yTe, tXTe, beta);

            fprintf('\ndegree %d ; lambda %f;  test: %.4f; train:%.4f \n',...
            degree,lambda(index_best_lambda),...
            mseTe_degree(s,degree),mseTr_degree(s,degree));
        end

    end% for different seed, repeated again

    for s = 1:length(seeds)
        plot(1:9,mseTr_degree(s,:),'b');hold on;
        plot(1:9,mseTe_degree(s,:),'r');hold on;grid on;
    end
    
end
