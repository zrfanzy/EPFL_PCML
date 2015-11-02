load('Shanghai_regression.mat');
labely = zeros(length(y_train), 1);

origin_ytrain = y_train;

for i = 1 : length(y_train)
    if y_train(i) < 4000
        labely(i) = 1;
    elseif y_train(i) > 7600
        labely(i) = 3;
    else
        labely(i) = 2;
    end
end

originX = normalizeFeature(X_train);
X = normalizeFeature(X_train);

% find outlier with easy leastSquares & remove
% figure;
for label = 1 : 3
    ind = find(labely == label);
    N = length(y_train(ind));
    tX = [ones(N, 1) X(ind)];
    beta = leastSquares(y_train(ind), tX);
    
    tY = tX * beta;
    % histogram errors by leastsquares
    %subplot(3, 1, label);
    %hist(abs(tY - y_train(ind)), 100);
    
    % remove outliers
    for j = length(ind):-1:1
        if abs(tY(j) - y_train(ind(j))) > 2000
            X(ind(j),:)=[];
            y_train(ind(j),:)=[];
            labely(ind(j)) = [];
        end
    end
end

lamdasetting = [0.18, 0.14, 0.09];
degreesetting = [3, 3, 2];

sumerror = 0;

for label = 1 : 3 
    trerror = 0;
    teerror = 0;
    ind = find(labely == label);
    thisY = y_train(ind);
    N = length(thisY)

    seeds = 1:100;
    for s = 1 : length(seeds)
        setSeed(seeds(s));
        idx = randperm(length(thisY));
        
    tX = [ones(N, 1) mypoly(X(ind,:), degreesetting(label))];
    
    %{
    
    if (label == 1)
        regBeta1 = ridgeRegression(thisY, tX, lamdasetting(label));
    elseif label == 2
        regBeta2 = ridgeRegression(thisY, tX, lamdasetting(label));
    else
        regBeta3 = ridgeRegression(thisY, tX, lamdasetting(label));
    end
    %}
    
    
        prop = 0.8;
        [XTr, yTr, XTe, yTe] = split(thisY(idx), tX(idx,:), prop);
        tXTr = XTr;
        tXTe = XTe;
        beta = ridgeRegression(yTr,tXTr,lamdasetting(label));
        trerror = trerror + computeCost(yTr, tXTr, beta);
        teerror = teerror + computeCost(yTe, tXTe, beta);
    end
    fprintf('test: %.4f; train:%.4f \n',...
            teerror / length(seeds), trerror / length(seeds)); 
    sumerror = sumerror + teerror /length(seeds) * N
end
csvwrite('test_errors_regression.csv',['rms' sumerror / length(y_train)])
%{

tX = [ones(length(origin_ytrain), 1) X_train];

p1 = 1.0 ./ (1.0 + exp(-tX * beta1));
p2 = 1.0 ./ (1.0 + exp(-tX * beta2));
p3 = 1.0 ./ (1.0 + exp(-tX * beta3));
%zeros(length(y), 1);
lab = zeros(length(origin_ytrain), 1);
clear outy;
clear errorsum;
errorsum = 0;
for i = 1 : length(origin_ytrain)
    if (p1(i) > p2(i) & p1(i) > p3(i))
        lab(i) = 1;
        tx = [1 mypoly(originX(i,:), degreesetting(1))];
        outy = tx * regBeta1;
    elseif (p2(i) > p1(i) & p2(i) > p3(i))
        lab(i) = 2;
        tx = [1 mypoly(originX(i,:), degreesetting(2))];
        outy =  tx * regBeta2;
    else
        lab(i) = 3;
        tx = [1 mypoly(originX(i,:), degreesetting(3))];
        outy =  tx * regBeta3;
    end
    errorsum = errorsum + (origin_ytrain(i) - outy)*(origin_ytrain(i) - outy);
end
errorsum

%}
