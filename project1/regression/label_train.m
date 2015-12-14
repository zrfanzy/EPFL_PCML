load('Shanghai_regression.mat');
%labely = zeros(length(y_train), 1);

for i = 1 : length(y_train)
    if y_train(i) < 4000
        labely(i) = 1;
    elseif y_train(i) > 7600
        labely(i) = 3;
    else
        labely(i) = 2;
    end
end

x1 = find(labely == 1);
x1o = find(labely ~= 1);
x2 = find(labely == 2);
x2o = find(labely ~= 2);
x3 = find(labely == 3);
x3o = find(labely ~= 3);

%plot(X_train(x1,1), y_train(x1),'r*', X_train(x2, 1), y_train(x2), 'b*', X_train(x3, 1), y_train(x3), 'g*');

tX = [ones(length(y_train), 1) X_train];

% train label 1
%labely(x1o) = 0;
%labely(x1) = 1;
%beta1 = logisticRegression(labely, tX, 0.005);

% train label 2
%labely(x2) = 1;
%labely(x2o) = 0;
%beta2 = logisticRegression(labely, tX, 0.001);

% train label 3
%labely(x3) = 1;
%labely(x3o) = 0;
%beta3 = logisticRegression(labely, tX, 0.001);

load('reg.mat')

p1 = 1.0 ./ (1.0 + exp(-tX * beta1));
p2 = 1.0 ./ (1.0 + exp(-tX * beta2));
p3 = 1.0 ./ (1.0 + exp(-tX * beta3));
%zeros(length(y), 1);
lab = zeros(length(y_train), 1);
for i = 1 : length(y_train)
    if (p1(i) > p2(i) & p1(i) > p3(i))
        lab(i) = 1;
    elseif (p2(i) > p1(i) & p2(i) > p3(i))
        lab(i) = 2;
    else
        lab(i) = 3;
    end
end

x1 = find(labely == 1);
x2 = find(labely == 2);
x3 = find(labely == 3);

figure;
%plot(X_train(x1,1), y_train(x1),'r*', X_train(x2, 1), y_train(x2), 'b*', X_train(x3, 1), y_train(x3), 'g*');
plot(X_train(x1,12), X_train(x1,48),'r*', X_train(x2, 12), X_train(x2, 48), 'b*', X_train(x3, 12), X_train(x3, 48), 'g*');
