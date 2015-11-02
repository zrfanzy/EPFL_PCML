tX = [ones(length(X_test), 1) X_test];
originX = normalizeFeaturebyOther(X_train, X_test);

p1 = 1.0 ./ (1.0 + exp(-tX * beta1));
p2 = 1.0 ./ (1.0 + exp(-tX * beta2));
p3 = 1.0 ./ (1.0 + exp(-tX * beta3));
%zeros(length(y), 1);

for i = 1 : length(X_test)
    if (p1(i) > p2(i) & p1(i) > p3(i))
        tx = [1 mypoly(originX(i,:), degreesetting(1))];
        outY(i) = tx * regBeta1;
    elseif (p2(i) > p1(i) & p2(i) > p3(i))
        tx = [1 mypoly(originX(i,:), degreesetting(2))];
        outY(i)=  tx * regBeta2;
    else
        tx = [1 mypoly(originX(i,:), degreesetting(3))];
        outY(i) =  tx * regBeta3;
    end
end