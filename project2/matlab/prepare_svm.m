clear;
load ../train.mat

% load relabeled_y
load modified_y
train.y = modified_y;

% parameters for svm training
lambda = 0.0001;
maxIter = 1000000000;

% using 4/5 for training (1/4 for validation)
range = 1:4800;%[1:1200 2401:6000];
y = train.y(range)';
x = train.X_cnn(range, :)';

% find 4 labels
tag1 = find(y == 1);
tag2 = find(y == 2);
tag3 = find(y == 3);
tag4 = find(y == 4);
tags = {tag1, tag2, tag3, tag4};

% clear temp-use variables
clear tag1 tag2 tag3 tag4;

wova = [];
bova = [];
infoova = [];
% train 4 1vall model
for i = 1 : 4
    Ty = zeros(1, size(y, 2));
    for j = 1 : size(Ty, 2)
        if y(j) == i
            Ty(j) = 1;
        else
            Ty(j) = -1;
        end
    end
    [w b info] = vl_svmtrain(x, Ty, lambda, 'MaxNumIterations', maxIter);
    wova = [wova, w];
    bova = [bova, b];
    infoova = [infoova, info];
end

wovo = [];
bovo = [];
infoovo = [];
vslist = [];

% train 4 1v1 model
for i = 1 : 3
    for j = (i + 1) : 4
        
        % i VS j model
        Tx = x(:, [tags{i},tags{j}]);
        Ty1 = y([tags{i},tags{j}]);
        Ty = zeros(1, size(Ty1, 2));
        for k = 1 : size(Ty1,2)
            if Ty1(k) == 1
                Ty(k) = i;
            else
                Ty(k) = j;
            end
        end
        [w, b, info] = vl_svmtrain(Tx, Ty, lambda, 'MaxNumIterations', maxIter);
        wovo = [wovo, w];
        bovo = [bovo, b];
        infoovo = [infoovo, info];
        vslist = [vslist, [i, j]'];
    end
end

% clear temp-use variables
clear Tx Ty Ty1 w b info i j k tags x y range modified_y
clear lambda maxIter