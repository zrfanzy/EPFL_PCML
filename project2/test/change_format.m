% load origin prediction
load pred.mat;

% binary: if pred is 3(other) then Ytest is 0, otherwise 1
Ytest = ~(pred == 3);
Ytest = Ytest';
save('pred_binary', 'Ytest')

% multiclass: chaning from [0, 3] -> [1, 4]
Ytest = (pred + 1)';
save('pred_multiclass', 'Ytest')