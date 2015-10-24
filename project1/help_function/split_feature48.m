function col12 = split_feature48(X)
col12 = zeros(size(X,1),2);
index1 = find(X>14);
col12(index1,1) = X(index1);
index2 = find(X<=14);
col12(index2,2) = X(index2);