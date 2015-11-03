function y = normalization_claasi(y_train)
y_values = unique(y_train);
y = zeros(size(y_train,1),1);

  index = find(y_train >0);

  y(index) = 1;% X_in(:,index);

end